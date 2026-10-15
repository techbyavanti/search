#!/usr/bin/env python3
"""Trains the neural reranker's small MLP with a LambdaRank objective and exports it to ONNX.

Run via ../scripts/train-neural-reranker.sh, which first runs the Java
RerankFeatureExportCli to write models/neural-reranker/training-features.tsv:
one row per (query, hybrid candidate), with the exact feature vector
NeuralRerankStrategy serves (RerankFeatureBuilder.java is the only feature
implementation, so training and serving cannot drift apart) and the WANDS
grade (0 for unlabeled candidates, as in the eval).

Training uses only the "train" query split. 15% of those queries are kept
aside for early stopping and model selection, so the held-out queries
EvalCli scores are never used to choose anything.

Loss: LambdaRank. For each pair of candidates in one query's pool where one
has a higher grade, the pairwise logistic loss is weighted by how much
swapping the two would change that query's nDCG. This optimizes ranking
order directly, unlike a pointwise regression on the grade (kept as
--loss mse for comparison).

Feature standardization (mean/std from the training rows) is part of the
exported graph, so the Java side feeds raw features.

Writes:
  - models/neural-reranker/model.onnx (input "features" [batch, 17] -> output "score" [batch, 1])
  - models/neural-reranker/training-metadata.json
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

from query_split import split_query_ids

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_MODEL_DIR = ROOT_DIR / "models" / "neural-reranker"
FEATURES_FILE = OUTPUT_MODEL_DIR / "training-features.tsv"
HELD_OUT_QUERIES_FILE = ROOT_DIR / "search-eval" / "src" / "main" / "resources" / "neural-reranker-eval-queries.txt"

GRADE_BY_LABEL = {"exact": 2, "partial": 1, "irrelevant": 0}
NDCG_K = 10
VALIDATION_FRACTION = 0.15
SEED = 42
META_COLUMNS = ["query_id", "split", "product_id", "grade"]


class QueryPool:
    """One query's candidate pool, in hybrid order."""

    def __init__(self, query_id, features, grades, ideal_dcg):
        self.query_id = query_id
        self.features = torch.from_numpy(features)
        self.grades = torch.from_numpy(grades)
        self.grades_np = grades
        self.ideal_dcg = ideal_dcg


def load_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def dcg(grades, k):
    grades = np.asarray(grades[:k], dtype=np.float64)
    return float(np.sum((2.0 ** grades - 1.0) / np.log2(np.arange(len(grades)) + 2.0)))


def ideal_dcg_by_query():
    """Ideal DCG@10 from *all* of a query's judgments, as MetricsCalculator.ndcgAtK computes it."""
    judged = {}
    for row in load_tsv(DATASET_DIR / "label.csv"):
        judged.setdefault(row["query_id"], []).append(GRADE_BY_LABEL[row["label"].strip().lower()])
    return {qid: dcg(sorted(g, reverse=True), NDCG_K) for qid, g in judged.items()}


def load_pools():
    with open(FEATURES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        feature_names = header[len(META_COLUMNS):]
        rows_by_query, split_by_query = {}, {}
        for row in reader:
            rows_by_query.setdefault(row[0], []).append(row)
            split_by_query[row[0]] = row[1]

    ideal = ideal_dcg_by_query()
    pools = {}
    for qid, rows in rows_by_query.items():
        features = np.array([[float(v) for v in r[len(META_COLUMNS):]] for r in rows], dtype=np.float32)
        grades = np.array([float(r[3]) for r in rows], dtype=np.float32)
        pools[qid] = QueryPool(qid, features, grades, ideal.get(qid, 0.0))
    return feature_names, pools, split_by_query


def ndcg_at_k(scores, pool):
    if pool.ideal_dcg == 0.0:
        return 0.0
    order = np.argsort(-scores, kind="stable")  # ties keep hybrid order, like Java's stable sort
    return dcg(pool.grades_np[order], NDCG_K) / pool.ideal_dcg


def mean_ndcg(model, pools):
    model.eval()
    with torch.no_grad():
        values = [ndcg_at_k(model(p.features).squeeze(1).numpy(), p) for p in pools]
    return float(np.mean(values))


def hybrid_ndcg(pools):
    """nDCG@10 of the pool's own order, i.e. Hybrid without reranking."""
    return float(np.mean([ndcg_at_k(-np.arange(len(p.grades_np), dtype=np.float32), p) for p in pools]))


class RerankerNet(nn.Module):
    def __init__(self, feature_count, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(mean))
        self.register_buffer("std", torch.from_numpy(std))
        self.net = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features):
        return self.net((features - self.mean) / self.std)


def lambdarank_loss(scores, grades, ideal_dcg):
    """Sum over pairs (i more relevant than j) of |ΔnDCG_ij| * log(1 + exp(-(s_i - s_j)))."""
    n = scores.shape[0]
    order = torch.argsort(scores.detach(), descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n)
    discounts = 1.0 / torch.log2(ranks.float() + 2.0)
    gains = 2.0 ** grades - 1.0

    more_relevant = (grades[:, None] - grades[None, :]) > 0
    delta_ndcg = torch.abs((gains[:, None] - gains[None, :]) * (discounts[:, None] - discounts[None, :])) / ideal_dcg
    pair_loss = F.softplus(-(scores[:, None] - scores[None, :]))
    return (delta_ndcg * pair_loss)[more_relevant].sum()


def train(train_pools, validation_pools, feature_count, mean, std, loss_name, epochs, patience,
          queries_per_step=16, lr=1e-3, weight_decay=1e-4):
    torch.manual_seed(SEED)
    model = RerankerNet(feature_count, mean, std)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainable = [p for p in train_pools if p.ideal_dcg > 0]
    rng = random.Random(SEED)

    best_ndcg, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(trainable)
        total = 0.0
        for start in range(0, len(trainable), queries_per_step):
            optimizer.zero_grad()
            batch = trainable[start:start + queries_per_step]
            loss = 0.0
            for pool in batch:
                scores = model(pool.features).squeeze(1)
                if loss_name == "lambdarank":
                    loss = loss + lambdarank_loss(scores, pool.grades, pool.ideal_dcg)
                else:
                    loss = loss + F.mse_loss(scores, pool.grades, reduction="sum") / len(pool.grades_np)
            loss = loss / len(batch)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)

        validation_ndcg = mean_ndcg(model, validation_pools)
        if validation_ndcg > best_ndcg:
            best_ndcg, best_epoch = validation_ndcg, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d}: loss={total / len(trainable):.4f} validation nDCG@10={validation_ndcg:.4f}"
                  f" (best {best_ndcg:.4f} @ {best_epoch})")
        if epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch}; best validation nDCG@10 {best_ndcg:.4f} at epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, best_ndcg, best_epoch


def permutation_importance(model, pools, feature_names, repeats=5):
    """Drop in nDCG@10 when one feature's values are shuffled across all candidates."""
    base = mean_ndcg(model, pools)
    stacked = torch.cat([p.features for p in pools])
    sizes = [len(p.grades_np) for p in pools]
    rng = np.random.default_rng(SEED)
    drops = {}
    for j, name in enumerate(feature_names):
        values = []
        for _ in range(repeats):
            shuffled = stacked.clone()
            shuffled[:, j] = shuffled[rng.permutation(len(shuffled)), j]
            parts = torch.split(shuffled, sizes)
            with torch.no_grad():
                values.append(np.mean([ndcg_at_k(model(x).squeeze(1).numpy(), p) for x, p in zip(parts, pools)]))
        drops[name] = base - float(np.mean(values))
    return drops


def export_onnx(model, feature_count, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.zeros(1, feature_count, dtype=torch.float32), str(output_path),
        input_names=["features"], output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the newer dynamo-based exporter needs the optional onnxscript package; the legacy exporter doesn't
    )
    onnx.checker.check_model(str(output_path))


def verify_export(model, onnx_path, sample_features):
    """Checks the exported ONNX model against the in-memory torch model on real feature vectors."""
    session = ort.InferenceSession(str(onnx_path))
    with torch.no_grad():
        torch_scores = model(torch.from_numpy(sample_features)).numpy().flatten()
    onnx_scores = session.run(["score"], {"features": sample_features})[0].flatten()
    max_diff = float(np.max(np.abs(torch_scores - onnx_scores)))
    print(f"  max |torch - onnx| score diff on {len(sample_features)} samples: {max_diff:.2e}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX export mismatch too large: {max_diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", choices=["lambdarank", "mse"], default="lambdarank")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_MODEL_DIR)
    args = parser.parse_args()

    print(f"Loading {FEATURES_FILE.relative_to(ROOT_DIR)}...")
    feature_names, pools, split_by_query = load_pools()

    expected_train, expected_held_out = split_query_ids({qid for qid in split_by_query})
    held_out_file = set(HELD_OUT_QUERIES_FILE.read_text().split())
    exported_held_out = {qid for qid, s in split_by_query.items() if s == "held_out"}
    if not (held_out_file == exported_held_out == set(expected_held_out)):
        raise RuntimeError("Held-out split mismatch between query_split.py, the eval resource, and the feature export")

    train_ids = sorted((qid for qid, s in split_by_query.items() if s == "train"), key=int)
    random.Random(SEED).shuffle(train_ids)
    validation_count = int(len(train_ids) * VALIDATION_FRACTION)
    validation_pools = [pools[q] for q in train_ids[:validation_count]]
    fit_pools = [pools[q] for q in train_ids[validation_count:]]
    held_out_pools = [pools[q] for q in sorted(exported_held_out, key=int)]
    print(f"  {len(feature_names)} features; queries: {len(fit_pools)} fit, {len(validation_pools)} validation,"
          f" {len(held_out_pools)} held-out; {sum(len(p.grades_np) for p in fit_pools)} fit rows")

    fit_features = np.concatenate([p.features.numpy() for p in fit_pools])
    mean = fit_features.mean(axis=0).astype(np.float32)
    std = fit_features.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0

    print(f"Training ({args.loss})...")
    model, validation_ndcg, best_epoch = train(fit_pools, validation_pools, len(feature_names), mean, std,
                                               args.loss, args.epochs, args.patience)

    print("nDCG@10 (pool-only, same formula as the Java eval):")
    print(f"  validation: hybrid order {hybrid_ndcg(validation_pools):.4f} -> reranked {validation_ndcg:.4f}")
    held_out_ndcg = mean_ndcg(model, held_out_pools)
    print(f"  held-out:   hybrid order {hybrid_ndcg(held_out_pools):.4f} -> reranked {held_out_ndcg:.4f}"
          " (reported only; not used for any choice)")

    print("Permutation importance on validation queries (nDCG@10 drop):")
    importance = permutation_importance(model, validation_pools, feature_names)
    for name, drop in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"  {name:28s} {drop:+.4f}")

    onnx_path = args.output_dir / "model.onnx"
    print(f"Exporting to {onnx_path}...")
    export_onnx(model, len(feature_names), onnx_path)
    verify_export(model, onnx_path, fit_features[:64])

    metadata_path = args.output_dir / "training-metadata.json"
    metadata_path.write_text(json.dumps({
        "loss": args.loss,
        "feature_names": feature_names,
        "architecture": f"Linear({len(feature_names)}, 32) -> ReLU -> Linear(32, 16) -> ReLU -> Linear(16, 1)",
        "fit_query_count": len(fit_pools),
        "validation_query_count": len(validation_pools),
        "held_out_query_count": len(held_out_pools),
        "fit_row_count": int(fit_features.shape[0]),
        "best_epoch": best_epoch,
        "validation_ndcg_at_10": round(validation_ndcg, 4),
        "validation_hybrid_ndcg_at_10": round(hybrid_ndcg(validation_pools), 4),
        "permutation_importance": {k: round(v, 4) for k, v in importance.items()},
    }, indent=2) + "\n")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
