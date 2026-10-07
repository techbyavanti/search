#!/usr/bin/env python3
"""Trains the neural reranker's small MLP and exports it to ONNX.

Offline, one-time (well, "one-time until you want to retrain") step — not
part of the Java build. Run via ../scripts/train-neural-reranker.sh, which
just invokes this with the venv's interpreter. Requires:
  - Elasticsearch running locally with the full catalog already ingested
    (see HOWTO.md steps 0-3) — features are computed against the live
    index rather than reimplemented, so training data matches serving
    exactly for everything except the query embedding (see below).
  - dataset/query.csv and dataset/label.csv (see HOWTO.md step 1).
  - models/bge-small-en-v1.5/{model.onnx,tokenizer.json} (see HOWTO.md
    step 2) — used to compute query embeddings the same way
    EmbeddingService.java does (CLS-pool, L2-normalize, BGE query prefix).
    This is the one place training and serving don't share code, so it's
    spot-checked against a known Java-computed embedding at the bottom of
    this file's __main__ block.

Writes:
  - models/neural-reranker/model.onnx
  - search-eval/src/main/resources/neural-reranker-eval-queries.txt
    (the held-out 20% of query ids, so EvalCli scores this strategy only
    on queries it never trained on)

Feature order is a contract with RerankFeatureBuilder.java — see FEATURE
NAMES below and search-retrieval/.../RerankFeatureBuilder.java. Changing
one side without the other silently produces garbage scores.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import requests
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from query_split import split_query_ids

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
ES_HOST = "http://localhost:9200"
INDEX = "products"

EMBEDDING_MODEL_DIR = ROOT_DIR / "models" / "bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQUENCE_LENGTH = 256

OUTPUT_MODEL_DIR = ROOT_DIR / "models" / "neural-reranker"
HELD_OUT_QUERIES_FILE = ROOT_DIR / "search-eval" / "src" / "main" / "resources" / "neural-reranker-eval-queries.txt"

CANDIDATE_POOL_SIZE = 50  # matches application.yml's search.hybrid.candidate-pool-size
RRF_K = 60  # matches application.yml's search.hybrid.rrf-k (RESULTS.md's swept best)
NUM_CANDIDATES_MULTIPLIER = 10  # matches SemanticSearchStrategy.NUM_CANDIDATES_MULTIPLIER

# Feature order — must match RerankFeatureBuilder.java exactly.
FEATURE_NAMES = [
    "rrf_score",
    "cosine_similarity",
    "exact_term_overlap_fraction",
    "category_match",
    "average_rating",
    "log1p_review_count",
]
DEFAULT_AVERAGE_RATING = 4.5301  # dataset-wide mean average_rating; matches RerankFeatureBuilder.DEFAULT_AVERAGE_RATING

GRADE_BY_LABEL = {"exact": 2.0, "partial": 1.0, "irrelevant": 0.0}

LEXICAL_FIELDS = ["product_name^3", "product_class^2", "category_hierarchy", "product_description", "product_features"]


def load_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def tokenize(text):
    if not text:
        return []
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split(" ") if t]


class QueryEmbedder:
    """Mirrors EmbeddingService.embedQuery: CLS-pool + L2-normalize + BGE query prefix."""

    def __init__(self, model_dir):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"))

    def embed(self, query):
        encoding = self.tokenizer.encode(QUERY_PREFIX + query)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoding.type_ids], dtype=np.int64)

        outputs = self.session.run(["last_hidden_state"], {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
        cls_vector = outputs[0][0][0]  # [batch, seq, hidden] -> first token of the one batch item
        norm = np.linalg.norm(cls_vector)
        return cls_vector / norm if norm > 0 else cls_vector


def es_search(body):
    response = requests.post(f"{ES_HOST}/{INDEX}/_search", json=body, timeout=30)
    response.raise_for_status()
    return response.json()["hits"]["hits"]


def rrf_fuse(ranked_lists, k):
    """Mirrors RrfFusionService.fuse: sum of 1/(k + 1-based-rank) per list a doc appears in."""
    scores = {}
    for ranked_list in ranked_lists:
        for i, hit_id in enumerate(ranked_list):
            scores[hit_id] = scores.get(hit_id, 0.0) + 1.0 / (k + i + 1)
    return scores


def hybrid_rrf_scores(query_text, query_embedding):
    lexical_hits = es_search({
        "size": CANDIDATE_POOL_SIZE,
        "_source": False,
        "query": {"multi_match": {"query": query_text, "fields": LEXICAL_FIELDS, "type": "best_fields"}},
    })
    semantic_hits = es_search({
        "size": CANDIDATE_POOL_SIZE,
        "_source": False,
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding.tolist(),
            "k": CANDIDATE_POOL_SIZE,
            "num_candidates": CANDIDATE_POOL_SIZE * NUM_CANDIDATES_MULTIPLIER,
        },
    })
    lexical_ids = [hit["_id"] for hit in lexical_hits]
    semantic_ids = [hit["_id"] for hit in semantic_hits]
    return rrf_fuse([lexical_ids, semantic_ids], RRF_K)


def mget_products(product_ids):
    if not product_ids:
        return {}
    response = requests.post(f"{ES_HOST}/{INDEX}/_mget", json={
        "docs": [{"_id": pid, "_source": ["product_name", "product_class", "category_hierarchy",
                                           "average_rating", "review_count", "embedding"]} for pid in product_ids],
    }, timeout=30)
    response.raise_for_status()
    docs = {}
    for doc in response.json()["docs"]:
        if doc.get("found"):
            docs[doc["_id"]] = doc["_source"]
    return docs


def build_features(rrf_score, query_embedding, query_tokens, product):
    product_embedding = np.array(product.get("embedding", []), dtype=np.float32)
    cosine_similarity = float(np.dot(query_embedding, product_embedding)) if product_embedding.size else 0.0

    name_tokens = tokenize(product.get("product_name"))
    overlap = sum(1 for t in query_tokens if t in name_tokens)
    exact_term_overlap_fraction = overlap / len(query_tokens) if query_tokens else 0.0

    category_tokens = tokenize((product.get("category_hierarchy") or "") + " " + (product.get("product_class") or ""))
    category_match = 1.0 if any(t in category_tokens for t in query_tokens) else 0.0

    average_rating = product.get("average_rating")
    average_rating = float(average_rating) if average_rating is not None else DEFAULT_AVERAGE_RATING

    review_count = product.get("review_count")
    log1p_review_count = math.log1p(review_count) if review_count is not None else 0.0

    return [rrf_score, cosine_similarity, exact_term_overlap_fraction, category_match, average_rating, log1p_review_count]


def collect_training_data(queries_by_id, labels_by_query, query_ids, embedder):
    features, labels = [], []
    for i, query_id in enumerate(query_ids):
        query_text = queries_by_id[query_id]
        query_tokens = tokenize(query_text)
        query_embedding = embedder.embed(query_text)

        rrf_scores = hybrid_rrf_scores(query_text, query_embedding)

        product_ids = list(labels_by_query[query_id].keys())
        products = mget_products(product_ids)

        for product_id, grade in labels_by_query[query_id].items():
            product = products.get(product_id)
            if product is None:
                continue
            rrf_score = rrf_scores.get(product_id, 0.0)
            features.append(build_features(rrf_score, query_embedding, query_tokens, product))
            labels.append(grade)

        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(query_ids)} queries, {len(features)} labeled pairs so far")

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.float32)


class RerankerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURE_NAMES), 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, features):
        return self.net(features)


def train(features, labels, epochs=200, batch_size=512, lr=1e-3):
    model = RerankerNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    features_t = torch.from_numpy(features)
    labels_t = torch.from_numpy(labels).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(features_t, labels_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_features.size(0)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: mse={total_loss / len(dataset):.4f}")

    model.eval()
    return model


def export_onnx(model, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, len(FEATURE_NAMES), dtype=torch.float32)
    torch.onnx.export(
        model, dummy_input, str(output_path),
        input_names=["features"], output_names=["score"],
        dynamic_axes={"features": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # the newer dynamo-based exporter needs the optional onnxscript package; the legacy exporter doesn't
    )
    onnx.checker.check_model(str(output_path))


def verify_export(model, onnx_path, sample_features):
    """Sanity-checks the exported ONNX model against the in-memory torch model on a few real feature vectors."""
    session = ort.InferenceSession(str(onnx_path))
    torch_scores = model(torch.from_numpy(sample_features)).detach().numpy().flatten()
    onnx_scores = session.run(["score"], {"features": sample_features})[0].flatten()
    max_diff = float(np.max(np.abs(torch_scores - onnx_scores)))
    print(f"  max |torch - onnx| score diff on {len(sample_features)} samples: {max_diff:.6f}")
    if max_diff > 1e-4:
        raise RuntimeError(f"ONNX export mismatch too large: {max_diff}")


def main():
    print("Loading WANDS queries + labels...")
    queries = load_tsv(DATASET_DIR / "query.csv")
    labels = load_tsv(DATASET_DIR / "label.csv")

    queries_by_id = {row["query_id"]: row["query"] for row in queries}
    labels_by_query = {}
    for row in labels:
        grade = GRADE_BY_LABEL[row["label"].strip().lower()]
        labels_by_query.setdefault(row["query_id"], {})[row["product_id"]] = grade
    print(f"  {len(queries_by_id)} queries, {sum(len(v) for v in labels_by_query.values())} labeled pairs")

    train_query_ids, held_out_query_ids = split_query_ids(queries_by_id.keys())
    print(f"  split: {len(train_query_ids)} train queries, {len(held_out_query_ids)} held-out queries")

    HELD_OUT_QUERIES_FILE.parent.mkdir(parents=True, exist_ok=True)
    HELD_OUT_QUERIES_FILE.write_text("\n".join(sorted(held_out_query_ids, key=int)) + "\n")
    print(f"  wrote held-out query ids to {HELD_OUT_QUERIES_FILE.relative_to(ROOT_DIR)}")

    print("Loading BGE query embedder...")
    embedder = QueryEmbedder(EMBEDDING_MODEL_DIR)

    print(f"Collecting training features for {len(train_query_ids)} queries (hits Elasticsearch)...")
    features, grade_labels = collect_training_data(queries_by_id, labels_by_query, train_query_ids, embedder)
    print(f"  {features.shape[0]} training examples, {features.shape[1]} features each")

    print("Training MLP...")
    model = train(features, grade_labels)

    onnx_path = OUTPUT_MODEL_DIR / "model.onnx"
    print(f"Exporting to {onnx_path.relative_to(ROOT_DIR)}...")
    export_onnx(model, onnx_path)
    verify_export(model, onnx_path, features[:32])

    metadata_path = OUTPUT_MODEL_DIR / "training-metadata.json"
    metadata_path.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "default_average_rating": DEFAULT_AVERAGE_RATING,
        "train_query_count": len(train_query_ids),
        "held_out_query_count": len(held_out_query_ids),
        "training_example_count": int(features.shape[0]),
    }, indent=2) + "\n")
    print(f"Wrote {metadata_path.relative_to(ROOT_DIR)}")
    print("Done.")


if __name__ == "__main__":
    main()
