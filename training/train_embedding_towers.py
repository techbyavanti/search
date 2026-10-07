#!/usr/bin/env python3
"""Fine-tunes bge-small-en-v1.5 on WANDS relevance judgments, in one of two
modes, and exports the result(s) to ONNX with the exact tensor contract
EmbeddingService.java already expects (input_ids/attention_mask/
token_type_ids -> last_hidden_state, CLS-pooled + L2-normalized at serving
time) — so no new Java inference class is needed; a fine-tuned tower is
just another model directory EmbeddingService can load.

Modes:
  --mode shared      one encoder, used for both queries and products
                      (closer to today's off-the-shelf bge-small-en-v1.5).
  --mode two-tower    two independently-trained encoders, one specialized
                      per role.

Both modes are trained with the same triplet margin-ranking loss over
WANDS's Exact/Partial/Irrelevant grades, on the same 80% query split Track
A's reranker trains on (query_split.py) — the held-out 20% is scored later
via search-eval, not by this script (this project routes quality claims
through the real Java eval harness, not a Python-side proxy metric).

Requires:
  - dataset/product.csv, dataset/query.csv, dataset/label.csv.
  - models/bge-small-en-v1.5/tokenizer.json (for tokenization only).
  - Network access on first run, to download the BAAI/bge-small-en-v1.5
    PyTorch checkpoint via transformers — a separate download from the
    ONNX export already in models/, since ONNX Runtime is inference-only
    and fine-tuning needs real weights to backprop through.

Writes:
  --mode shared     -> models/learned-shared-tower/{model.onnx,tokenizer.json}
  --mode two-tower  -> models/learned-query-tower/{model.onnx,tokenizer.json}
                       models/learned-product-tower/{model.onnx,tokenizer.json}
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from transformers import AutoModel

from query_split import split_query_ids
from train_neural_reranker import load_tsv  # reuse the same TSV loader

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"

BASE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LOCAL_TOKENIZER_PATH = MODELS_DIR / "bge-small-en-v1.5" / "tokenizer.json"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_SEQUENCE_LENGTH = 256
MAX_DESCRIPTION_CHARS = 1200  # matches EmbeddingTextBuilder.MAX_DESCRIPTION_CHARS

TRIPLETS_PER_QUERY_CAP = 15
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 2e-5
MARGIN = 0.1

GRADE_BY_LABEL = {"exact": 2.0, "partial": 1.0, "irrelevant": 0.0}


def build_product_text(name, product_class, category_hierarchy, description):
    """Mirrors EmbeddingTextBuilder.build(name, class, categoryHierarchy, description) exactly."""
    parts = [name]
    if product_class:
        parts.append(product_class)
    if category_hierarchy:
        parts.append(re.sub(r"\s*/\s*", " > ", category_hierarchy))
    if description:
        parts.append(description[:MAX_DESCRIPTION_CHARS])
    return ". ".join(parts)


class Tower(nn.Module):
    """CLS-pool + L2-normalize wrapper around a HuggingFace AutoModel, matching EmbeddingService.java's pooling exactly."""

    def __init__(self, base_model_name=BASE_MODEL_NAME):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        # Newer transformers versions thread a `use_cache` kwarg through BertModel.forward
        # that collides with the legacy TorchScript ONNX tracer's own argument handling
        # ("got multiple values for argument 'use_cache'") unless it's disabled up front —
        # irrelevant here anyway since caching only matters for autoregressive generation.
        self.encoder.config.use_cache = False

    def forward(self, input_ids, attention_mask, token_type_ids):
        last_hidden_state = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        ).last_hidden_state
        cls_vectors = last_hidden_state[:, 0, :]
        return nn.functional.normalize(cls_vectors, p=2, dim=1)


def load_tokenizer():
    tokenizer = Tokenizer.from_file(str(LOCAL_TOKENIZER_PATH))
    tokenizer.enable_truncation(max_length=MAX_SEQUENCE_LENGTH)
    tokenizer.enable_padding()
    return tokenizer


def encode_batch(tokenizer, texts):
    encodings = tokenizer.encode_batch(texts)
    input_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long)
    attention_mask = torch.tensor([e.attention_mask for e in encodings], dtype=torch.long)
    token_type_ids = torch.tensor([e.type_ids for e in encodings], dtype=torch.long)
    return input_ids, attention_mask, token_type_ids


def build_triplets(train_query_ids, queries_by_id, labels_by_query, rng):
    """(query_text, higher_product_id, lower_product_id) triplets from within-query grade differences."""
    triplets = []
    for query_id in train_query_ids:
        products_by_grade = {}
        for product_id, grade in labels_by_query.get(query_id, {}).items():
            products_by_grade.setdefault(grade, []).append(product_id)

        grades = sorted(products_by_grade.keys(), reverse=True)
        pairs = []
        for i, higher_grade in enumerate(grades):
            for lower_grade in grades[i + 1:]:
                for higher_id in products_by_grade[higher_grade]:
                    for lower_id in products_by_grade[lower_grade]:
                        pairs.append((higher_id, lower_id))

        if not pairs:
            continue
        rng.shuffle(pairs)
        for higher_id, lower_id in pairs[:TRIPLETS_PER_QUERY_CAP]:
            triplets.append((queries_by_id[query_id], higher_id, lower_id))
    return triplets


def train(mode, triplets, products_by_id, tokenizer, epochs=EPOCHS, batch_size=BATCH_SIZE, print_every=50):
    query_tower = Tower()
    product_tower = query_tower if mode == "shared" else Tower()

    params = list(query_tower.parameters())
    if mode == "two-tower":
        params += list(product_tower.parameters())
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)
    loss_fn = nn.MarginRankingLoss(margin=MARGIN)

    query_tower.train()
    product_tower.train()

    total_steps = (len(triplets) + batch_size - 1) // batch_size
    rng = np.random.default_rng(42)
    for epoch in range(epochs):
        order = rng.permutation(len(triplets))
        total_loss, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            step_start = time.monotonic()
            batch_indices = order[start:start + batch_size]
            batch = [triplets[i] for i in batch_indices]

            query_texts = [QUERY_PREFIX + q for q, _, _ in batch]
            higher_texts = [build_product_text(*products_by_id[h]) for _, h, _ in batch]
            lower_texts = [build_product_text(*products_by_id[l]) for _, _, l in batch]

            q_ids, q_mask, q_types = encode_batch(tokenizer, query_texts)
            h_ids, h_mask, h_types = encode_batch(tokenizer, higher_texts)
            l_ids, l_mask, l_types = encode_batch(tokenizer, lower_texts)

            optimizer.zero_grad()
            query_embeddings = query_tower(q_ids, q_mask, q_types)
            higher_embeddings = product_tower(h_ids, h_mask, h_types)
            lower_embeddings = product_tower(l_ids, l_mask, l_types)

            cos_higher = (query_embeddings * higher_embeddings).sum(dim=1)
            cos_lower = (query_embeddings * lower_embeddings).sum(dim=1)
            target = torch.ones_like(cos_higher)

            loss = loss_fn(cos_higher, cos_lower, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            step_seconds = time.monotonic() - step_start
            if steps == 1 or steps % print_every == 0:
                print(f"  epoch {epoch + 1}/{epochs} step {steps}/{total_steps}: "
                      f"loss={total_loss / steps:.4f}, {step_seconds:.1f}s/step")

        print(f"  epoch {epoch + 1}/{epochs} done: mean loss={total_loss / max(steps, 1):.4f}")

    query_tower.eval()
    product_tower.eval()
    return query_tower, product_tower


def export_tower(tower, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dummy_len = 16
    dummy_ids = torch.zeros(1, dummy_len, dtype=torch.long)
    dummy_mask = torch.ones(1, dummy_len, dtype=torch.long)
    dummy_types = torch.zeros(1, dummy_len, dtype=torch.long)

    # Export the raw encoder (not the CLS-pool/normalize wrapper) so the ONNX
    # graph's output is last_hidden_state, matching EmbeddingService.java's
    # expected contract exactly — pooling/normalizing happens in Java.
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        tower.encoder, (dummy_ids, dummy_mask, dummy_types), str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))

    tokenizer_bytes = LOCAL_TOKENIZER_PATH.read_bytes()
    (output_dir / "tokenizer.json").write_bytes(tokenizer_bytes)
    return onnx_path


def verify_export(tower, onnx_path, sample_text):
    tokenizer = load_tokenizer()
    ids, mask, types = encode_batch(tokenizer, [QUERY_PREFIX + sample_text])

    with torch.no_grad():
        torch_hidden = tower.encoder(input_ids=ids, attention_mask=mask, token_type_ids=types).last_hidden_state
    torch_cls = torch_hidden[0, 0, :].numpy()

    session = ort.InferenceSession(str(onnx_path))
    onnx_hidden = session.run(["last_hidden_state"], {
        "input_ids": ids.numpy(), "attention_mask": mask.numpy(), "token_type_ids": types.numpy(),
    })[0]
    onnx_cls = onnx_hidden[0, 0, :]

    max_diff = float(np.max(np.abs(torch_cls - onnx_cls)))
    print(f"  max |torch - onnx| CLS-token diff: {max_diff:.6f}")
    if max_diff > 1e-3:
        raise RuntimeError(f"ONNX export mismatch too large: {max_diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["shared", "two-tower"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-triplets", type=int, default=None,
                         help="Cap total training triplets — CPU fine-tuning of a full BERT-small forward+backward "
                              "is genuinely slow (single-digit seconds/step is normal); use a small value here "
                              "(e.g. 64) to smoke-test the pipeline before committing to a full run, which at the "
                              "default triplet count/epochs can take hours on CPU.")
    args = parser.parse_args()

    print("Loading WANDS products, queries, labels...")
    products = load_tsv(DATASET_DIR / "product.csv")
    queries = load_tsv(DATASET_DIR / "query.csv")
    labels = load_tsv(DATASET_DIR / "label.csv")

    products_by_id = {row["product_id"]: (row["product_name"], row["product_class"] or None,
                                            row["category hierarchy"] or None, row["product_description"] or None)
                       for row in products}
    queries_by_id = {row["query_id"]: row["query"] for row in queries}
    labels_by_query = {}
    for row in labels:
        grade = GRADE_BY_LABEL[row["label"].strip().lower()]
        labels_by_query.setdefault(row["query_id"], {})[row["product_id"]] = grade

    train_query_ids, held_out_query_ids = split_query_ids(queries_by_id.keys())
    print(f"  split: {len(train_query_ids)} train queries, {len(held_out_query_ids)} held-out queries (shared with Track A)")

    rng = __import__("random").Random(42)
    triplets = build_triplets(train_query_ids, queries_by_id, labels_by_query, rng)
    print(f"  built {len(triplets)} training triplets")
    if args.max_triplets is not None and len(triplets) > args.max_triplets:
        rng.shuffle(triplets)
        triplets = triplets[:args.max_triplets]
        print(f"  capped to {len(triplets)} triplets (--max-triplets)")

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print(f"Loading {BASE_MODEL_NAME} and fine-tuning (mode={args.mode})...")
    query_tower, product_tower = train(args.mode, triplets, products_by_id, tokenizer,
                                        epochs=args.epochs, batch_size=args.batch_size)

    if args.mode == "shared":
        output_dir = MODELS_DIR / "learned-shared-tower"
        print(f"Exporting to {output_dir.relative_to(ROOT_DIR)}...")
        onnx_path = export_tower(query_tower, output_dir)
        verify_export(query_tower, onnx_path, "queen size platform bed frame")
    else:
        query_dir = MODELS_DIR / "learned-query-tower"
        product_dir = MODELS_DIR / "learned-product-tower"
        print(f"Exporting query tower to {query_dir.relative_to(ROOT_DIR)}...")
        query_onnx_path = export_tower(query_tower, query_dir)
        verify_export(query_tower, query_onnx_path, "queen size platform bed frame")
        print(f"Exporting product tower to {product_dir.relative_to(ROOT_DIR)}...")
        product_onnx_path = export_tower(product_tower, product_dir)
        verify_export(product_tower, product_onnx_path, "Solid wood queen platform bed frame with wooden slats.")

    print("Done.")


if __name__ == "__main__":
    main()
