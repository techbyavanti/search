#!/usr/bin/env bash
set -euo pipefail

# Fetches the pre-exported ONNX weights + tokenizer for the embedding model
# (BAAI/bge-small-en-v1.5) and the reranker (cross-encoder/ms-marco-MiniLM-L-6-v2),
# via Xenova's ONNX exports, into models/.
# https://huggingface.co/Xenova/bge-small-en-v1.5
# https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2
#
# The reranker fetches the dynamic-INT8-quantized export rather than fp32 —
# reranking scores a whole 50-candidate pool per query in one forward pass
# (unlike the embedding model's batch-of-1 query encode), so it's far more
# latency-sensitive; quantization cuts real per-query cost substantially
# with negligible ranking-quality impact for this small a model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${ROOT_DIR}/models"

fetch_model() {
  local repo="$1"
  local dir="${MODELS_DIR}/$2"
  local onnx_file="${3:-model.onnx}"
  local base="https://huggingface.co/${repo}/resolve/main"

  mkdir -p "${dir}"
  echo "Downloading ${repo} (${onnx_file})..."
  curl -fsSL "${base}/onnx/${onnx_file}" -o "${dir}/model.onnx"
  curl -fsSL "${base}/tokenizer.json" -o "${dir}/tokenizer.json"
  curl -fsSL "${base}/config.json" -o "${dir}/config.json"
  du -h "${dir}"/*
}

fetch_model "Xenova/bge-small-en-v1.5" "bge-small-en-v1.5"
fetch_model "Xenova/ms-marco-MiniLM-L-6-v2" "ms-marco-MiniLM-L-6-v2" "model_quantized.onnx"

echo "Done."
