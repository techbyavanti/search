#!/usr/bin/env bash
set -euo pipefail

# Trains the neural reranker (training/train_neural_reranker.py) and exports
# it to models/neural-reranker/model.onnx. One-time/offline — requires
# Elasticsearch running with the full catalog ingested (scripts/ingest-
# catalog.sh) and the embedding model downloaded (scripts/download-models.sh).
# Creates/reuses a venv under training/.venv so this doesn't touch system
# Python.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/training/.venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating venv at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -q -r "${ROOT_DIR}/training/requirements.txt"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/training/train_neural_reranker.py"
