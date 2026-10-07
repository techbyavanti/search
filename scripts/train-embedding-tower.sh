#!/usr/bin/env bash
set -euo pipefail

# Fine-tunes bge-small-en-v1.5 on WANDS in one of two modes and exports the
# result to ONNX (training/train_embedding_towers.py). One-time/offline —
# requires network access on first run (downloads the BAAI/bge-small-en-v1.5
# PyTorch checkpoint) and dataset/*.csv (scripts/download-dataset.sh).
# Reuses the training/.venv set up by scripts/train-neural-reranker.sh.
#
# `shared` also populates every product's learned_embedding field
# (IngestionCli --learned-model-dir) — it's the winning mode from Track B's
# tower comparison (see TRAINING.md) and what search-api actually deploys,
# so training it without ingesting would leave the field stale/absent.
# `two-tower` only trains + exports; it lost the comparison and isn't wired
# into search-api, so there's nothing to (re-)ingest for it by default.
#
# Usage: ./scripts/train-embedding-tower.sh shared|two-tower

if [ $# -ne 1 ] || { [ "$1" != "shared" ] && [ "$1" != "two-tower" ]; }; then
  echo "Usage: $0 shared|two-tower" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/training/.venv"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating venv at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -q -r "${ROOT_DIR}/training/requirements.txt"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/training/train_embedding_towers.py" --mode "$1"

if [ "$1" = "shared" ]; then
  echo "Populating learned_embedding from models/learned-shared-tower..."
  cd "${ROOT_DIR}"
  mvn -q -pl search-ingestion -am package -DskipTests
  java -jar search-ingestion/target/search-ingestion.jar --learned-model-dir models/learned-shared-tower
fi
