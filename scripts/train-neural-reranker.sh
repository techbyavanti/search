#!/usr/bin/env bash
set -euo pipefail

# Trains the neural reranker and exports it to models/neural-reranker/model.onnx.
# Requires Elasticsearch running with the full catalog ingested
# (scripts/ingest-catalog.sh) and the embedding model downloaded
# (scripts/download-models.sh).
#
# 1. Java (RerankFeatureExportCli) writes the hybrid candidate pools and
#    their feature vectors, using the same code that serves them.
# 2. Python (training/train_neural_reranker.py) trains the MLP with a
#    LambdaRank loss and exports it to ONNX. Extra arguments are passed to
#    it (for example --loss mse).
#
# Creates/reuses a venv under training/.venv so this doesn't touch system Python.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/training/.venv"

cd "${ROOT_DIR}"
mvn -q -pl search-eval -am package -DskipTests
java -cp search-eval/target/search-eval.jar com.avanti.search.eval.RerankFeatureExportCli

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating venv at ${VENV_DIR}..."
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install -q -r "${ROOT_DIR}/training/requirements.txt"
"${VENV_DIR}/bin/python" "${ROOT_DIR}/training/train_neural_reranker.py" "$@"
