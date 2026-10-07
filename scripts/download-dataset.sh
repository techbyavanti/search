#!/usr/bin/env bash
set -euo pipefail

# Fetches the WANDS (Wayfair ANnotated Dataset) CSVs into dataset/.
# https://github.com/wayfair/WANDS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${ROOT_DIR}/dataset"
RAW_BASE="https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"

mkdir -p "${DATASET_DIR}"

for file in product.csv query.csv label.csv; do
  echo "Downloading ${file}..."
  curl -fsSL "${RAW_BASE}/${file}" -o "${DATASET_DIR}/${file}"
done

echo "Done. Row counts:"
for file in product.csv query.csv label.csv; do
  count=$(($(wc -l < "${DATASET_DIR}/${file}") - 1))
  echo "  ${file}: ${count} rows"
done
