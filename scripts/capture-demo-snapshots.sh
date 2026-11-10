#!/usr/bin/env bash
set -euo pipefail

# Captures live /api/search/compare results for a curated set of demo queries
# against a locally-running search-api (see README's "Local setup"), writing
# one JSON file per query into docs/data/. Each strategy is fetched to FETCH_K
# and scripts/collapse-demo-duplicates.py keeps the best-ranked listing per
# product name, cut to TOP_K, so a demo column never shows one name twice.
# These captures are what's baked into docs/index.html's QUERY_DATA block for
# the static GitHub Pages demo — after running this, regenerate that block by
# hand from the new JSON files, then run scripts/extract-product-details.py to
# refresh the per-result "why was this shown" product data
# (docs/data/product-details.json / the PRODUCT_DETAILS block) to match.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/docs/data"
HOST="${1:-http://localhost:8080}"
TOP_K=5
FETCH_K=30

QUERIES=(
  "queen size platform bed frame"
  "cozy reading chair for small apartment"
  "modern oak dining table"
  "outdoor patio furniture set"
  "small space storage ottoman"
  "kids bunk bed with stairs"
  "farmhouse dining bench"
  "velvet accent chair"
)

mkdir -p "${DATA_DIR}"

i=0
for query in "${QUERIES[@]}"; do
  i=$((i + 1))
  encoded=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "${query}")
  out="${DATA_DIR}/q${i}.json"
  curl -fsSL "${HOST}/api/search/compare?query=${encoded}&topK=${FETCH_K}" -o "${out}"
  python3 "${ROOT_DIR}/scripts/collapse-demo-duplicates.py" --top-k "${TOP_K}" "${out}"
  echo "Captured \"${query}\" -> ${out}"
done
