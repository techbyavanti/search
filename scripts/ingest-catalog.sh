#!/usr/bin/env bash
set -euo pipefail

# Builds search-ingestion and bulk-indexes the full WANDS catalog (with real
# embeddings) into Elasticsearch. Requires Elasticsearch already running
# (see start-elasticsearch.sh) and dataset/models already fetched (see
# download-dataset.sh / download-models.sh).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
mvn -q -pl search-ingestion -am package -DskipTests
java -jar search-ingestion/target/search-ingestion.jar --recreate "$@"
