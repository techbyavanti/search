#!/usr/bin/env bash
set -euo pipefail

# Starts Elasticsearch + Kibana via docker compose and waits for the ES
# cluster health endpoint to respond before returning, so callers (e.g. an
# ingestion step chained after this one) don't race a container that's
# still booting.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT_SECONDS="${1:-120}"

cd "${ROOT_DIR}"
docker compose up -d elasticsearch kibana

echo "Waiting for Elasticsearch cluster health (up to ${TIMEOUT_SECONDS}s)..."
elapsed=0
until curl -sf http://localhost:9200/_cluster/health >/dev/null; do
  if [ "${elapsed}" -ge "${TIMEOUT_SECONDS}" ]; then
    echo "Elasticsearch did not become healthy within ${TIMEOUT_SECONDS}s" >&2
    exit 1
  fi
  sleep 2
  elapsed=$((elapsed + 2))
done

echo "Elasticsearch is up (after ${elapsed}s)."
