#!/usr/bin/env bash
set -euo pipefail

# Builds search-eval and runs it against the WANDS queries/labels, writing
# RESULTS.md and per-strategy CSVs under results/.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"
mvn -q -pl search-eval -am package -DskipTests
java -jar search-eval/target/search-eval.jar "$@"
