#!/usr/bin/env bash
set -euo pipefail

# Scans every module's resolved Maven dependencies for known CVEs, the
# Dockerfile for misconfigurations, and the tree for secrets, via Trivy
# (brew install trivy). Local data and model files are skipped. Mirrors the
# "cve-scan" job in .github/workflows/build.yml so failures are reproducible
# locally before they show up in CI.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

trivy fs --scanners vuln,misconfig,secret --severity CRITICAL,HIGH,MEDIUM \
  --skip-dirs dataset,models,training/.venv .
