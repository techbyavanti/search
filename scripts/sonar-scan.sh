#!/usr/bin/env bash
set -euo pipefail

# Runs SonarQube Community analysis with a quality gate on the whole codebase
# and fails if the gate fails. Mirrors the "sonarqube" CI job.
#
#   ./scripts/sonar-scan.sh          # starts a throwaway SonarQube on :9001, scans, removes it
#   ./scripts/sonar-scan.sh --keep   # same, but leaves it running to browse the report
#
# With SONAR_HOST_URL set (CI), it uses that server instead of starting one.
# The server must be fresh (default admin/admin login).
#
# Sonar's default gate checks only new code, which is empty on a fresh server,
# so this script sets its own gate on overall code: A ratings for reliability,
# security, and maintainability, every security hotspot reviewed, coverage at
# or above COVERAGE_FLOOR (default 26, measured without models/ as in CI;
# raise it as tests are added), and at most 3% duplicated lines.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SONAR_IMAGE="sonarqube:26.10.0.133078-community"
CONTAINER_NAME="search-sonarqube"
LOCAL_PORT=9001
GATE_NAME="search-overall"
COVERAGE_FLOOR="${COVERAGE_FLOOR:-26}"
MAX_DUPLICATED_LINES_PERCENT=3
STARTUP_TIMEOUT_SECONDS=300
# SonarQube's documented first-start login, replaced with a random password below.
FACTORY_LOGIN="admin"
FACTORY_PASSWORD="admin"

keep_server=false
[[ "${1:-}" == "--keep" ]] && keep_server=true

start_local_server() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -d --name "${CONTAINER_NAME}" -p "127.0.0.1:${LOCAL_PORT}:9000" \
    -e SONAR_ES_BOOTSTRAP_CHECKS_DISABLE=true "${SONAR_IMAGE}" >/dev/null
  if [[ "${keep_server}" == false ]]; then
    trap 'docker rm -f "${CONTAINER_NAME}" >/dev/null' EXIT
  fi
}

wait_until_up() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
  until curl -fsS "${SONAR_HOST_URL}/api/system/status" 2>/dev/null | grep -q '"status":"UP"'; do
    if ((SECONDS > deadline)); then
      echo "SonarQube at ${SONAR_HOST_URL} did not start within ${STARTUP_TIMEOUT_SECONDS}s" >&2
      exit 1
    fi
    sleep 5
  done
}

api() {
  curl -sS --fail-with-body -u "admin:${admin_password}" -X POST "${SONAR_HOST_URL}/api/$1" "${@:2}"
}

configure_server() {
  # Random hex plus a fixed suffix, so the password always meets Sonar's policy
  # (upper case, lower case, digit, and special character).
  admin_password="$(openssl rand -hex 16)Aa1!"
  curl -sS --fail-with-body -u "${FACTORY_LOGIN}:${FACTORY_PASSWORD}" -X POST "${SONAR_HOST_URL}/api/users/change_password" \
    --data-urlencode "login=${FACTORY_LOGIN}" --data-urlencode "previousPassword=${FACTORY_PASSWORD}" \
    --data-urlencode "password=${admin_password}"

  api qualitygates/create --data-urlencode "name=${GATE_NAME}" >/dev/null
  add_condition reliability_rating GT 1
  add_condition security_rating GT 1
  add_condition sqale_rating GT 1
  add_condition security_hotspots_reviewed LT 100
  add_condition coverage LT "${COVERAGE_FLOOR}"
  add_condition duplicated_lines_density GT "${MAX_DUPLICATED_LINES_PERCENT}"
  api qualitygates/set_as_default --data-urlencode "name=${GATE_NAME}"

  sonar_token="$(api user_tokens/generate --data-urlencode "name=scan-$(date +%s)" \
    --data-urlencode "type=GLOBAL_ANALYSIS_TOKEN" | python3 -c 'import json, sys; print(json.load(sys.stdin)["token"])')"
}

add_condition() {
  api qualitygates/create_condition --data-urlencode "gateName=${GATE_NAME}" \
    --data-urlencode "metric=$1" --data-urlencode "op=$2" --data-urlencode "error=$3" >/dev/null
}

if [[ -z "${SONAR_HOST_URL:-}" ]]; then
  SONAR_HOST_URL="http://localhost:${LOCAL_PORT}"
  start_local_server
fi

wait_until_up
configure_server

print_gate_conditions() {
  curl -fsS -u "admin:${admin_password}" "${SONAR_HOST_URL}/api/qualitygates/project_status?projectKey=search" |
    python3 -c 'import json, sys
for c in json.load(sys.stdin)["projectStatus"]["conditions"]:
    print("%-6s %s: %s (fails when %s %s)" % (c["status"], c["metricKey"], c.get("actualValue", "-"), c["comparator"], c["errorThreshold"]))' >&2
}

if ! SONAR_TOKEN="${sonar_token}" mvn -B verify sonar:sonar \
  -Dsonar.host.url="${SONAR_HOST_URL}" \
  -Dsonar.qualitygate.wait=true; then
  echo "Quality gate conditions:" >&2
  print_gate_conditions || true
  exit 1
fi

echo "Quality gate passed: ${SONAR_HOST_URL}/dashboard?id=search"
if [[ "${keep_server}" == true ]]; then
  echo "Login: admin / ${admin_password}"
fi
