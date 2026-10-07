#!/usr/bin/env bash
set -uo pipefail

# Stops search-api however it was started: as a docker-compose service
# (started via `docker compose up -d --build search-api`), or as a plain
# local process (started via `java -jar search-api/target/search-api.jar`
# or IntelliJ's "search-api (app)" config outside the IDE's own process
# manager) bound to localhost:8080.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

stopped=false

if docker compose ps -q search-api 2>/dev/null | grep -q .; then
  echo "Stopping the search-api docker-compose container..."
  docker compose stop search-api
  stopped=true
fi

pids=$(lsof -ti tcp:8080 2>/dev/null || true)
if [ -n "${pids}" ]; then
  echo "Killing local process(es) on :8080 (pid(s): $(echo "${pids}" | tr '\n' ' '))..."
  echo "${pids}" | xargs kill
  stopped=true
fi

if [ "${stopped}" = false ]; then
  echo "Nothing appears to be running on localhost:8080 or as the search-api container."
fi
