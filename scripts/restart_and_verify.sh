#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
TIMEOUT_SEC="${2:-60}"

echo "[1/3] restart api container"
docker compose restart api >/dev/null

echo "[2/3] wait health endpoint"
start_ts=$(date +%s)
while true; do
  if curl -fsS "${BASE_URL}/api/health" >/dev/null 2>&1; then
    break
  fi
  now_ts=$(date +%s)
  if (( now_ts - start_ts > TIMEOUT_SEC )); then
    echo "FAIL: api health check timeout (${TIMEOUT_SEC}s)"
    exit 2
  fi
  sleep 1
done

echo "[3/3] run smoke checks"
python3 scripts/smoke_check.py --base-url "${BASE_URL}" --timeout "${TIMEOUT_SEC}"

echo "DONE: restart + verify passed"
