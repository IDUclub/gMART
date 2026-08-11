#!/bin/bash
# Run both local arms over the 202 base gold queries, back to back:
#
#   1. no_catalog ablation   (ABLATION_NO_CATALOG=1)  -> results_ablation_gold
#   2. control, same build   (ABLATION_NO_CATALOG=0)  -> results_control_gold
#
# The control arm is what makes the ablation interpretable: both arms run on the
# SAME local build, so the difference is the catalog grounding alone and not the
# code drift between this branch and the deployment the baseline ran against.
#
# Each arm restarts the stack with its own flag, then runs the harness for both
# models. Rows already in a results.jsonl are skipped, so re-running this after
# a reboot continues where it stopped. ~2.3 h per arm per model at concurrency 2.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1        # repo root
source .venv/bin/activate 2>/dev/null || true

CREDS_FILE="${GMART_BENCH_CREDS:-$HOME/.config/gmart-bench/creds.env}"
[ -f "$CREDS_FILE" ] && source "$CREDS_FILE"

if [ -z "${KEYCLOAK_USER:-}" ] || [ -z "${KEYCLOAK_PASSWORD:-}" ] || [ -z "${AUTH_HELPER_API_KEY:-}" ]; then
  echo "ERROR: set KEYCLOAK_USER, KEYCLOAK_PASSWORD, AUTH_HELPER_API_KEY first."
  exit 1
fi

COMPOSE="docker compose -f docker-compose.yaml -f docker-compose.override.yaml -f docker-compose.ablation.yaml"
OLLAMA="${OLLAMA_HOST_URL:-http://a.dgx:11434}"

# a.dgx is shared. When somebody else's model occupies it, our requests either
# queue for tens of minutes or hang outright (the harness timeout bounds a single
# SSE read, not the whole request). So before an arm starts, block until the box
# actually answers a trivial generation for our model.
wait_for_model() {                # $1 = model tag
  local model="$1" waited=0
  while true; do
    if curl -s -m 240 -o /dev/null -w '%{http_code}' \
         "$OLLAMA/api/generate" \
         -d "{\"model\":\"$model\",\"prompt\":\"ping\",\"stream\":false,\"options\":{\"num_predict\":1}}" \
       | grep -q '^200$'; then
      echo "$(date '+%H:%M:%S') $model is served (waited ${waited}s)"
      return 0
    fi
    resident=$(curl -s -m 15 "$OLLAMA/api/ps" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | tr '\n' ' ')
    echo "$(date '+%H:%M:%S') $model not available yet (resident: ${resident:-none}) — retrying in 120s"
    sleep 120
    waited=$((waited + 120))
  done
}

run_arm() {                      # $1 = flag value, $2 = out dir, $3 = --ablation label
  echo "=== arm: ABLATION_NO_CATALOG=$1 -> $2 ==="
  ABLATION_NO_CATALOG="$1" $COMPOSE up -d >/dev/null 2>&1
  for _ in $(seq 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:80/ping || true)" = "200" ] && break
    sleep 2
  done
  actual=$(docker exec agents printenv ABLATION_NO_CATALOG 2>/dev/null || echo "?")
  if [ "$actual" != "$1" ]; then
    echo "ERROR: stack came up with ABLATION_NO_CATALOG=$actual, expected $1 — aborting."
    exit 1
  fi
  wait_for_model gpt-oss:20b
  python3 -u benchmarks/harness/run_benchmark.py \
    --models gemma3:12b gpt-oss:20b \
    --dataset benchmarks/data/gold/gold_202.csv \
    --temperature 0 --timeout 360 --concurrency 2 \
    --ablation "$3" \
    --agents-base http://localhost:80 \
    --ollama-host http://a.dgx:11434 \
    --token-url https://idu-auth-helper.idulab.ru/api/token \
    --out-dir "$2"
}

run_arm 1 benchmarks/data/results_ablation_gold no_catalog
run_arm 0 benchmarks/data/results_control_gold none
echo "=== both arms complete ==="
