#!/bin/bash
# Resume the no-catalog ablation arm from wherever it stopped — reboot-safe.
#
# Unlike the baseline arm this one runs against the LOCAL stack, so the stack
# has to be up with the ablation flag before the harness starts:
#
#   ABLATION_NO_CATALOG=1 docker compose -f docker-compose.yaml \
#     -f docker-compose.override.yaml -f docker-compose.ablation.yaml up -d
#
# Credentials come from $HOME/.config/gmart-bench/creds.env, same as the
# baseline arm. Results append to benchmarks/data/results_ablation, and rows
# already present are skipped, so re-running this after a reboot continues.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1        # repo root
source .venv/bin/activate 2>/dev/null || true

CREDS_FILE="${GMART_BENCH_CREDS:-$HOME/.config/gmart-bench/creds.env}"
if [ -f "$CREDS_FILE" ]; then
  # shellcheck disable=SC1090
  source "$CREDS_FILE"
  echo "loaded credentials from $CREDS_FILE"
fi

if pgrep -f "run_benchmark.py --models" >/dev/null; then
  echo "a benchmark is already running (PID $(pgrep -f 'run_benchmark.py --models' | head -1)) — nothing to do"
  exit 0
fi
if [ -z "${KEYCLOAK_USER:-}" ] || [ -z "${KEYCLOAK_PASSWORD:-}" ] || [ -z "${AUTH_HELPER_API_KEY:-}" ]; then
  echo "ERROR: set KEYCLOAK_USER, KEYCLOAK_PASSWORD, AUTH_HELPER_API_KEY first."
  exit 1
fi
if [ "$(docker inspect -f '{{.State.Running}}' agents 2>/dev/null)" != "true" ]; then
  echo "ERROR: the local agents container is not running — bring the stack up first."
  exit 1
fi
if [ "$(docker exec agents printenv ABLATION_NO_CATALOG 2>/dev/null)" != "1" ]; then
  echo "ERROR: the local stack is running WITHOUT ABLATION_NO_CATALOG=1 — this would"
  echo "       silently produce a second baseline arm instead of the ablation."
  exit 1
fi

nohup python3 -u benchmarks/harness/run_benchmark.py \
  --models gemma3:12b gpt-oss:20b \
  --dataset benchmarks/data/gold/expanded_goldfirst.csv \
  --temperature 0 --timeout 360 --concurrency 2 \
  --ablation no_catalog \
  --agents-base http://localhost:80 \
  --ollama-host http://a.dgx:11434 \
  --token-url https://idu-auth-helper.idulab.ru/api/token \
  --out-dir benchmarks/data/results_ablation \
  >> benchmarks/out/ablation.log 2>&1 &

echo "ablation arm running as PID $! — follow: tail -f benchmarks/out/ablation.log"
