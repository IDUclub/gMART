#!/bin/bash
# Resume the benchmark batch from wherever it stopped — reboot-safe.
#
# The run is resumable: it reads the existing results.jsonl, skips rows already
# done (by idx) and appends the rest. So after a reboot (or terminal close) just
# provide credentials and run this script; it continues from the last written row.
#
# 1) Provide credentials in the environment (they are NOT stored in this script):
#      export KEYCLOAK_USER='...'
#      export KEYCLOAK_PASSWORD='...'
#      export AUTH_HELPER_API_KEY='...'
#    (or: source your own creds file that exports them)
# 2) bash benchmarks/harness/resume_run.sh
#
# nohup detaches it so it also survives closing the terminal / SSH session
# (a full reboot still kills it — just run this script again afterwards).

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1        # repo root
source .venv/bin/activate 2>/dev/null || true

# auto-load persistent credentials (outside the repo, survives reboot)
CREDS_FILE="${GMART_BENCH_CREDS:-$HOME/.config/gmart-bench/creds.env}"
if [ -f "$CREDS_FILE" ]; then
  # shellcheck disable=SC1090
  source "$CREDS_FILE"
  echo "loaded credentials from $CREDS_FILE"
fi

if pgrep -f "run_benchmark.py --models" >/dev/null; then
  echo "already running (PID $(pgrep -f 'run_benchmark.py --models' | head -1)) — nothing to do"
  exit 0
fi
if [ -z "${KEYCLOAK_USER:-}" ] || [ -z "${KEYCLOAK_PASSWORD:-}" ] || [ -z "${AUTH_HELPER_API_KEY:-}" ]; then
  echo "ERROR: set KEYCLOAK_USER, KEYCLOAK_PASSWORD, AUTH_HELPER_API_KEY first."
  exit 1
fi

nohup python3 -u benchmarks/harness/run_benchmark.py \
  --models gemma3:12b gpt-oss:20b \
  --dataset benchmarks/data/gold/expanded_goldfirst.csv \
  --temperature 0 --timeout 360 --concurrency 2 \
  --agents-base http://10.32.1.99 \
  --ollama-host http://a.dgx:11434 \
  --token-url https://idu-auth-helper.idulab.ru/api/token \
  --out-dir benchmarks/data/results_rerun_full \
  >> benchmarks/out/rerun_full.log 2>&1 &

echo "resumed as PID $! — follow: tail -f benchmarks/out/rerun_full.log"
echo "progress: .venv/bin/python3 benchmarks/harness/status_report.py"
