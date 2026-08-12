#!/bin/bash
# Measure how the harness scales with --concurrency before committing to a
# multi-day run. Each level runs the same N rows from scratch against the local
# stack and reports wall time and rows/hour, so the level is chosen from data
# rather than from a guess about the shared GPU's parallelism.
#
#   bash benchmarks/harness/measure_concurrency.sh [N] [levels...]
#   bash benchmarks/harness/measure_concurrency.sh 12 2 4 6

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate 2>/dev/null || true

CREDS_FILE="${GMART_BENCH_CREDS:-$HOME/.config/gmart-bench/creds.env}"
[ -f "$CREDS_FILE" ] && source "$CREDS_FILE"

N="${1:-12}"; shift || true
LEVELS=("$@"); [ ${#LEVELS[@]} -eq 0 ] && LEVELS=(2 4 6)
MODEL="${MEASURE_MODEL:-gpt-oss:20b}"
OUT_ROOT="benchmarks/data/results_concurrency"

printf '%-6s %-8s %-10s %-12s %s\n' "conc" "rows" "wall_s" "s_per_row" "rows_per_h"
for c in "${LEVELS[@]}"; do
  dir="$OUT_ROOT/c$c"
  rm -rf "$dir"
  start=$(date +%s)
  python3 -u benchmarks/harness/run_benchmark.py \
    --models "$MODEL" \
    --dataset benchmarks/data/gold/gold_202.csv \
    --temperature 0 --timeout 360 --concurrency "$c" --limit "$N" \
    --agents-base http://localhost:80 \
    --ollama-host http://a.dgx:11434 \
    --token-url https://idu-auth-helper.idulab.ru/api/token \
    --out-dir "$dir" > "benchmarks/out/concurrency_c$c.log" 2>&1
  end=$(date +%s)
  wall=$((end - start))
  rows=$(cat "$dir"/*/results.jsonl 2>/dev/null | wc -l)
  [ "$rows" -eq 0 ] && rows=1
  printf '%-6s %-8s %-10s %-12s %s\n' \
    "$c" "$rows" "$wall" \
    "$(awk -v w=$wall -v r=$rows 'BEGIN{printf "%.1f", w/r}')" \
    "$(awk -v w=$wall -v r=$rows 'BEGIN{printf "%.0f", r*3600/w}')"
done
echo "done"
