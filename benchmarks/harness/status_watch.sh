#!/bin/bash
# Emit a batch status every time total completed records crosses a 200 boundary,
# plus a final line when the batch stops. One stdout block per event (Monitor).
cd /home/leon/Projects/IDU/IDUClub/ICII/gMART || exit 1
DIR=benchmarks/data/results_rerun_full
PY=.venv/bin/python3
last=-1
while true; do
  total=$(cat $DIR/*/results.jsonl 2>/dev/null | wc -l)
  bucket=$(( total / 200 ))
  if [ "$total" -gt 0 ] && [ "$bucket" -ne "$last" ]; then
    last=$bucket
    "$PY" benchmarks/harness/status_report.py
  fi
  if ! pgrep -f "run_benchmark.py --models" >/dev/null; then
    echo "=== BATCH STOPPED (total=$total records) — restart to resume ==="
    "$PY" benchmarks/harness/status_report.py
    exit 0
  fi
  sleep 45
done
