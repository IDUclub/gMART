#!/bin/bash
# Full experiment chain on the vLLM (OpenAI-compatible) backend.
#
#   1. finish reframing the gold questions as restriction checks
#   2. regenerate the ~2200-row paraphrase set from that gold
#   3. main arm   (catalog on)  over the expanded set, both models
#   4. ablation   (catalog off) over the 202 base questions, both models
#
# Each stage is resumable: the reframe and expansion overwrite their outputs only
# on success, and the harness skips result rows it already has, so re-running the
# script after an interruption continues rather than restarts. Failed rows are the
# exception — --retry-errors sets them aside and queues them again, so a fix to the
# stack or the model server actually reaches the rows it was meant to fix.
#
# ChatStorage (with MongoDB) and Redis are NOT needed: the stack runs with
# DISABLE_CHAT_HISTORY=1 and DISABLE_PIPELINE_STATE=1. A benchmark row never
# reconnects and reruns on an expired token, so the state those services keep has
# no reader — and on the largest scenarios writing it was failing whole rows.
#
# Progress is meant to be watched with progress.py; everything here goes to the
# log, errors included.

set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1
source .venv/bin/activate 2>/dev/null || true

CREDS_FILE="${GMART_BENCH_CREDS:-$HOME/.config/gmart-bench/creds.env}"
[ -f "$CREDS_FILE" ] && source "$CREDS_FILE"

if [ -z "${KEYCLOAK_USER:-}" ] || [ -z "${KEYCLOAK_PASSWORD:-}" ] || [ -z "${AUTH_HELPER_API_KEY:-}" ]; then
  echo "ERROR: set KEYCLOAK_USER, KEYCLOAK_PASSWORD, AUTH_HELPER_API_KEY first."
  exit 1
fi

GPT=http://a6k4.dgx:8001
GEMMA=http://a6k4.dgx:8002
ENDPOINTS="gpt-oss-20b=$GPT,gemma-3-27b=$GEMMA"
MODELS=(gpt-oss-20b gemma-3-27b)
# vLLM batches requests, so the ceiling is the agents pipeline and Urban API,
# not the GPU; 8 rows in flight per model was the starting point.
CONCURRENCY="${CONCURRENCY:-8}"
GOLD=benchmarks/data/gold/exp_data_restrictions.csv
EXPANDED=benchmarks/data/gold/expanded_restrictions.csv
BASE202=benchmarks/data/gold/gold_202_restrictions.csv

# Two chains writing into the same results files interleave their rows and
# produce duplicates: killing the driver alone leaves its harness children alive,
# so refuse to start until they are gone too.
if pgrep -f "run_benchmark\.py" >/dev/null 2>&1; then
  echo "ERROR: a benchmark harness is already running:"
  pgrep -af "run_benchmark\.py" | sed "s/^/  /"
  echo "stop it first:  pkill -f 'run_vllm_experiments\\.sh'; pkill -f 'run_benchmark\\.py'"
  exit 1
fi

COMPOSE="docker compose -f docker-compose.yaml -f docker-compose.override.yaml -f docker-compose.ablation.yaml"

stage() { echo; echo "=== $* ==="; date '+%F %T'; }

wait_for_model() {                      # $1 = url, $2 = model
  local waited=0
  until curl -s -m 120 -o /dev/null -w '%{http_code}' "$1/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$2\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
      | grep -q '^200$'; do
    echo "$(date '+%H:%M:%S') $2 at $1 not answering — retrying in 60s (waited ${waited}s)"
    sleep 60
    waited=$((waited + 60))
  done
}

restack() {                             # $1 = ABLATION_NO_CATALOG value
  # --build, because the images are built from this working tree: without it a
  # source fix silently does not reach the run (which has already happened once).
  # Only the two services are named: pipeline_storage (Redis) stays down, which is
  # the point of DISABLE_PIPELINE_STATE.
  ABLATION_NO_CATALOG="$1" $COMPOSE up -d --build idu_agents idu_mcp >/dev/null 2>&1
  for _ in $(seq 40); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:80/ping || true)" = "200" ] && break
    sleep 2
  done
  local actual
  actual=$(docker exec agents printenv ABLATION_NO_CATALOG 2>/dev/null || echo "?")
  if [ "$actual" != "$1" ]; then
    echo "ERROR: stack came up with ABLATION_NO_CATALOG=$actual, expected $1"
    exit 1
  fi
  # ChatStorage (with its MongoDB) and Redis are deliberately not part of the
  # experiment stack; without the flags the pipeline would still write to both.
  for flag in DISABLE_CHAT_HISTORY DISABLE_PIPELINE_STATE; do
    if [ "$(docker exec agents printenv "$flag" 2>/dev/null || echo "?")" != "1" ]; then
      echo "ERROR: stack came up without $flag=1"
      exit 1
    fi
  done
}

run_arm() {                             # $1 = dataset, $2 = out dir, $3 = label
  # One harness process per model, run at the same time: the two models sit on
  # separate vLLM servers, so serialising them halves throughput for nothing.
  local pids=()
  for model in "${MODELS[@]}"; do
    local url=$GPT; [ "$model" = "gemma-3-27b" ] && url=$GEMMA
    wait_for_model "$url" "$model"
    python3 -u benchmarks/harness/run_benchmark.py \
      --models "$model" \
      --dataset "$1" \
      --temperature 0 --timeout 600 --concurrency "$CONCURRENCY" \
      --ablation "$3" --retry-errors \
      --llm-api openai --model-endpoints "$ENDPOINTS" \
      --agents-base http://localhost:80 \
      --token-url https://idu-auth-helper.idulab.ru/api/token \
      --out-dir "$2" &
    pids+=($!)
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  # An arm that gave up (the service went away) must not be followed by the next
  # one: the ablation would run against an unfinished main arm, and the operator
  # would find both incomplete.
  if [ "$failed" != "0" ]; then
    echo "ERROR: a model run exited non-zero — stopping the chain here."
    echo "       fix the cause, then rerun this script; --retry-errors picks the"
    echo "       failed rows back up."
    exit 1
  fi
}

# --------------------------------------------------------------------------- #
stage "1/4 reframing the gold questions (gpt-oss-20b via vLLM)"
if [ ! -f "$GOLD" ] || [ "${FORCE_REFRAME:-0}" = "1" ]; then
  python3 -u benchmarks/harness/reframe_gold.py \
    --ollama-host "$GPT/v1" --model gpt-oss-20b
else
  echo "already present: $GOLD (FORCE_REFRAME=1 to redo)"
fi

stage "2/4 expanding to paraphrases"
if [ ! -f "$EXPANDED" ] || [ "${FORCE_EXPAND:-0}" = "1" ]; then
  python3 -u benchmarks/harness/expand_dataset.py \
    --base "$GOLD" --out "$EXPANDED" \
    --ollama-host "$GPT/v1" --model gpt-oss-20b --n-paraphrases 10
else
  echo "already present: $EXPANDED (FORCE_EXPAND=1 to redo)"
fi
python3 - <<PY
import pandas as pd
d = pd.read_csv("$EXPANDED", sep=";", engine="python")
d[d["variant"] == 0].to_csv("$BASE202", sep=";", index=False)
print(f"expanded rows: {len(d)}; base rows written to $BASE202")
PY

stage "3/4 main arm — catalog ON, over the expanded set"
restack 0
run_arm "$EXPANDED" benchmarks/data/results_vllm_main none

stage "4/4 ablation arm — catalog OFF, over the 202 base questions"
restack 1
run_arm "$BASE202" benchmarks/data/results_vllm_ablation no_catalog

stage "done"
