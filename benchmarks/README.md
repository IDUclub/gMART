# Restrictions benchmark — re-run for the IAAI Emerging Applications paper

Two-stage evaluation methodology (per reviewer guidance):

```
8 models → operational + task-aware screening → 2 candidates (Gemma 3 12B, GPT-OSS 20B)
         → semantic + geometric validation vs expert gold → model choice for the pilot
```

## Layout

| Path | What |
|---|---|
| `data/gold/exp_data.csv` | 202 expert records: question, expected answer, expected layers, reference-GeoJSON cloud link, per-scenario catalog. **The gold set.** |
| `data/gold/full_dataset.csv` | 202 base + paraphrase augmentation (2222 rows, 32 territories). |
| `data/results/<model>/results.jsonl` | Previous operational runs (8 models × 2222). Layers only — no plan. |
| `eval/gold_parser.py` | Parse the 202 gold records → structured ground truth (intent, source/target entity, distance, expected layers, expected count) with per-field confidence flags. |
| `eval/semantic_eval.py` | Task-aware evaluation → `out/task_aware_report.md` (end states, failure taxonomy, task-aware success, count agreement). |
| `harness/run_benchmark.py` | Live re-run harness: preflight + Ollama model lifecycle + rich logging (**logs the RestrictionPlan**). |
| `harness/expand_dataset.py` | Paraphrase expansion of the 202 base queries (distance/entities preserved). |
| `harness/geometry_eval.py` | Object-selection P/R/F1 + geometry IoU vs reference GeoJSON *(pending the GeoJSON upload)*. |

`data/` and `out/` are git-ignored (large / local).

## What is scored where, and why

The gold set is **entirely `restrictions`-mode** tasks (spatial filter → count within a
zone): 200 restrictions + 2 clarification, 0 buffers_only. So "task-aware success" here means
the restrictions output (`objects` + `generators` layers) was produced with no execution error.

| Metric | Source | Status |
|---|---|---|
| End states (mutually exclusive, =100%) | existing results | ✅ computable now |
| Failure taxonomy (infra vs model vs backend) | existing results | ✅ computable now |
| Task-aware restrictions success | existing results | ✅ computable now |
| Object-count agreement (cross-check) | existing results + gold NL | ⚠️ weak (NL negation/OCR noise) |
| Intent / source & target entity / **buffer parameter** accuracy | **re-run** (needs the RestrictionPlan) | ⛔ old runs didn't log the plan |
| Object-selection P/R/F1 + geometry IoU | **reference GeoJSON** | ⛔ pending upload |

The old runs logged only final layers, so entity/parameter correctness cannot be recovered from
them — this is why the previous experiment could not demonstrate a "correct plan / correct spatial
result". `run_benchmark.py` fixes that by logging the plan.

## Running

```bash
# 1. Post-hoc screening over existing results (no server needed)
python benchmarks/eval/semantic_eval.py

# 2. Live re-run (needs gMART agents up + Urban API JWT; Ollama on a.dgx)
python benchmarks/harness/run_benchmark.py \
    --agents-base http://<gmart-agents> --ollama-host http://a.dgx:11434 \
    --token "$URBAN_API_JWT" --models gemma3:12b gpt-oss:20b \
    --out-dir benchmarks/data/results_rerun
#   Ablation: add  --ablation no_catalog   (requires the pipeline toggle, Task 4)

# 3. Dataset expansion (Ollama only)
python benchmarks/harness/expand_dataset.py --model gpt-oss:20b \
    --n-paraphrases 10 --out benchmarks/data/gold/expanded.csv

# 4. Geometry (after reference GeoJSON is placed under data/gold/reference/)
python benchmarks/harness/geometry_eval.py
```

### Model lifecycle (run_benchmark.py)

For each requested model: if present on the Ollama host → use and keep; if absent → pull, run,
then delete (leave the shared server as found). `--keep-pulled` overrides the delete.
As of last check `a.dgx:11434` has `gpt-oss:20b` (kept), and would pull+delete `gemma3:12b`.
