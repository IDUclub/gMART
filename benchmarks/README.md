# Restrictions benchmark — re-run for the IAAI Emerging Applications paper

Two-stage evaluation methodology (per reviewer guidance):

```
8 models → operational + task-aware screening → 2 candidates (Gemma 3 12B, GPT-OSS 20B)
         → semantic + geometric validation vs expert gold → model choice for the pilot
```

The re-run is driven **in code**, not through the deployed application. See
[Why the run has no HTTP interface](#why-the-run-has-no-http-interface).

## Layout

| Path | What |
|---|---|
| `data/gold/exp_data.csv` | 202 expert records: question, expected answer, expected layers, reference-GeoJSON cloud link, per-scenario catalog. **The gold set.** |
| `data/gold/full_dataset.csv` | 202 base + paraphrase augmentation (2222 rows, 32 territories). |
| `data/results/<model>/results.jsonl` | Previous operational runs (8 models × 2222) over HTTP. Layers only — no plan. |
| `harness/inproc_runner.py` | **The re-run harness.** Drives the pipeline in this process: no FastAPI, no SSE, no Redis, no ChatStorage; optionally no MCP transport either. Records the plan, per-stage timings, and a failure class decided where the failure happens. |
| `harness/prefetch_scenarios.py` | Fills the offline Urban API store so runs need no network. Run once with the VPN up. |
| `harness/run_benchmark.py` | The previous HTTP/SSE harness. Kept for the transport comparison and for reproducing the old numbers; not the path for new results. |
| `data/gold/exp_data_restrictions.csv` | The 95 of those 202 that are restrictions tasks this pipeline can be asked. **The set the re-run uses.** |
| `eval/gold_parser.py` | Parse the 202 gold records → structured ground truth (intent, source/target entity, distance, expected layers, expected count) with per-field confidence flags. |
| `eval/restrictions_filter.py` | Cut the gold set down to rows the pipeline can actually answer → `out/dataset_filter.md` lists every dropped row and why. |
| `eval/inproc_eval.py` | **The report for the re-run.** Task-aware success by task type, plan correctness, end states, failure taxonomy, ablation and transport comparisons → `out/inproc_report.md`. |
| `eval/semantic_eval.py` | The same shape of report for the older HTTP runs (which carry no plan) → `out/task_aware_report.md`. |
| `harness/expand_dataset.py` | Paraphrase expansion of the 202 base queries (distance/entities preserved). |
| `harness/geometry_eval.py` | Object-selection P/R/F1 + geometry IoU vs reference GeoJSON. Reads either harness's records: inline layers, or the GeoJSON files the in-process runner writes under `--save-layers` *(scoring still pending the reference GeoJSON upload)*. |

`data/`, `out/` and `runtime/` are git-ignored (large / local).

## The gold set is not all restrictions tasks

The 202 expert records were written as useful urban-planning questions, not as a
test suite for this pipeline. **95 of them are restrictions tasks it can be
asked**; the rest fail for reasons the model has no part in, and counting those
as model failures is the completion-proxy mistake in a new place.

`eval/restrictions_filter.py` makes the cut, on verifiable grounds only:

| dropped | rows | why it is not a model failure |
|---|---|---|
| `entity_absent` | 82 | one of the two entities is not in that scenario's Urban API catalog — scenario 558 exposes only «Нежилое здание» while its questions ask about жилые дома, парки, школы. No plan can name what the catalog does not have. |
| `time_distance` | 13 | the radius is in *minutes* («пешей доступности, 10 минут»). `CreateBuffers` builds metric buffers; a travel-time catchment is another tool in another service. |
| `no_distance` | 5 | no radius at all, so `buffer_size` cannot be filled and asking is the correct behaviour |
| `data_unavailable` | 4 | the layer cannot be fetched: private scenario (403), missing scenario (404), or the Urban API line-geometry 500 |
| `complement` | 3 | asks for what is **outside** the zone («вне радиуса», «не оснащены … в радиусе»). The pipeline selects objects *inside* a buffer; the complement is a different operation, not a synonym. |

Not dropped, and reported instead: **45 of the kept 95** are phrased as a share
or a verdict («какой процент жилых домов…», «являются ли библиотеки
доступными…»). The spatial operation underneath is the same buffer-and-contain,
so they stay; `--drop-provision` removes them if the paper wants restrictions
phrasing only.

Entity resolution is the risky part of that filter — it deletes expert records —
so it is deliberately conservative and its two failure directions are pinned by
tests: «жилыми зданиями» must still reach «Жилой дом» through the synonym table,
and «парк» must never resolve to «Парковка» (0.67 string similarity, versus 0.83
for the inflection «школами»/«Школа» — which is why the match is on shared
prefix length, not on a similarity threshold).

## Why the run has no HTTP interface

The previous experiment reached the pipeline through the deployed application — an SSE
endpoint, Redis-backed pipeline state, ChatStorage, a token proxy, and MCP over HTTP.
None of that is the object of study, and all of it generated failures that were then
counted against the model: technical error rates ran 31–66 % per model, with message-size
limits, stream timeouts and transport hiccups among the causes.

`inproc_runner.py` keeps the part the paper is about — entity extraction, plan building,
buffer and restriction construction — and drops the rest:

| dropped | how |
|---|---|
| FastAPI + SSE | the pipeline is an async generator, iterated directly |
| Redis pipeline state | `NullPipelineStateStore` (`DISABLE_PIPELINE_STATE`) |
| ChatStorage | `persist_history=False` + `DISABLE_CHAT_HISTORY` |
| `/auth/token` proxy | the token is passed in |
| MCP over HTTP | `LocalIduMcpClient` — the IDU tools called in-process (`--transports local`) |

What still speaks HTTP is what is genuinely a separate service: **Urban API** and the
**model server**. Urban API traffic can itself be served from disk (below).

Two things the SSE harness could not do:

* **The plan is recorded.** The pipeline emits no plan event — `RestrictionsResponse` has
  no such type — so `run_benchmark.py`'s `_find_plan` scrape of the event stream had
  nothing to find, and `restriction_plan` came back null. The in-process runner captures
  the `RestrictionPlan` object where it is built, so intent, source/target entity and the
  buffer parameter can actually be scored.
* **Failures are classified at the point they are raised** — an offline data gap, an Urban
  API status, a geometry tool error, a model failure and a transport error are different
  exceptions at different stages, not one message string parsed afterwards.

`--transports local mcp-http` runs both and quantifies what the transport itself costs —
which is the measured version of the reviewer's "separate model errors from infrastructure
errors", rather than an assertion.

## Offline Urban API data

`src/idu_mcp/common/api_handlers/urban_data_store.py` stores Urban API GET responses on
disk, keyed on `(endpoint, params)` at the `JsonApiHandler` choke point. Every Urban API
call underneath is already scoped to a single entity — `?name=X`, `?service_type_id=N`,
the per-scenario catalogs — so entries are per entity and **any combination of entity
names a plan asks for is answered from pieces already stored**. (The older `ScenarioCache`
keys on the whole tuple of names one call happened to ask for, and therefore misses on
every unseen combination; it also never covered the catalogs, which the pipeline reads on
every single query.)

With `persist_history=False` this handler is the *only* channel to Urban API, so one layer
covers all of it.

| `--urban-data` | behaviour |
|---|---|
| `live` (default, and production) | always the network, nothing stored |
| `record` | fetch and store; nothing expires, nothing is evicted |
| `replay` | **never** touches the network; a miss raises `UrbanDataUnavailable` |

A replay miss is a distinct end state (`data_unavailable`) and is excluded from every
model-facing denominator. It deliberately does **not** fall back to an empty layer: an
empty layer travels through the pipeline happily and would be scored as "the model
selected no objects", turning a dropped VPN into a model failure.

**Single-user only.** The key carries no caller identity, exactly like `ScenarioCache`.
Off by default; never enable it on a deployment serving several users.

## Model server: Ollama through the OpenAI-compatible API

`LLM_BACKEND=openai` (the default) points at any OpenAI-compatible server, Ollama's own
`/v1` included; a bare origin gets `/v1` appended. One thing does not survive that switch:
**`num_ctx` is not an OpenAI parameter and is dropped**, so the context length is whatever
the server was started with — 4096 on Ollama by default. At 4096 the restrictions context
folds or the request comes back 400, and it lands in the table as a model failure.

So set it on the server and tell the harness the same number:

```bash
# on the Ollama host
OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_KEEP_ALIVE=-1 ollama serve
# in the run's environment
export MODEL_CONTEXT_TOKENS="gemma3:12b=32768,gpt-oss:20b=32768"
```

A mismatch between those two numbers manufactures failures.

### gpt-oss:20b needs `OPENAI_THINK_OFF_EFFORT=low`, or it scores 0 %

The planner calls the model with `think=False` and a JSON-schema
`response_format`. `think=False` becomes `reasoning_effort`, whose default here
is `"none"` — and on Ollama's `/v1`, `reasoning_effort="none"` **together with**
a `json_schema` response format makes gpt-oss:20b return an empty `content` with
`finish_reason="stop"`. Neither an error nor a 400: just nothing.

The plan then fails validation on every retry, the builder raises, and the row
lands in the table as a failure of the model. Measured on three rows:

| `OPENAI_THINK_OFF_EFFORT` | outcome |
|---|---|
| `none` (default) | 3/3 `planning_failure` — empty completion, no plan |
| `low` | `full_success`, correct plan, 622 objects selected in the zone |

**It cannot be set globally, and the two models want opposite values.** Ollama
rejects `reasoning_effort` outright on a model that has no thinking mode:

```
400 {"error": {"message": "\"gemma3:12b\" does not support thinking"}}
```

so `low` costs gemma3 every row — 95/95 `planning_failure` in 0.5 s each — while
`none` costs gpt-oss every row. The setting is per process, and the runner walks
its `--models` inside one process, so **the two models are two invocations**:

```bash
# gemma3:12b — the default ("none"); "low" is a 400 on this model
uv run python benchmarks/harness/inproc_runner.py --models gemma3:12b …
# gpt-oss:20b — "none" returns empty completions
OPENAI_THINK_OFF_EFFORT=low \
uv run python benchmarks/harness/inproc_runner.py --models gpt-oss:20b …
```

Both write into the same `--out-dir` (one subdirectory per model), so the report
reads them together. Check a new model the cheap way before committing a run to
it — one request carrying the schema, then read `content`: empty means that arm
would have been all zeros, a 400 means the toggle is wrong for it.

### On a local workstation (Windows, the Ollama desktop app)

The desktop app has no `ollama serve` command line to put variables in front of, and
`[Environment]::SetEnvironmentVariable(...,'User')` does **not** reach a process started
from the PowerShell session that set it — that session's environment block was captured
at launch. Set them on the session first, then start the app:

```powershell
Get-Process 'ollama app','ollama' | Stop-Process -Force
$env:OLLAMA_CONTEXT_LENGTH='16384'; $env:OLLAMA_MAX_LOADED_MODELS='1'
$env:OLLAMA_FLASH_ATTENTION='1';    $env:OLLAMA_KV_CACHE_TYPE='q8_0'
Start-Process "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe"
```

Verify by measurement, not by reading it back: `/api/ps` reports the model's *maximum*
context, not the one the runner was loaded with, so it says `131072` either way. Send a
prompt longer than 4096 tokens and read `prompt_eval_count` from the reply — if it comes
back near the full prompt length the server is not truncating.

`OLLAMA_MAX_LOADED_MODELS=1` matters on a single GPU: the runner walks the models one
after another, and without it the previous one keeps its VRAM and the next is pushed onto
the CPU. Kill the app rather than only `ollama.exe` — the `llama-server.exe` child
survives its parent, and an orphan holds a full model's worth of VRAM while answering
nothing. `--concurrency 1` for the same reason: one request in flight per server.

## Running

```bash
# 0. Fill the offline store — once, with the VPN up
python benchmarks/harness/prefetch_scenarios.py \
    --dataset benchmarks/data/gold/exp_data.csv \
    --scope catalog \
    --token "$URBAN_API_JWT" --urban-api-url "$URBAN_API_URL"
#   writes runtime/urban_data/ + a gaps.json report; re-running fills only the gaps
#   scope=catalog is the one to use: it costs minutes and a few tens of MB (the
#   scenario layers are small), and it covers every entity a valid plan can name,
#   so no legitimate model choice is charged to the store as data_unavailable

# 0b. Cut the gold set down to what this pipeline can be asked (95 of 202)
python benchmarks/eval/restrictions_filter.py
#   writes data/gold/exp_data_restrictions.csv + out/dataset_filter.md
#   needs the store from step 0 — the catalogs are read from it

# 1. The re-run — offline, in-process, both ablation arms
python benchmarks/harness/inproc_runner.py \
    --dataset benchmarks/data/gold/exp_data_restrictions.csv \
    --models gemma3:12b gpt-oss:20b \
    --arms base no_catalog \
    --urban-data replay \
    --llm-host http://localhost:11434 \
    --out-dir benchmarks/data/results_inproc \
    --concurrency 1 \
    --save-layers
#   results:  <out-dir>/<model>/<arm>--<transport>/results.jsonl
#   layers:   .../layers/<row>/<layer>.geojson   (feeds geometry_eval.py)

# 1b. Transport cost, on a subsample
python benchmarks/harness/inproc_runner.py --limit 30 \
    --transports local mcp-http --models gemma3:12b --urban-data replay ...

# 2. The report
python benchmarks/eval/inproc_eval.py \
    --results benchmarks/data/results_inproc \
    --gold benchmarks/data/gold/exp_data.csv \
    --out benchmarks/out/inproc_report.md

# 2b. Post-hoc screening over the existing (HTTP) results — no server needed
python benchmarks/eval/semantic_eval.py

# 3. Dataset expansion (Ollama only)
python benchmarks/harness/expand_dataset.py --model gpt-oss:20b \
    --n-paraphrases 10 --out benchmarks/data/gold/expanded.csv

# 4. Geometry (after reference GeoJSON is placed under data/gold/reference/)
python benchmarks/harness/geometry_eval.py \
    --results benchmarks/data/results_inproc/gemma3_12b/base--local/results.jsonl
```

A run resumes: rows already written **without an error** are skipped, failed rows are
redone. So refilling a data gap and re-running picks up exactly the rows that need it.

## Record schema (`results.jsonl`, one row per query)

| field | meaning |
|---|---|
| `idx`, `model`, `prompt`, `scenario_id` | the query |
| `transport`, `arm` | `local`/`mcp-http`, `base`/`no_catalog` |
| `model_plan` | the plan **the model emitted**, before canonicalisation. Entity and parameter accuracy are scored here. |
| `restriction_plan` | the plan **the pipeline agreed to execute**, after canonicalisation and mode validation |
| `layer_counts`, `layer_files` | features per produced layer, and where each was written |
| `tool_calls`, `stages` | which tools ran, and seconds spent per pipeline stage |
| `llm_response`, `clarification` | the final text |
| `error`, `error_class`, `error_stage` | the failure, its class, and the stage it hit |
| `missing_data` | which Urban API request was absent offline |
| `end_state` | one mutually-exclusive outcome |
| `duration_sec` | wall clock |

**End states** partition the rows (they sum to 100 % of an arm): `full_success`,
`partial_spatial`, `clarification`, `planning_failure`, `tool_infra_failure`, `timeout`,
`empty`, `data_unavailable`.

Success is task-aware, not a universal proxy: a `restrictions` task needs both `objects`
and `generators`, a `buffers_only` task has no `objects` layer by design and is not
penalised for lacking one — the flaw in the old completion proxy.

**Failure classes**: `data_unavailable`, `urban_api`, `tool_execution`, `invalid_plan`,
`llm_backend`, `timeout`, `token_expired`, `transport`, `other`. `other` is a gap in the
taxonomy and keeps a truncated traceback so it can be closed.

## Score the model's plan, not the pipeline's

Canonicalisation is lossy in a way that destroys evidence, and scoring the wrong
one of the two plans reports a model failure that did not happen.

`build_plan` resolves every entity against the catalog, then `_validate_mode`
flips the mode to `needs_clarification` if `target_entities` **or**
`restriction_rules` came back empty — and on that flip `target_entities` is
blanked in the returned plan. So a model that names the counted entity correctly
at the top level, but omits the *same* entity from `restriction_rules[].target_names`,
has every restriction rule dropped, the mode flipped, and its correct
`target_entities` erased on the way out.

That is not hypothetical: it is what gemma3:12b does on every row, and read off
`restriction_plan` it looks like "the model never named a target entity" — 0 %
entity accuracy for a model that named it right. gpt-oss:20b fills
`target_names` and the same rows come back `full_success`.

Both plans are therefore recorded. **`model_plan` is what the model produced and
is what entity/parameter accuracy is scored on; `restriction_plan` is what the
pipeline agreed to execute**, and the gap between them is a result in its own
right — the cost of the typed schema's strictness, which is one of the guardrails
the ablation is supposed to be measuring.

## What is scored where, and why

The gold set is **entirely `restrictions`-mode** tasks (spatial filter → count within a
zone): 200 restrictions + 2 clarification, 0 buffers_only.

| Metric | Source | Status |
|---|---|---|
| End states (mutually exclusive, =100%) | either harness | ✅ |
| Failure taxonomy (infra vs model vs backend) | in-process run (at source) | ✅ |
| Task-aware restrictions success | either harness | ✅ |
| Object-count agreement (cross-check) | results + gold NL | ⚠️ weak (NL negation/OCR noise) |
| Intent / entity roles / **buffer parameter** accuracy | in-process run (plan recorded) | ✅ (⛔ from the old HTTP runs — the plan never reached the stream) |
| Object-selection P/R/F1 + geometry IoU | reference GeoJSON | ⛔ pending upload |

## One trap in the gold set: the entity roles are inverted

`gold_parser` and `RestrictionPlan` name the two entities in opposite directions,
and `gold_parser` says so in its own comments:

| meaning | gold field | plan field | produced layer |
|---|---|---|---|
| the entity the zone is drawn **around** | `target_entity` | `source_entities` | `generators` |
| the entity **counted inside** the zone | `source_entity` | `target_entities` | `objects` |

Comparing the two `source` fields to each other looks obviously right and is
wrong: it reports near-zero entity accuracy and charges it to the models.
`inproc_eval.py` names its columns **Buffered** and **Counted** for exactly this
reason, and a test pins the convention in both directions.

## What the Urban API instance contributes to the failure column

Against `urban-api.testing.idulab.ru` the catalog prefetch of the 32 gold scenarios comes
back with three kinds of gap, none of which is the model's doing, and all of which the
taxonomy has to keep separate from it:

| gap | scenarios | what it is |
|---|---|---|
| `403 AccessDeniedError` | 420 | a private project; needs a real Urban API JWT |
| `404` on the scenario | 10071 | not present on this instance |
| `500 Object of type LineString is not JSON serializable` | 950, 1046, 1244, 1440, 1738 | a serialisation bug on the line-geometry physical objects (roads, power lines, water networks) |

The 500 is the one that matters: **19 of the 202 gold rows name an entity it affects**, so
those rows cannot produce a spatial result on this instance no matter which model plans
them. Under `--urban-data replay` they land as `data_unavailable` and are excluded from
the model-facing denominators; run them against an instance that serialises line geometry
before reading anything into the road-related failures.

## Model lifecycle (run_benchmark.py, HTTP path only)

For each requested model: if present on the Ollama host → use and keep; if absent → pull,
run, then delete (leave the shared server as found). `--keep-pulled` overrides the delete.
`inproc_runner.py` has no such lifecycle — it assumes the models are already pulled, which
on a workstation means `ollama pull gemma3:12b` (8.1 GB) and `ollama pull gpt-oss:20b`
(13.8 GB) before the run.
