# In-process evaluation — restrictions pipeline vs expert gold

Runs driven in code (no HTTP interface, no Redis, no ChatStorage). Every percentage below is over rows that say something about the model: rows whose data was unavailable offline are excluded and reported separately in Table 0.


## Table 0. Coverage

`data_unavailable` means the offline store had no answer for a request the plan made — a gap in the prefetched data, not a model failure. It is excluded from every other table. A non-zero column here is an instruction to re-run `prefetch_scenarios.py`, not a result.

| Run | Rows | Scored | Data gaps | Gold rows | Median s |
|---|---|---|---|---|---|
| gemma4:12b / no_catalog / local | 95 | 95 | 0 | 95 | 9.2 |


## Table 1. Task-aware success by task type (gold set)

Success is what the task actually requires: a `restrictions` task needs both output layers and an answer; a `buffers_only` task has no `objects` layer by design and is not penalised for lacking one; a `needs_clarification` task succeeds by asking. The single universal completion proxy the previous report used understates the first and overstates nothing — which is why it is replaced rather than kept.

The criterion comes from the gold task, never from the mode the model declared for itself. Reading it from the model's own mode lets a run score by lowering the bar — see Table 1b.

| Run | Buffers-only | Restrictions | Clarification | Overall |
|---|---|---|---|---|
| gemma4:12b / no_catalog / local | n/a | 0.0% (0/95) | n/a | 0.0% (0/95) |


## Table 1b. Mode evasion

Rows whose gold task asks for a count and where the model declared `buffers_only` — it draws the zone and never counts anything inside it. The pipeline runs such a plan happily and it produces a layer, so a completion test that trusts the declared mode reads it as a success. **Reported** is what that test would have said; **actual** is the same rows judged by what the task asked for. A large gap between the two columns means the model is answering an easier question than the one put to it, and any success rate quoted for it is about the substitution rather than about the task.

| Run | Declared buffers_only | Reported success | Actual success |
|---|---|---|---|
| gemma4:12b / no_catalog / local | 0 (0.0%) | 0 (0.0%) | 0.0% (0/95) |


## Table 2. Plan correctness (gold set)

Scored against the `RestrictionPlan` **the model emitted** (`model_plan`), not the one the pipeline went on to execute. The two differ: canonicalisation drops restriction rules that carry no `target_names`, an empty rule list flips the mode to `needs_clarification`, and that flip blanks `target_entities`. Scoring the executed plan therefore reports a missing target entity for a model that named it correctly — see Table 2b. Each field is scored only on gold rows the parser is confident about, and the comparable count is shown next to the percentage. Rows that failed before a plan existed are not counted here — they are already counted as planning failures in Table 3.

The entity columns are named by role rather than by either side's field name, because the two disagree: the gold set calls the buffered entity `target` while the plan schema builds buffers around `source_entities`. **Buffered** is the entity the zone is drawn around (plan `source_entities`, gold `target_entity`); **counted** is the entity found inside it (plan `target_entities`, gold `source_entity`).

| Run | Intent | Buffered entity | Counted entity | Distance |
|---|---|---|---|---|
| gemma4:12b / no_catalog / local | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) | 0.0% (0/95) |


## Table 2b. What the typed schema costs

Rows where the model planned `restrictions` and the pipeline refused the plan. These are not model errors of intent or of entity choice — Table 2 scores those on the model's own plan — but failures to satisfy the schema exactly, and the pipeline treats a partial plan as no plan at all. `missing target_names` is the observed cause: the counted entity is named in `target_entities` and omitted from `restriction_rules[].target_names`, which must agree.

| Run | Plans downgraded | of which: missing `target_names` |
|---|---|---|
| gemma4:12b / no_catalog / local | 0 | 0 |


## Table 3. Mutually-exclusive end states (% of scored rows, sums to 100)

| Run | full_success | partial_spatial | clarification | planning_failure | tool_infra_failure | timeout | empty |
|---|---|---|---|---|---|---|---|
| gemma4:12b / no_catalog / local | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 |


## Table 4. Failure taxonomy (row counts)

Each class is decided where the exception was raised, not by matching the error message afterwards. `other` is an unclassified failure — a gap in this taxonomy — and each such record keeps a truncated traceback.

| Run | invalid_plan | unexecutable_plan | llm_backend | tool_execution | urban_api | transport | timeout | token_expired | other | model | infra |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b / no_catalog / local | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |


## Not computable here
- **Object-selection P/R/F1 and geometry IoU** — scored against the reference GeoJSON by `benchmarks/harness/geometry_eval.py`, which reads the layers this run wrote to disk (`--save-layers`).
