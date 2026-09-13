# Isolated SDK integration stack

Builds gMART Agents/IDU MCP, ObjectEffectsAPI, IDU_DVD, NormGraph and ChatStorage from
explicit checkouts. Redis, MongoDB, Neo4j, Qdrant and MinIO run in a separate Compose
project/network. Published ports bind only to localhost; existing containers are not stopped.
Kafka is unconfigured, so the stand does not publish into or consume from shared topics.

Copy `stack.env.example` to `stack.local.env`, set checkout paths and credentials, and supply
an OpenAI-compatible embedding endpoint/model/dimension. All generative clients use the same
`LLM_BASE_URL` and `LLM_MODEL`; DVD and NormGraph share one embedding space. The generative
`local-gpu` endpoint is not an embedding server. Do not substitute random/fake embeddings for
an end-to-end document test.

The requested embedding origin is `http://a.dgx:8010`; clients append `/v1/embeddings`.
The hostname must resolve both on the host and inside Docker (corporate DNS/VPN may be needed).
Verified model: `ai-sage/Giga-Embeddings-instruct`, 2048 dimensions. The server returns 404 for
`/v1/models`, but `/v1/embeddings` returns the model ID and vectors successfully.

From this directory:

```powershell
docker compose --env-file stack.local.env build
docker compose --env-file stack.local.env up -d
python smoke.py --env-file stack.local.env --output results --mode health
python smoke.py --env-file stack.local.env --output results --mode llm
python smoke.py --env-file stack.local.env --output results --mode artifacts
python smoke.py --env-file stack.local.env --output results --mode effects --scenario 772
python smoke.py --env-file stack.local.env --output results --mode seed-documents
python smoke.py --env-file stack.local.env --output results --mode documents
```

Run `smoke.py` with the gMART development Python environment. `llm` sends only a synthetic
string. `artifacts` serializes a synthetic table and nonempty GeoJSON using gMART, stores them
through ChatStorage HTTP/MongoDB, and verifies the reopened history payloads. It creates a
test chat in the isolated database. `effects` calls the real local MCP and upstream Urban API;
it expects the known missing-normative case in scenario 772 and does not call the LLM.

`--mode data` and `--mode analysis` send real scenario tool results to the configured LLM.
Use only with authorization for that data transfer. They save SSE traces and verify exact
terminal replay. `analysis` specifically exercises the known missing-school-normative case:
separate school/kindergarten service counts, tables and layers, followed by provision. It requires
an attempted provision calculation, a concrete normative blocker and both complete artifact sets
in reopened history, then exports the verified tables/GeoJSON. It fails if the model omits a task
or returns incomplete output; see `docs/diagnostics/2026-09-13-scenario-772.md` for current results.
`--mode seed-documents` uploads a synthetic document named `LOCAL SDK TEST`
through the direct fragment ingestion API, waits for the durable queue, verifies DVD vector
search, runs NormGraph extraction/search and checks idempotent repeated sync. It leaves the
fixture in the isolated databases. This tests direct ingestion, not file parsing/OCR.
`--mode documents` compares the fixture's source clause and extracted restriction through the
real analytical orchestrator, requiring both specialists to complete, the correct numeric answer,
both evidence references, full artifacts in reopened chat history and exact terminal replay.
It sends only the synthetic fixture to the configured inference services.

`stability.py --env-file stack.local.env --output results --runs 20` runs a fixed,
sequential audit across scenario 772 paraphrases, data-only queries, the synthetic
document comparison and continuation. It fingerprints the source and running image
and does not retry failed analyses. `verify_stability.py` rechecks saved runs with
fresh authentication, separating useful results from clean goal completion, stable
artifact IDs and the requested source links. It never invokes inference.

`contract_probes.py` reproduces negative goal/replay boundary cases using only
synthetic in-memory data. `store_concurrency.py --output redis-results.json` tests
concurrent synthetic context writes on localhost:16389, with unique keys expiring
after five minutes. These diagnostic probes return exit 1 when defects reproduce;
this is not a test-runner failure. See the 2026-09-13 orchestrator stability report
for the tested snapshot, per-run results and current limitations.

Ports: Agents/UI 18000, IDU MCP 18002, Effects 18080, ChatStorage 18010, DVD 18100,
NormGraph 18020, Redis 16389, Neo4j Bolt 17687. Urban API/MCP and inference remain external.
This is a hybrid integration stand, not an offline deployment. Stop only this project with
`docker compose --env-file stack.local.env down`.

## Functional orchestrator harness

Run from the repository root with its development Python environment:

```powershell
python tests/integration/local_stack/harness.py --mode deterministic --output output/orchestrator-harness
python tests/integration/local_stack/harness.py --mode live --env-file stack.local.env --output output/orchestrator-harness-live
```

`full` runs both phases, and refuses to start inference after a deterministic failure.
The live phase always uses the same 20 inputs and order: ten mixed scenario analyses,
four data-only queries, four DVD/NormGraph comparisons, and two continuations from the
first accepted mixed analysis. Retries are internal bounded orchestrator behavior;
the harness never silently repeats a failed case. Keep source, commit and image fixed
throughout a series. Do not overwrite a previous output directory when changing a build.

| Functional boundary | Deterministic coverage | Live acceptance |
|---|---|---|
| Goal creation and next action | Actual Agents SDK with controlled provider responses; invalid outputs, scope, evidence and independent work | Three paraphrases, immutable goal and no pending requirements at a clean finish |
| Six specialists | Scenario data, provision, restriction, compliance, DVD and NormGraph; success/error/exception/clarification/suspension matrix | Actual Urban data and missing normative, plus both synthetic document sources |
| Data and arithmetic | Real typed-selection workflow with controlled MCP; entity IDs, duplicates, empty/partial selections, metric dimensions | Counts, differences, exact table/layer IDs, full persisted payloads |
| Failure and continuation | Six resource budgets, failed specialists, disconnect, saved evidence, current-invocation attempts | Two new requests retry the actual calculation without fetching successful selections again |
| Authentication and ownership | RSA JWT verification; expiry, wrong issuer, forged signature; replay and artifact endpoint isolation; atomic request claim | Fresh auth for each request and verification |
| Persistence and concurrency | Real store over fakeredis, source snapshots and stable IDs | Real Redis at 1/4/8 concurrent writers; ChatStorage HTTP/MongoDB; exact replay |
| User interface | History reconstruction, continuation anchor, stable table/layer identities; TypeScript and production build | Artifact exports and authenticated source snapshot endpoints |

`harness.json` contains phase exit codes and timings, alongside individual logs and
JUnit XML. A timeout is a failure, not a skip. `live/verified-summary.json` separates
functional usefulness from a clean controller finish and verifies that source links
resolve to confirmed source records. Missing events and skipped continuation prerequisites
fail acceptance. Windows explicitly excludes the four POSIX `fcntl` workspace tests;
run the full Python suite in a Linux container before claiming cross-platform acceptance.
The standalone `tests/unit/test_orchestrator_harness.py` matrix uses synthetic data;
the rest of `tests/unit` retains the specialists' detailed contracts and failure probes.

This is functional coverage of the supported interfaces and observed failure classes,
not proof that every possible natural-language request or provider failure will succeed.
Live stability is reported separately and only for a completed, immutable series.

The strengthened verifier also requires exactly twenty distinct requests in fixture order,
an unchanged build fingerprint, and the original clause 1.1 / version 2026 paired with
the matching NormGraph restriction (school, parking, >= 50 m). For scenario 772 it checks
the actual missing-normative payload: only school service type 22 and request territory
58. A broad calculation for unrelated catalog entries fails even if it includes schools.
New runs fingerprint the harness scripts as well as application source and image.

Controller routing defaults to `ORCHESTRATOR_CONTROL_REASONING_EFFORT=medium`;
ready analytical synthesis uses `ORCHESTRATOR_SYNTHESIS_REASONING_EFFORT=high`.
After an exhausted high-reasoning attempt, subsequent controller calls retain medium
for that request. Goal creation retains its separate medium default. Transient controller
transport errors have at most one retry; permanent errors and specialist side effects
are not replayed by that retry. All attempts share the request budget.

Failure before a valid goal now persists a blocked context before emitting the final
answer, and continuation restores the original query. Comparison tables carry the same
artifact ID in events and storage. Confirmed source snapshots can also be opened from
loaded chat history after the runtime source cache expires.
