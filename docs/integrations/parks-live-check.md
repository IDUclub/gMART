# Live verification

## Update — 2026-11-06

The current production settings were read from GitHub environment `production`:
`OPENAI_BASE_URL=http://10.32.11.27:8002/v1`, model `qwen3.8-27b`, and
`OPENAI_THINK_MODE=chat_template`. The old local `.env.agents` still used port
8001 and `reasoning_effort`; evaluation overrides were used without changing it.

Verified with real dependencies:

- The model listed `qwen3.8-27b` and correctly identified a synthetic image's
  red left half and blue right half through the multimodal API.
- Requests carrying priorities `-60` and `-100` were both accepted (HTTP 200).
  The gMART keys are `parks_planning` and `parks_analysis` respectively.
- The model parsed a natural-language request and a follow-up edit from one to
  two children's playgrounds, keeping the rest tag. No generation occurred
  before confirmation. Confirmation created exactly one task,
  `RgLy6XF46tH4GqUi`, which completed with 25 GeoJSON features and TEP.
- A recovery run with production model settings completed the model analysis,
  persisted result and recommendations to real ChatStorage, and read the
  checkpoint in a new gMART service instance. The resumed run made zero POSTs
  to generate. Local evidence: `tmp/parks-live/2026-11-06-production/report.json`.
- The actual gMART HTTP/ASGI SSE route was exercised with a real service bearer
  token and real Redis journal: replay matched stored frames, and a request
  with the final `Last-Event-ID` returned an empty tail. This was an in-process
  HTTP transport, not a browser/proxy test.

Long-running model calls exposed missing per-stage output budgets. Calls now
set `think=False`, with 4096 tokens for planning and 8192 for analysis. With the
production `chat_template` mode the adapter sends `enable_thinking=false`;
a live probe confirmed the model returned content without a reasoning trace.

The generator returned three playgrounds and four rest objects instead of the
requested two and one. The initial model analysis incorrectly treated repeated
tags as potential duplicates and called the requirement fulfilled. Backend
context now computes `count_comparison` from tag multiplicities and upstream
TEP counts; the prompt must report mismatches and must not invent numerical
norms or manually recompute areas from coordinates. A regression test covers
match, mismatch, and unavailable counts. The final live retry completed in
approximately 53 seconds and explicitly reported CHILDREN requested 2 / actual 3
and REST requested 1 / actual 4 as unfulfilled requirements. It saved the new
analysis in real ChatStorage and passed fresh-instance cursor replay without
another generation call. Evidence: `tmp/parks-live/2026-11-06-final/report.json`,
request `b2ffb84a-65f1-4aba-b9a4-14bd7df08c2f`. Numerical TEP labels/units remain
provider-defined; model recommendations are not a normative compliance check.

Deployment audit: `PARKS_API_URL` was absent both from the production workflow
environment and from `docker-compose.actions.yaml`. Those forwarding paths
are fixed and Compose rendering verified. The GitHub `production` variable
itself must still be set to `https://api.1.urban.actcognitive.org` before deployment;
external environment settings were not changed during this audit. External
Redis durability and the deployed reverse proxy remain operator checks.

After merging current `origin/dev`, 3021 unit tests passed. The follow-up parks
and priority regression checks were also run after the analysis changes: 62 passed.

## Historical check — 2026-11-02

Status: **partial, vision verification blocked by model connectivity**.

Real dependencies checked:

- ParkPlanner `https://api.1.urban.actcognitive.org`: `/map/checkShape`,
  `/map/generate`, `/map/getResultStatus`, `/map/getResult`, `/map/getEconomyParams`.
- Isolated Redis 7 at `localhost:16385`, with actual checkpoints and event journal.
- Real configured Keycloak and ChatStorage. Keycloak had intermittent timeouts;
  a later attempt succeeded. A real test chat was created and read back.

Task `Flt2kckjtkcfFQ4s` was created for an approximately 21,354 m² synthetic
test boundary near 30.30° E, 59.97° N, with two entry points and the requested
tags REST, CHILDREN, GREENERY. The task progressed from 0 to 1; GeoJSON and
nonempty economy parameters were returned. Upstream output is a generated
proposal, not a guarantee that the exact requested object counts will match.

A real gMART `generating` checkpoint was initialized with that existing task.
The recovery run retrieved status, GeoJSON and TEP without any POST to generate.
It emitted heartbeat events, the result layer and economy parameters, then
entered analysis. The model call failed at the connection stage.

The recovery run with ChatStorage used request
`70a39257-5249-4bd0-8f93-cab9c2274100`, test chat
`54bee2a4-f670-4308-8baf-edc5aa31813b`. The generated layer and TEP were read
back as structured `data` parts in the real chat despite the failed model call.
Raw local traces are under `tmp/parks-live/`; no credentials are published.

## Fix

Previously the result was persisted to ChatStorage only after LLM analysis.
If the model was unavailable, the generated park reached SSE readers but was
absent from chat history. The result now persists before analysis, with a
separate checkpoint for result history; recommendations are a later text message.
Retrying analysis does not append the result again. A regression test covers
failure, stored geometry, recovery, and absence of repeated generation.

## Not verified

The configured `http://10.32.11.27:8001/v1/models` returned
`ConnectError: All connection attempts failed`; `localhost:11434/api/tags` was
also unreachable. Thus no live model ID or vision capability was verified.
The historical `gpt-oss-20b` identifier in the partial trace is only a request
label; it is not evidence of a running vision model.

Still required: natural-language draft/edit/confirmation with a reachable model,
successful final recommendations, a real image-input check, and a live HTTP/SSE
reconnect scenario. The recovery check above invoked gMART in process with real
dependencies; it does not claim those remaining checks have passed.

Repeatable commands and the exact evaluation scope are in
[evaluation/parks/README.md](../../evaluation/parks/README.md).
