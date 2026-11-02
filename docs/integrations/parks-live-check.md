# Live verification — 2026-11-02

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
