# Local SDK/service integration — 2026-09-12

Fix checkouts: ObjectEffectsAPI `fix/missing-service-normatives`, NormGraph
`fix/reconcile-incomplete-extraction`, gMART `fix/local-sdk-integration`.

## Verified

- Six application images build from source. Local Agents, IDU MCP, ObjectEffectsAPI,
  ChatStorage, Redis, MongoDB, Neo4j, Qdrant and MinIO start independently of existing containers.
- Graphify provider configuration confirms `local-gpu` at `http://10.32.11.27:8001/v1`,
  model `gpt-oss-20b`. `/v1/models` reports a 65536-token context window.
- Local Agents HTTP/SSE → Agents SDK → that vLLM returns `LOCAL_SDK_OK` from a synthetic prompt.
- Local ObjectEffectsAPI MCP → dev Urban API, scenario 772/service 22: missing normative now
  returns `missing_service_normative` with requested territory/service IDs and `required_action`.
  No LLM is called in this check. The previous version fails the same empty-response regression
  with `KeyError: service_type`; the patched version passes.
- gMART artifact serialization → local ChatStorage HTTP → MongoDB → reopened chat history:
  a synthetic table and nonempty GeoJSON match exactly. The local ChatStorage configuration
  needed explicit JWT audiences `urban-api,account`; signature/issuer verification stays enabled.
- ObjectEffectsAPI: 9 tests pass, including preserving another service's layers on partial failure.
- NormGraph: recovery tests cover failed/partial extraction, zero-result completion, unchanged
  source replay, and skipped ingestion. Two live Neo4j lifecycle tests pass, including completion
  reset on source hash change.

## Not yet verified

- DVD/NormGraph document ingestion and retrieval over the complete running stack require a
  configured OpenAI-compatible embedding endpoint. The specified chat server returns 404 for
  `/v1/embeddings`; legacy local configs reference Ollama/bge-m3. Their images were built, but
  these two API containers were not started with a fabricated replacement endpoint.
- The full real-scenario analytical SSE run did not execute: automatic approval review rejected
  transfer of scenario 772 data/layers to the remote model pending explicit data-transfer consent.
  No workaround or indirect replay of that rejected request was performed.
- No normative corpus was fabricated or inserted into the shared dev contour. Empty Urban API
  normatives and an empty normative corpus still need legitimate source data; code cannot supply it.

See `tests/integration/local_stack/README.md` for reproduction. Test outputs and credentials are
local-only and excluded from commits. This report does not claim a complete analytical/dev pass.
