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

- The full real-scenario analytical SSE run did not execute: automatic approval review rejected
  transfer of scenario 772 data/layers to the remote model pending explicit data-transfer consent.
  No workaround or indirect replay of that rejected request was performed.
- No normative corpus was fabricated or inserted into the shared dev contour. Empty Urban API
  normatives and an empty normative corpus still need legitimate source data; code cannot supply it.

See `tests/integration/local_stack/README.md` for reproduction. Test outputs and credentials are
local-only and excluded from commits. This report does not claim a complete analytical/dev pass.

## Follow-up — 2026-09-13

The corporate VPN resolved the embedding connectivity blocker: `a.dgx` resolves to `10.32.2.3`
on Windows and inside Docker. POST `http://a.dgx:8010/v1/embeddings` returns 200 with model
`ai-sage/Giga-Embeddings-instruct`, dimension 2048. GET `/v1/models` returns 404 on this server.
DVD and NormGraph now run with this shared embedding space; all six application health checks pass.

The synthetic `LOCAL SDK TEST` document completed direct fragment ingestion through DVD's
durable queue and vector retrieval. NormGraph extracted one restriction through the configured
remote vLLM, persisted it in Neo4j, and found it by vector search: school building to open car
park distance >= 50 m, source clause 1.1. Repeated sync skipped the completed extraction and
preserved the count. Only the isolated local corpus was seeded; parsing/OCR was not exercised.

The analytical smoke test exposed two application defects, now covered by regressions:
- DVD's ambiguous `[N]` instruction produced unsupported `[N1]` citations. Drafting now specifies
  literal numeric source labels, and malformed labels receive a deterministic correction before
  the semantic audit. Source-grounding review remains mandatory.
- The analytical reviewer attempted numeric table comparisons using `analysis_text` artifacts.
  Invalid references now trigger at most two corrective review calls using preserved evidence,
  without replaying specialists. Continued invalid decisions block honestly and retain artifacts.

Repeated runs also encountered vLLM HTTP 500 `unexpected tokens remaining in message header`
with `<|constrain|>analysis`. The OpenAI adapter now retries this specific non-streaming gpt-oss
high-effort failure once at medium effort, charging both attempts to the budget. Other errors and
a repeated failure still propagate. Transport regressions cover recovery and bounded failure.
NormGraph drafting/audit instructions now agree on numeric citations and include restriction IDs
when the user asks for a restriction record.

The final synthetic analytical run **passed**: both `documents` and `norms` specialists completed,
the answer compared the 50 m requirement, both evidence references were returned, and every
artifact's full content was verified in reopened ChatStorage history. Terminal SSE replay matched
exactly. Run `b37b683b-7103-4bd4-9743-c32e17324675` took about 47 s with 14 model calls, 3 tool calls,
56,936 charged tokens (including one conservatively estimated failed call), and one reasoning
fallback. This is one successful acceptance run, not a reliability claim for the remote model.

Validation: 943 gMART unit tests passed (the existing POSIX-only workspace test file is excluded
on Windows); the NormGraph prompt test passed again after the final citation change. All-file
Black/isort checks passed. Live outputs remain local-only under `output/local-sdk-stack/`.
