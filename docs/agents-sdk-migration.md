# Agents SDK execution

gMART uses OpenAI Agents SDK for Python (`openai-agents` 0.22.2 in `uv.lock`)
as its agent runtime. There is no switch back to a second execution implementation.
The selected models and inference servers remain deployment settings.

## Execution boundary

| Responsibility | Implementation |
| --- | --- |
| Named agent runs and text streaming | `src/agents/runtime/runner.py`: `Agent`, `Runner.run`, `Runner.run_streamed` |
| Typed plans, critique, context summaries, bounded JSON repair | `run_structured` and an SDK `AgentOutputSchemaBase`; existing Pydantic models and domain validators |
| Inference protocol and server-specific settings | `runtime/model.py`: an SDK `Model` over the existing OpenAI-compatible/native Ollama adapters |
| Execution of selected MCP calls | `runtime/tools.py`: `function_tool`, request-local context, one SDK run per selected operation |
| Sequential specialist delegation | `stream_planned`, called by the orchestrator; forwards existing domain events |
| Checkpoints, reconnect, cancellation flags, token refresh | Existing Redis pipeline state and service boundaries |
| Geospatial calculations, evidence checks, retrieval constraints, catalogue validation | Existing domain services and MCP servers |

All six specialist routes (`restriction`, `compliance`, `provision`, `documents`,
`norms`, `scenario_data`), the orchestrator, simple LLM calls, chat titles and the
context worker use this runtime. Services do not call inference adapters directly.
Model discovery still uses the adapters' `list`, `ps` and context metadata methods.

Simple orchestrator tasks retain sequential execution. Complex tasks use the
[analytical loop](analytical-orchestrator.md), shared budgets, evidence storage and
review of the remaining plan. RAG evidence review, answer continuation and scenario-data recovery
remain explicit domain workflows; they run their model stages through SDK agents.

`PlannedCallModel` is a deterministic SDK adapter for an **already selected** tool
or specialist step. It emits one function call without contacting inference.
`stop_on_first_tool` ends that run after the operation. This prevents an additional
LLM decision from changing the validated plan. It is not an autonomous handoff
loop, and it does not add inference requests to execute a selected operation.

Tool arguments, caller identity, MCP `meta`, GeoJSON and native results stay in
local operation context. SDK receives a completion receipt; services retain the
original results and record the same public tool names/arguments for ChatStorage.
Original exception types are re-raised outside the SDK tool boundary, including
`TokenExpiredError`. Failed tool operations are not automatically replayed by the
runtime. Existing checkpoint/token-refresh logic decides whether to retry.

## Model and deployment compatibility

- `LLM_BACKEND`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OLLAMA_API_URL` and the
  existing reasoning/context settings keep their meaning. No OpenAI-hosted model
  or API key is newly required.
- Non-strict provider JSON schemas remain supported. SDK validates results with
  Pydantic and the specialist's domain checks. Existing reasoning/budget fallback
  policies remain bounded; schema-free final attempts still validate the output.
- Provider `length`, `max_tokens`, `incomplete` and `content_filter` termination
  cannot turn a valid JSON prefix into an accepted structured result.
- Text/reasoning chunks and provider stop reasons retain the existing service
  shape. Closing a streaming run closes the provider stream. Stopping a delegated
  step on clarification prevents it from proceeding to another operation.
- Specialist REST, A2A, SSE, Redis checkpoint and MCP contracts are preserved.
  The analytical orchestrator adds optional request parameters and final metadata. Reconnect retains each pipeline's existing semantics; in particular,
  the outer orchestrator replays buffered events, not unfinished steps.
- SDK tracing is disabled per run. Requests, evidence and credentials are not sent
  to an additional telemetry endpoint.
- The top-level `agents` import belongs to OpenAI Agents SDK. Application imports
  remain `src.agents.*`; launch from the repository root as before.

Install with `uv sync --locked`. Both existing Docker build paths install the
same lockfile. The SDK requires `websockets<17`; the lockfile resolves 16.1.1.

## Validation

`tests/unit/test_agents_sdk_runtime.py` runs the real SDK with fake inference and
an HTTP mock transport. It checks endpoint/model/settings preservation, typed
repair, incomplete answers, concurrent run isolation, stream closure, exception
identity and stopping delegated work. Existing pipeline suites exercise the SDK
while mocking the inference and service boundaries.

On Windows, the unchanged `test_workspace_store.py` imports POSIX `fcntl` and
cannot be collected. Run that module on Linux. The remaining unit suite can run
with `--ignore=tests/unit/test_workspace_store.py` and dummy `SERVICE_AUTH_*`
settings for import-time dependency construction. These tests do not establish
live inference quality or replace deployment smoke tests against the configured
servers.

Design reference: [OpenAI Agents SDK models and providers](https://developers.openai.com/api/docs/guides/agents/models).
