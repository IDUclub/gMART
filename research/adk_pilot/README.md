# ADK pilot: the restriction agent on Google ADK 2.13

Research prototype, not wired into the app. It answers one question: can the
restriction pipeline run as an idiomatic ADK `Workflow` while the frontend
(REST/SSE) and A2A contracts stay exactly as they are?

## What is here

| Module | Role |
| --- | --- |
| `gmart_adk/llm.py` | `GmartLlm`, an ADK `BaseLlm` over gMART's `BaseLlmAdapter` (Ollama / OpenAI-compatible). `GmartGenerateContentConfig` adds `think`, backend options and JSON repair per agent. |
| `gmart_adk/harness.py` | Model callbacks (`request_params`, `conversation`) and `AuditPlugin`, which sees every model call and node. |
| `gmart_adk/events.py` | `gmart_event` for function nodes and `LegacyEventAdapter`, which turns ADK events back into gMART's pipeline items. |
| `gmart_adk/restriction/workflow.py` | The restriction pipeline as a `Workflow`: plan, repair loops, clarification, explanation, MCP steps, answer. |
| `gmart_adk/restriction/service.py` | `AdkRestrictionService`, a drop-in for `RestrictionParserService.run_restriction_execution_pipline`. |
| `tests/` | Parity with the current service, the A2A contract through the unchanged executor, the model bridge, graph/retry/harness/concurrency. |

Out of scope for the pilot: the compliance branch, NormGraph retrieval, chat
creation and persistence, reconnect replay from Redis, token suspension.

## Running the tests

From the repository root, in an environment with the project's dependencies:

```bash
uv pip install -r research/adk_pilot/requirements.txt
pytest research/adk_pilot/tests -q
```

Everything runs on mocks: no model, MCP server or Redis is needed.
