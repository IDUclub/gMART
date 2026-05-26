from __future__ import annotations

import json
from collections.abc import AsyncIterable
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_effects_mcp_client,
    get_idu_mcp_client,
    get_orchestrator_a2a_service,
    get_orchestrator_pipeline_service,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.dto.orchestrator_request_dto import OrchestratorRequestDTO
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.schema.orchestrator_response import OrchestratorResponse
from src.agents.services.orchestrator_a2a_service import OrchestratorA2AService
from src.agents.services.orchestrator_pipeline_service import (
    OrchestratorPipelineService,
)

orchestrator_router = APIRouter(prefix="/orchestrator", tags=["orchestrator", "a2a"])


# ── Frontend REST endpoint ────────────────────────────────────────────────────


@orchestrator_router.get(
    "/run/stream",
    response_class=EventSourceResponse,
    summary="Orchestrator — frontend SSE streaming endpoint",
)
async def run_orchestrator_stream(
    request: Request,
    user_request: Annotated[OrchestratorRequestDTO, Depends(OrchestratorRequestDTO)],
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    effects_mcp_client: EffectsMcpClient = Depends(get_effects_mcp_client),
    pipeline_service: OrchestratorPipelineService = Depends(
        get_orchestrator_pipeline_service
    ),
) -> AsyncIterable[OrchestratorResponse]:
    """
    Frontend SSE endpoint for the orchestrator agent.

    Classifies the user query via LLM and delegates to
    **restriction-creation-agent** and/or **provision-effects-agent**.
    Streams merged events from all invoked sub-pipelines.

    **Query parameters:**

    | Parameter    | Required | Description                                         |
    |---|---|---|
    | request      | ✅       | Natural-language user query                         |
    | scenario_id  | ✅       | Urban API scenario ID                               |
    | project_id   |          | Project ID for provision effects (skip if absent)   |
    | model        |          | Ollama model name (default: gpt-oss:20b)             |
    | temperature  |          | LLM temperature (default: 1.0)                      |
    | chat_id      |          | Existing Chat Storage UUID for history continuity   |
    | request_id   |          | Pipeline request ID to reconnect a suspended run    |

    **Event stream** emits ``OrchestratorResponse`` objects with ``type``:
    - ``routing`` — orchestrator classification / delegation status
    - ``status`` — sub-pipeline step status
    - ``chunk`` — LLM text chunk
    - ``feature_collection`` — GeoJSON layer
    - ``error``, ``pipeline_started``, ``token_expired``, ``pipeline_suspended``,
      ``service_event``, ``tool_call`` — standard pipeline events
    """
    async for chunk in stream_with_error_handling(
        pipeline_service.run_orchestrator_pipeline,
        request,
        pipeline_service,
        user_request.model,
        rerun=False,
        idu_mcp_client=idu_mcp_client,
        effects_mcp_client=effects_mcp_client,
        temperature=user_request.temperature,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        project_id=user_request.project_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
    ):
        yield OrchestratorResponse(**chunk)


# ── Agent card discovery ──────────────────────────────────────────────────────


@orchestrator_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_orchestrator_agent_card(
    request: Request,
    service: OrchestratorA2AService = Depends(get_orchestrator_a2a_service),
) -> dict[str, Any]:
    return service.get_agent_card(str(request.base_url))


@orchestrator_router.get("/agent.json", include_in_schema=False)
async def get_orchestrator_agent_card_legacy(
    request: Request,
    service: OrchestratorA2AService = Depends(get_orchestrator_a2a_service),
) -> dict[str, Any]:
    return service.get_agent_card(str(request.base_url))


# ── JSON-RPC endpoint ─────────────────────────────────────────────────────────


@orchestrator_router.post(
    "/a2a",
    summary="Orchestrator agent — A2A JSON-RPC endpoint",
)
async def handle_orchestrator_a2a(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    service: OrchestratorA2AService = Depends(get_orchestrator_a2a_service),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    effects_mcp_client: EffectsMcpClient = Depends(get_effects_mcp_client),
):
    """
    Orchestrator A2A JSON-RPC endpoint.

    Accepts a JSON-RPC 2.0 request or a batch array.
    Routes the user query to **restriction-creation-agent** and/or
    **provision-effects-agent** based on LLM intent classification.

    **Required in params.metadata (or message.metadata):**
    - `scenario_id` (int) — Urban API scenario ID

    Streaming methods (`tasks/sendSubscribe`, `message/stream`,
    `SendStreamingMessage`) return an SSE stream; all other methods
    return a plain JSON response.

    The SSE stream interleaves events from all invoked sub-agents:
    - Orchestrator status updates (routing steps)
    - Sub-agent task creation, status updates, and GeoJSON artifacts
    """
    payload_data = _payload_to_plain_data(payload)
    if service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_events(service, payload_data, idu_mcp_client, effects_mcp_client)
        )
    return await service.handle_json_rpc(
        payload_data, idu_mcp_client, effects_mcp_client
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _stream_events(
    service: OrchestratorA2AService,
    payload: Any,
    idu_mcp_client: IduMcpClient,
    effects_mcp_client: EffectsMcpClient,
):
    async for event in service.stream_json_rpc(
        payload, idu_mcp_client, effects_mcp_client
    ):
        yield {"data": json.dumps(event, ensure_ascii=False)}


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
