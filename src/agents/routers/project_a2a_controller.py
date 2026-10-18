from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_project_a2a_service,
    resolve_a2a_caller,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.services.project_files.project_a2a_service import ProjectA2AService

project_a2a_router = APIRouter(prefix="/projects", tags=["projects", "a2a"])


@project_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_project_agent_card(
    request: Request,
    project_a2a_service: ProjectA2AService = Depends(get_project_a2a_service),
) -> dict[str, Any]:
    return project_a2a_service.get_agent_card(str(request.base_url))


@project_a2a_router.get("/agent.json", include_in_schema=False)
async def get_project_agent_card_legacy(
    request: Request,
    project_a2a_service: ProjectA2AService = Depends(get_project_a2a_service),
) -> dict[str, Any]:
    return project_a2a_service.get_agent_card(str(request.base_url))


@project_a2a_router.post(
    "/a2a",
    summary="Project-from-files agent — A2A JSON-RPC endpoint",
)
async def handle_project_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    project_a2a_service: ProjectA2AService = Depends(get_project_a2a_service),
    token: str = Depends(verify_bearer_token),
):
    """
    A2A JSON-RPC endpoint of the agent that creates a project from files.

    Accepts a single JSON-RPC 2.0 request or a batch array. Streaming methods
    (``SendStreamingMessage``, ``message/stream``, ``tasks/sendSubscribe``) return an
    SSE stream; all other methods return a plain JSON response.
    """
    payload_data = _payload_to_plain_data(payload)
    caller = await resolve_a2a_caller(payload_data, token)
    token = caller.pipeline_token
    if project_a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(project_a2a_service, payload_data, token)
        )
    return await project_a2a_service.handle_json_rpc(payload_data, token)


async def _stream_json_rpc_events(
    project_a2a_service: ProjectA2AService, payload: Any, token: str
):
    async for event in project_a2a_service.stream_json_rpc(payload, token):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
