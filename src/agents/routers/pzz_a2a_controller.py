from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    a2a_pzz_mcp_client,
    get_pzz_a2a_service,
    get_pzz_mcp_client,
    resolve_a2a_caller,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.pzz_mcp_client import PzzMcpClient
from src.agents.services.pzz.pzz_a2a_service import PzzA2AService

pzz_a2a_router = APIRouter(prefix="/pzz", tags=["pzz", "a2a"])


@pzz_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_pzz_agent_card(
    request: Request,
    pzz_a2a_service: PzzA2AService = Depends(get_pzz_a2a_service),
) -> dict[str, Any]:
    return pzz_a2a_service.get_agent_card(str(request.base_url))


@pzz_a2a_router.get("/agent.json", include_in_schema=False)
async def get_pzz_agent_card_legacy(
    request: Request,
    pzz_a2a_service: PzzA2AService = Depends(get_pzz_a2a_service),
) -> dict[str, Any]:
    return pzz_a2a_service.get_agent_card(str(request.base_url))


@pzz_a2a_router.post(
    "/a2a",
    summary="PZZ compliance agent — A2A JSON-RPC endpoint",
)
async def handle_pzz_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    pzz_a2a_service: PzzA2AService = Depends(get_pzz_a2a_service),
    pzz_mcp_client: PzzMcpClient = Depends(get_pzz_mcp_client),
    token: str = Depends(verify_bearer_token),
):
    """
    A2A JSON-RPC endpoint for the PZZ compliance agent.

    Accepts a single JSON-RPC 2.0 request or a batch array. Streaming methods
    (``SendStreamingMessage``, ``message/stream``, ``tasks/sendSubscribe``) return an
    SSE stream; all other methods return a plain JSON response.
    """
    payload_data = _payload_to_plain_data(payload)
    caller = await resolve_a2a_caller(payload_data, token)
    if caller.is_synapse:
        pzz_mcp_client = await a2a_pzz_mcp_client(caller.user_id)
    token = caller.pipeline_token
    if pzz_a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(
                pzz_a2a_service, payload_data, pzz_mcp_client, token
            )
        )
    return await pzz_a2a_service.handle_json_rpc(payload_data, pzz_mcp_client, token)


async def _stream_json_rpc_events(
    pzz_a2a_service: PzzA2AService,
    payload: Any,
    pzz_mcp_client: PzzMcpClient,
    token: str,
):
    async for event in pzz_a2a_service.stream_json_rpc(payload, pzz_mcp_client, token):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
