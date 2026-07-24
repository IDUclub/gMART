from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_urban_data_a2a_service,
    get_urban_data_mcp_client,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient
from src.agents.services.urban_data_a2a_service import UrbanDataA2AService

urban_data_a2a_router = APIRouter(prefix="/urban-data", tags=["urban-data", "a2a"])


@urban_data_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_urban_data_agent_card(
    request: Request,
    urban_data_a2a_service: UrbanDataA2AService = Depends(get_urban_data_a2a_service),
) -> dict[str, Any]:
    return urban_data_a2a_service.get_agent_card(str(request.base_url))


@urban_data_a2a_router.get("/agent.json", include_in_schema=False)
async def get_urban_data_agent_card_legacy(
    request: Request,
    urban_data_a2a_service: UrbanDataA2AService = Depends(get_urban_data_a2a_service),
) -> dict[str, Any]:
    return urban_data_a2a_service.get_agent_card(str(request.base_url))


@urban_data_a2a_router.post(
    "/a2a",
    summary="Urban-data QA agent — A2A JSON-RPC endpoint",
)
async def handle_urban_data_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    urban_data_a2a_service: UrbanDataA2AService = Depends(get_urban_data_a2a_service),
    urban_data_mcp_client: UrbanDataMcpClient = Depends(get_urban_data_mcp_client),
    token: str = Depends(verify_bearer_token),
):
    """
    A2A JSON-RPC endpoint for the urban-data QA agent.

    Accepts a single JSON-RPC 2.0 request or a batch array. Streaming methods
    (``SendStreamingMessage``, ``message/stream``, ``tasks/sendSubscribe``) return an
    SSE stream; all other methods return a plain JSON response.
    """
    payload_data = _payload_to_plain_data(payload)
    if urban_data_a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(
                urban_data_a2a_service, payload_data, urban_data_mcp_client, token
            )
        )
    return await urban_data_a2a_service.handle_json_rpc(
        payload_data, urban_data_mcp_client, token
    )


async def _stream_json_rpc_events(
    urban_data_a2a_service: UrbanDataA2AService,
    payload: Any,
    urban_data_mcp_client: UrbanDataMcpClient,
    token: str,
):
    async for event in urban_data_a2a_service.stream_json_rpc(
        payload, urban_data_mcp_client, token
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
