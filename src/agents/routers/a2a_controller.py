from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.dependencies.dependencies import (
    get_a2a_service,
    get_idu_mcp_client,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.a2a_service import A2AService

a2a_router = APIRouter(tags=["a2a"])


@a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_agent_card(
    request: Request,
    a2a_service: A2AService = Depends(get_a2a_service),
) -> dict[str, Any]:
    """
    A2A agent card endpoint.
    Args:
        request (Request): Incoming FastAPI request.
        a2a_service (A2AService): A2A service instance.
    Returns:
        dict[str, Any]: A2A agent card.
    """

    return a2a_service.get_agent_card(str(request.base_url))


@a2a_router.get("/.well-known/agent.json", include_in_schema=False)
async def get_legacy_agent_card(
    request: Request,
    a2a_service: A2AService = Depends(get_a2a_service),
) -> dict[str, Any]:
    """
    Legacy A2A agent card endpoint.
    Args:
        request (Request): Incoming FastAPI request.
        a2a_service (A2AService): A2A service instance.
    Returns:
        dict[str, Any]: A2A agent card.
    """

    return a2a_service.get_agent_card(str(request.base_url))


@a2a_router.post("/a2a")
async def handle_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    a2a_service: A2AService = Depends(get_a2a_service),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
):
    """
    A2A JSON-RPC endpoint.
    Args:
        payload (A2AJsonRpcPayloadDTO): A2A JSON-RPC request or batch request.
        a2a_service (A2AService): A2A service instance.
        idu_mcp_client (IduMcpClient): MCP client for geospatial tools.
    Returns:
        dict | list[dict] | EventSourceResponse: A2A JSON-RPC response or SSE stream.
    """

    payload_data = _payload_to_plain_data(payload)
    if a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(a2a_service, payload_data, idu_mcp_client),
        )

    return await a2a_service.handle_json_rpc(payload_data, idu_mcp_client)


async def _stream_json_rpc_events(
    a2a_service: A2AService,
    payload: Any,
    idu_mcp_client: IduMcpClient,
):
    """
    Function converts A2A stream events to SSE payloads.
    Args:
        a2a_service (A2AService): A2A service instance.
        payload (Any): A2A JSON-RPC request.
        idu_mcp_client (IduMcpClient): MCP client for geospatial tools.
    Yields:
        dict[str, str]: SSE event data.
    """

    async for event in a2a_service.stream_json_rpc(payload, idu_mcp_client):
        yield {"data": json.dumps(event, ensure_ascii=False)}


def _payload_to_plain_data(payload: A2AJsonRpcPayloadDTO) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Function converts A2A DTO to plain dict data.
    Args:
        payload (A2AJsonRpcPayloadDTO): A2A request DTO.
    Returns:
        dict[str, Any] | list[dict[str, Any]]: Plain JSON-compatible data.
    """

    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
