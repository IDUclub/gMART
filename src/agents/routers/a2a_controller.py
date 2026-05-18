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

# Canonical restriction A2A router — /restriction prefix, consistent with /provision/a2a
restriction_a2a_router = APIRouter(prefix="/restriction", tags=["restriction", "a2a"])

# Legacy root-level router — kept for backward compatibility, marked deprecated
a2a_router = APIRouter(tags=["a2a"])


# ── Agent card discovery ──────────────────────────────────────────────────────


@restriction_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
@a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_agent_card(
    request: Request,
    a2a_service: A2AService = Depends(get_a2a_service),
) -> dict[str, Any]:
    return a2a_service.get_agent_card(str(request.base_url))


@restriction_a2a_router.get("/agent.json", include_in_schema=False)
@a2a_router.get("/.well-known/agent.json", include_in_schema=False)
async def get_legacy_agent_card(
    request: Request,
    a2a_service: A2AService = Depends(get_a2a_service),
) -> dict[str, Any]:
    return a2a_service.get_agent_card(str(request.base_url))


# ── JSON-RPC handler (shared logic) ──────────────────────────────────────────


async def _handle_restriction_a2a(
    payload: A2AJsonRpcPayloadDTO,
    a2a_service: A2AService,
    idu_mcp_client: IduMcpClient,
):
    payload_data = _payload_to_plain_data(payload)
    if a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(a2a_service, payload_data, idu_mcp_client),
        )
    return await a2a_service.handle_json_rpc(payload_data, idu_mcp_client)


# ── Canonical endpoint ────────────────────────────────────────────────────────


@restriction_a2a_router.post(
    "/a2a",
    summary="Restriction agent — A2A JSON-RPC endpoint",
)
async def handle_restriction_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    a2a_service: A2AService = Depends(get_a2a_service),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
):
    """
    Canonical A2A JSON-RPC endpoint for the restriction generation agent.

    Accepts a single JSON-RPC 2.0 request or a batch array.
    Streaming methods (``SendStreamingMessage``, ``message/stream``,
    ``tasks/sendSubscribe``) return an SSE stream; all other methods
    return a plain JSON response.
    """
    return await _handle_restriction_a2a(payload, a2a_service, idu_mcp_client)


# ── Deprecated root-level endpoint ───────────────────────────────────────────


@a2a_router.post(
    "/a2a",
    deprecated=True,
    summary="[Deprecated] Restriction agent — A2A JSON-RPC endpoint",
)
async def handle_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    a2a_service: A2AService = Depends(get_a2a_service),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
):
    """
    **Deprecated.** Use ``POST /restriction/a2a`` instead.

    Kept for backward compatibility. Will be removed in a future release.
    """
    return await _handle_restriction_a2a(payload, a2a_service, idu_mcp_client)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _stream_json_rpc_events(
    a2a_service: A2AService,
    payload: Any,
    idu_mcp_client: IduMcpClient,
):
    async for event in a2a_service.stream_json_rpc(payload, idu_mcp_client):
        yield {"data": json.dumps(event, ensure_ascii=False)}


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
