from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_dvd_a2a_service,
    get_dvd_mcp_client,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.services.dvd_a2a_service import DocumentQaA2AService

dvd_a2a_router = APIRouter(prefix="/documents", tags=["documents", "a2a"])


@dvd_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_dvd_agent_card(
    request: Request,
    dvd_a2a_service: DocumentQaA2AService = Depends(get_dvd_a2a_service),
) -> dict[str, Any]:
    return dvd_a2a_service.get_agent_card(str(request.base_url))


@dvd_a2a_router.get("/agent.json", include_in_schema=False)
async def get_dvd_agent_card_legacy(
    request: Request,
    dvd_a2a_service: DocumentQaA2AService = Depends(get_dvd_a2a_service),
) -> dict[str, Any]:
    return dvd_a2a_service.get_agent_card(str(request.base_url))


@dvd_a2a_router.post(
    "/a2a",
    summary="Document-QA agent — A2A JSON-RPC endpoint",
)
async def handle_dvd_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    dvd_a2a_service: DocumentQaA2AService = Depends(get_dvd_a2a_service),
    dvd_mcp_client: DvdMcpClient = Depends(get_dvd_mcp_client),
    token: str = Depends(verify_bearer_token),
):
    """
    A2A JSON-RPC endpoint for the regulatory-documents QA (RAG) agent.

    Accepts a single JSON-RPC 2.0 request or a batch array. Streaming methods
    (``SendStreamingMessage``, ``message/stream``, ``tasks/sendSubscribe``) return an
    SSE stream; all other methods return a plain JSON response.
    """
    payload_data = _payload_to_plain_data(payload)
    if dvd_a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(
                dvd_a2a_service, payload_data, dvd_mcp_client, token
            )
        )
    return await dvd_a2a_service.handle_json_rpc(payload_data, dvd_mcp_client, token)


async def _stream_json_rpc_events(
    dvd_a2a_service: DocumentQaA2AService,
    payload: Any,
    dvd_mcp_client: DvdMcpClient,
    token: str,
):
    async for event in dvd_a2a_service.stream_json_rpc(payload, dvd_mcp_client, token):
        yield {"data": json.dumps(event, ensure_ascii=False)}


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
