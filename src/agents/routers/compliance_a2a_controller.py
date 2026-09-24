from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.a2a.compliance_executor import ComplianceMcpClients
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    a2a_idu_mcp_client,
    a2a_normgraph_mcp_client,
    get_app_config,
    get_compliance_a2a_service,
    get_idu_mcp_client,
    get_optional_normgraph_mcp_client,
    resolve_a2a_caller,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.services.compilance.compliance_a2a_service import (
    ComplianceA2AService,
)

compliance_a2a_router = APIRouter(prefix="/compliance", tags=["compliance", "a2a"])


@compliance_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_compliance_agent_card(
    request: Request,
    compliance_a2a_service: ComplianceA2AService = Depends(get_compliance_a2a_service),
) -> dict[str, Any]:
    return compliance_a2a_service.get_agent_card(str(request.base_url))


@compliance_a2a_router.get("/agent.json", include_in_schema=False)
async def get_compliance_agent_card_legacy(
    request: Request,
    compliance_a2a_service: ComplianceA2AService = Depends(get_compliance_a2a_service),
) -> dict[str, Any]:
    return compliance_a2a_service.get_agent_card(str(request.base_url))


@compliance_a2a_router.post(
    "/a2a",
    summary="Normative compliance agent — A2A JSON-RPC endpoint",
)
async def handle_compliance_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    compliance_a2a_service: ComplianceA2AService = Depends(get_compliance_a2a_service),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    normgraph_mcp_client: NormGraphMcpClient | None = Depends(
        get_optional_normgraph_mcp_client
    ),
    token: str = Depends(verify_bearer_token),
):
    """
    A2A JSON-RPC endpoint for the normative compliance check of scenario objects.

    Accepts a single JSON-RPC 2.0 request or a batch array. Streaming methods
    (``SendStreamingMessage``, ``message/stream``, ``tasks/sendSubscribe``) return an
    SSE stream; all other methods return a plain JSON response. A document choice
    ends the task in ``input-required``; the next message with the same ``contextId``
    answers it.
    """
    payload_data = _payload_to_plain_data(payload)
    caller = await resolve_a2a_caller(payload_data, token)
    if caller.is_synapse:
        idu_mcp_client = await a2a_idu_mcp_client(caller.user_id)
        if get_app_config().NORM_GRAPH_MCP_URL:
            normgraph_mcp_client = await a2a_normgraph_mcp_client(caller.user_id)
    clients = ComplianceMcpClients(idu=idu_mcp_client, normgraph=normgraph_mcp_client)
    token = caller.pipeline_token
    if compliance_a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(
                compliance_a2a_service, payload_data, clients, token
            )
        )
    return await compliance_a2a_service.handle_json_rpc(payload_data, clients, token)


async def _stream_json_rpc_events(
    compliance_a2a_service: ComplianceA2AService,
    payload: Any,
    clients: ComplianceMcpClients,
    token: str,
):
    async for event in compliance_a2a_service.stream_json_rpc(payload, clients, token):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
