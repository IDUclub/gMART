from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    get_scenario_data_a2a_service,
    get_urban_mcp_client,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
from src.agents.services.scenario_data_a2a_service import ScenarioDataA2AService

scenario_data_a2a_router = APIRouter(
    prefix="/scenario-data", tags=["scenario-data", "a2a"]
)


@scenario_data_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_scenario_data_agent_card(
    request: Request,
    service: ScenarioDataA2AService = Depends(get_scenario_data_a2a_service),
) -> dict[str, Any]:
    return service.get_agent_card(str(request.base_url))


@scenario_data_a2a_router.get("/agent.json", include_in_schema=False)
async def get_scenario_data_agent_card_legacy(
    request: Request,
    service: ScenarioDataA2AService = Depends(get_scenario_data_a2a_service),
) -> dict[str, Any]:
    return service.get_agent_card(str(request.base_url))


@scenario_data_a2a_router.post(
    "/a2a", summary="Scenario-data agent — A2A JSON-RPC endpoint"
)
async def handle_scenario_data_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    service: ScenarioDataA2AService = Depends(get_scenario_data_a2a_service),
    urban_mcp_client: UrbanMcpClient = Depends(get_urban_mcp_client),
    token: str = Depends(verify_bearer_token),
):
    payload_data = _payload_to_plain_data(payload)
    if service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(service, payload_data, urban_mcp_client, token)
        )
    return await service.handle_json_rpc(payload_data, urban_mcp_client, token)


async def _stream_json_rpc_events(
    service: ScenarioDataA2AService,
    payload: Any,
    urban_mcp_client: UrbanMcpClient,
    token: str,
):
    async for event in service.stream_json_rpc(payload, urban_mcp_client, token):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
