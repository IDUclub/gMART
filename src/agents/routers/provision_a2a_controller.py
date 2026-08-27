from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.a2a.a2a_format import synapse_prepopulated_params_agent_card
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    a2a_effects_mcp_client,
    a2a_idu_mcp_client,
    get_effects_mcp_client,
    get_idu_mcp_client,
    get_provision_a2a_service,
    resolve_a2a_caller,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.provision_a2a_service import ProvisionA2AService

provision_a2a_router = APIRouter(prefix="/provision", tags=["provision", "a2a"])


@provision_a2a_router.get("/.well-known/agent-card.json", include_in_schema=False)
async def get_provision_agent_card(
    request: Request,
    provision_a2a_service: ProvisionA2AService = Depends(get_provision_a2a_service),
) -> dict[str, Any]:
    return provision_a2a_service.get_agent_card(str(request.base_url))


@provision_a2a_router.get(
    "/.well-known/synapse-agent-card.json", include_in_schema=False
)
async def get_synapse_provision_agent_card(
    request: Request,
    provision_a2a_service: ProvisionA2AService = Depends(get_provision_a2a_service),
) -> dict[str, Any]:
    return synapse_prepopulated_params_agent_card(
        provision_a2a_service.get_agent_card(str(request.base_url))
    )


@provision_a2a_router.get("/agent.json", include_in_schema=False)
async def get_provision_agent_card_legacy(
    request: Request,
    provision_a2a_service: ProvisionA2AService = Depends(get_provision_a2a_service),
) -> dict[str, Any]:
    return provision_a2a_service.get_agent_card(str(request.base_url))


@provision_a2a_router.post("/a2a")
async def handle_provision_a2a_json_rpc(
    payload: A2AJsonRpcPayloadDTO = Body(...),
    provision_a2a_service: ProvisionA2AService = Depends(get_provision_a2a_service),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    effects_mcp_client: EffectsMcpClient = Depends(get_effects_mcp_client),
    token: str = Depends(verify_bearer_token),
):
    payload_data = _payload_to_plain_data(payload)
    caller = await resolve_a2a_caller(payload_data, token)
    if caller.is_synapse:
        idu_mcp_client = await a2a_idu_mcp_client(caller.user_id)
        effects_mcp_client = await a2a_effects_mcp_client(caller.user_id)
    token = caller.pipeline_token
    if provision_a2a_service.is_streaming_request(payload_data):
        return EventSourceResponse(
            _stream_json_rpc_events(
                provision_a2a_service,
                payload_data,
                idu_mcp_client,
                effects_mcp_client,
                token,
            )
        )
    return await provision_a2a_service.handle_json_rpc(
        payload_data, idu_mcp_client, effects_mcp_client, token
    )


async def _stream_json_rpc_events(
    provision_a2a_service: ProvisionA2AService,
    payload: Any,
    idu_mcp_client: IduMcpClient,
    effects_mcp_client: EffectsMcpClient,
    token: str,
):
    async for event in provision_a2a_service.stream_json_rpc(
        payload, idu_mcp_client, effects_mcp_client, token
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _payload_to_plain_data(
    payload: A2AJsonRpcPayloadDTO,
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item.model_dump(mode="json", exclude_none=True) for item in payload]
    return payload.model_dump(mode="json", exclude_none=True)
