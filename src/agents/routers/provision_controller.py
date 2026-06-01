from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_effects_mcp_client,
    get_idu_mcp_client,
    get_provision_service,
)
from src.agents.dto.provision_request_dto import ProvisionRequestDTO
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.schema.provision_response import ProvisionResponse
from src.agents.services.provsion_service import ProvisionService

provision_router = APIRouter(prefix="/provision", tags=["provision"])


@provision_router.get(
    "/calculate_effects/stream",
    response_class=EventSourceResponse,
    summary="Stream provision effects pipeline results",
)
async def calculate_provision_effects(
    request: Request,
    user_request: Annotated[ProvisionRequestDTO, Depends(ProvisionRequestDTO)],
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    effects_mcp_client: EffectsMcpClient = Depends(get_effects_mcp_client),
    provision_service: ProvisionService = Depends(get_provision_service),
) -> AsyncIterable[ProvisionResponse]:

    async for chunk in stream_with_error_handling(
        provision_service.run_provision_pipeline,
        request,
        provision_service,
        user_request.model,
        rerun=False,
        idu_mcp_client=idu_mcp_client,
        effects_mcp_client=effects_mcp_client,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        temperature=user_request.temperature,
        request_id=user_request.request_id,
    ):
        yield ProvisionResponse(**chunk)
