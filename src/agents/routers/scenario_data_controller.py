from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_scenario_data_service,
    get_urban_mcp_client,
)
from src.agents.dto.scenario_data_request_dto import ScenarioDataRequestDTO
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
from src.agents.schema.scenario_data_response import ScenarioDataResponse
from src.agents.services.scenario_data_service import ScenarioDataService

scenario_data_router = APIRouter(prefix="/scenario-data", tags=["scenario-data"])


@scenario_data_router.get(
    "/qa/stream",
    response_class=EventSourceResponse,
    summary="Answer questions about scenario data and stream Urban API layers",
)
async def stream_scenario_data(
    request: Request,
    user_request: Annotated[ScenarioDataRequestDTO, Depends(ScenarioDataRequestDTO)],
    token: str = Depends(verify_bearer_token),
    urban_mcp_client: UrbanMcpClient = Depends(get_urban_mcp_client),
    service: ScenarioDataService = Depends(get_scenario_data_service),
) -> AsyncIterable[ScenarioDataResponse]:
    async for chunk in stream_with_error_handling(
        service.run_scenario_data_pipeline,
        request,
        service,
        user_request.model,
        rerun=False,
        continue_on_disconnect=True,
        urban_mcp_client=urban_mcp_client,
        token=token,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        temperature=user_request.temperature,
    ):
        yield ScenarioDataResponse(**chunk)
