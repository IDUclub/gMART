from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_effects_mcp_client,
    get_idu_mcp_client,
    get_optional_dvd_mcp_client,
    get_optional_normgraph_mcp_client,
    get_optional_urban_data_mcp_client,
    get_orchestrator_service,
)
from src.agents.dto.orchestrator_request_dto import OrchestratorRequestDTO
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient
from src.agents.schema.orchestrator_response import OrchestratorResponse
from src.agents.services.orchestrator_service import OrchestratorService

orchestrator_router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


@orchestrator_router.get(
    "/route/stream",
    response_class=EventSourceResponse,
    summary="Route a user request across gMART agents and stream aggregated results",
)
async def stream_orchestration(
    request: Request,
    user_request: Annotated[OrchestratorRequestDTO, Depends(OrchestratorRequestDTO)],
    token: str = Depends(verify_bearer_token),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    effects_mcp_client: EffectsMcpClient = Depends(get_effects_mcp_client),
    dvd_mcp_client: DvdMcpClient | None = Depends(get_optional_dvd_mcp_client),
    normgraph_mcp_client: NormGraphMcpClient | None = Depends(
        get_optional_normgraph_mcp_client
    ),
    urban_data_mcp_client: UrbanDataMcpClient | None = Depends(
        get_optional_urban_data_mcp_client
    ),
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service),
) -> AsyncIterable[OrchestratorResponse]:
    """
    Single entry point for all gMART agents.

    An LLM planner maps the request onto a sequential plan of steps over the
    restriction / provision / documents / norms / urban-data agents; each step's
    events are forwarded inside ``step_event`` envelopes and the run ends with a structured
    ``orchestrator_final`` per-step summary. When no agent fits, a single
    ``clarification`` event with a question for the user is emitted instead.
    """

    async for chunk in stream_with_error_handling(
        orchestrator_service.run_orchestration_pipeline,
        request,
        orchestrator_service,
        user_request.model,
        rerun=False,
        idu_mcp_client=idu_mcp_client,
        effects_mcp_client=effects_mcp_client,
        dvd_mcp_client=dvd_mcp_client,
        normgraph_mcp_client=normgraph_mcp_client,
        urban_data_mcp_client=urban_data_mcp_client,
        token=token,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        temperature=user_request.temperature,
    ):
        yield OrchestratorResponse(**chunk)
