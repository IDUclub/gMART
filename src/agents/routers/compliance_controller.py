from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_idu_mcp_client,
    get_optional_normgraph_mcp_client,
    get_restriction_parser_service,
)
from src.agents.dto.restriction_request_dto import RestrictionRequestDTO
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.restriction_parser_service import RestrictionParserService

compliance_router = APIRouter(prefix="/compliance", tags=["compliance"])


@compliance_router.get("/check/stream", response_class=EventSourceResponse)
async def check_compliance(
    request: Request,
    user_request: Annotated[RestrictionRequestDTO, Depends(RestrictionRequestDTO)],
    token: str = Depends(verify_bearer_token),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    normgraph_mcp_client: NormGraphMcpClient | None = Depends(
        get_optional_normgraph_mcp_client
    ),
    restriction_service: RestrictionParserService = Depends(
        get_restriction_parser_service
    ),
) -> AsyncIterable[RestrictionsResponse]:
    """Check request-scoped spatial restrictions against current scenario layers."""

    async for chunk in stream_with_error_handling(
        restriction_service.run_compliance_pipeline,
        request,
        restriction_service,
        user_request.model,
        rerun=False,
        mcp_client=idu_mcp_client,
        token=token,
        normgraph_mcp_client=normgraph_mcp_client,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        temperature=user_request.temperature,
        request_id=user_request.request_id,
    ):
        yield RestrictionsResponse(**chunk)
