from typing import Annotated, AsyncIterable

from fastapi import APIRouter, Depends
from fastapi.sse import EventSourceResponse

from src.agents.dependencies.dependencies import get_idu_mcp_client, get_restriction_parser_service
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.restriction_parser_service import RestrictionParserService
from src.agents.dto.restriction_request_dto import RestrictionRequestDTO
from src.agents.schema.restrictions_response import RestrictionsResponse

restriction_router = APIRouter(prefix="/restrictions", tags=["restrictions"])


@restriction_router.get("/generate_restrictions/stream", response_class=EventSourceResponse)
async def generate_restrictions_response(
    user_request: Annotated[RestrictionRequestDTO, Depends(RestrictionRequestDTO)],
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    restriction_service: RestrictionParserService = Depends(get_restriction_parser_service)
) -> AsyncIterable[RestrictionsResponse]:

    async for chunk in restriction_service.run_restriction_execution_pipline(
            idu_mcp_client, user_request.model, user_request.request, user_request.scenario_id
    ):
        yield RestrictionsResponse(**chunk)
