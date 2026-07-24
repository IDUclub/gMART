from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_urban_data_mcp_client,
    get_urban_data_qa_service,
)
from src.agents.dto.urban_data_request_dto import UrbanDataQaRequestDTO
from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient
from src.agents.schema.urban_data_response import UrbanDataResponse
from src.agents.services.urban_data_qa_service import UrbanDataQaService

urban_data_router = APIRouter(prefix="/urban-data", tags=["urban-data"])


@urban_data_router.get(
    "/qa/stream",
    response_class=EventSourceResponse,
    summary="Stream the tool-calling Q&A answer over urban data (external Urban MCP)",
)
async def stream_urban_data_qa(
    request: Request,
    user_request: Annotated[UrbanDataQaRequestDTO, Depends(UrbanDataQaRequestDTO)],
    token: str = Depends(verify_bearer_token),
    urban_data_mcp_client: UrbanDataMcpClient = Depends(get_urban_data_mcp_client),
    urban_data_qa_service: UrbanDataQaService = Depends(get_urban_data_qa_service),
) -> AsyncIterable[UrbanDataResponse]:

    async for chunk in stream_with_error_handling(
        urban_data_qa_service.run_urban_data_qa_pipeline,
        request,
        urban_data_qa_service,
        user_request.model,
        rerun=False,
        urban_data_mcp_client=urban_data_mcp_client,
        token=token,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        temperature=user_request.temperature,
    ):
        yield UrbanDataResponse(**chunk)
