from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_normgraph_mcp_client,
    get_normgraph_rag_service,
)
from src.agents.dto.norms_request_dto import NormsQaRequestDTO
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.schema.norms_response import NormsResponse
from src.agents.services.normgraph_rag_service import NormGraphRagService

norms_router = APIRouter(prefix="/norms", tags=["norms"])


@norms_router.get(
    "/qa/stream",
    response_class=EventSourceResponse,
    summary="Stream the iterative graph-RAG answer over normative restrictions (NormGraph)",
)
async def stream_norms_qa(
    request: Request,
    user_request: Annotated[NormsQaRequestDTO, Depends(NormsQaRequestDTO)],
    token: str = Depends(verify_bearer_token),
    normgraph_mcp_client: NormGraphMcpClient = Depends(get_normgraph_mcp_client),
    normgraph_rag_service: NormGraphRagService = Depends(get_normgraph_rag_service),
) -> AsyncIterable[NormsResponse]:

    async for chunk in stream_with_error_handling(
        normgraph_rag_service.run_norms_qa_pipeline,
        request,
        normgraph_rag_service,
        user_request.model,
        rerun=False,
        normgraph_mcp_client=normgraph_mcp_client,
        token=token,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        temperature=user_request.temperature,
    ):
        yield NormsResponse(**chunk)
