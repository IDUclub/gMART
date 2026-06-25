from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_dvd_mcp_client,
    get_dvd_rag_service,
)
from src.agents.dto.dvd_request_dto import DocumentQaRequestDTO
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.schema.dvd_response import DvdResponse
from src.agents.services.dvd_rag_service import DvdRagService

dvd_router = APIRouter(prefix="/documents", tags=["documents"])


@dvd_router.get(
    "/qa/stream",
    response_class=EventSourceResponse,
    summary="Stream the iterative RAG answer over regulatory documents (IDU_DVD)",
)
async def stream_document_qa(
    request: Request,
    user_request: Annotated[DocumentQaRequestDTO, Depends(DocumentQaRequestDTO)],
    token: str = Depends(verify_bearer_token),
    dvd_mcp_client: DvdMcpClient = Depends(get_dvd_mcp_client),
    dvd_rag_service: DvdRagService = Depends(get_dvd_rag_service),
) -> AsyncIterable[DvdResponse]:

    async for chunk in stream_with_error_handling(
        dvd_rag_service.run_document_qa_pipeline,
        request,
        dvd_rag_service,
        user_request.model,
        rerun=False,
        dvd_mcp_client=dvd_mcp_client,
        token=token,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        temperature=user_request.temperature,
    ):
        yield DvdResponse(**chunk)
