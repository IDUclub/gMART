import asyncio
import json
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.sse import EventSourceResponse

from src.agents.api_clients.dvd_api_client import DvdApiClient
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.exceptions.base_exceptions import AgentsInputException
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_dvd_api_client,
    get_dvd_mcp_client,
    get_dvd_rag_service,
    get_urban_api_client,
)
from src.agents.dto.dvd_request_dto import DocumentQaRequestDTO
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.schema.dvd_response import DvdResponse
from src.agents.services.dvd_rag_service import DvdRagService

dvd_router = APIRouter(prefix="/documents", tags=["documents"])

_TERMINAL_DOCUMENT_JOB_STATUSES = {"done", "error"}


async def _stream_document_job_snapshots(
    request: Request,
    dvd_api_client: DvdApiClient,
    job_id: str,
    initial: dict,
):
    """Emit changed IDU_DVD job snapshots and reconnect-safe heartbeats."""
    snapshot = initial
    previous = ""
    heartbeat_at = asyncio.get_running_loop().time()
    while True:
        encoded = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        if encoded != previous:
            yield f"data: {encoded}\n\n"
            previous = encoded
            heartbeat_at = asyncio.get_running_loop().time()
        if snapshot.get("status") in _TERMINAL_DOCUMENT_JOB_STATUSES:
            return
        if await request.is_disconnected():
            return
        await asyncio.sleep(0.75)
        if asyncio.get_running_loop().time() - heartbeat_at >= 15:
            yield ": keep-alive\n\n"
            heartbeat_at = asyncio.get_running_loop().time()
        snapshot = await dvd_api_client.get_user_document_job(job_id)


@dvd_router.post("/user-documents", status_code=202)
async def upload_user_document(
    file: UploadFile = File(...),
    project_id: str | None = Form(None),
    scenario_id: str | None = Form(None),
    name: str | None = Form(None),
    version: str | None = Form(None),
    token: str = Depends(verify_bearer_token),
    dvd_api_client: DvdApiClient = Depends(get_dvd_api_client),
    urban_api_client: UrbanApiClient = Depends(get_urban_api_client),
):
    """Upload a document to the current user's IDU_DVD project index."""

    if not project_id and not scenario_id:
        raise AgentsInputException("project_id or scenario_id is required")
    if not project_id:
        try:
            project_id = str(
                await urban_api_client.get_project_by_scenario(token, int(scenario_id))
            )
        except (TypeError, ValueError) as exc:
            raise AgentsInputException("scenario_id must be an integer") from exc
    return await dvd_api_client.upload_user_document(
        file,
        project_id=project_id,
        scenario_id=scenario_id,
        name=name,
        version=version,
    )


@dvd_router.get("/user-documents")
async def list_user_documents(
    project_id: str | None = None,
    scenario_id: str | None = None,
    dvd_api_client: DvdApiClient = Depends(get_dvd_api_client),
):
    """List documents visible in the current user's IDU_DVD project index."""

    if not project_id and not scenario_id:
        raise AgentsInputException("project_id or scenario_id is required")
    return await dvd_api_client.list_user_documents(
        project_id=project_id, scenario_id=scenario_id
    )


@dvd_router.patch("/user-documents/{name}", status_code=202)
async def update_user_document(
    name: str,
    file: UploadFile = File(...),
    project_id: str | None = Form(None),
    scenario_id: str | None = Form(None),
    version: str | None = Form(None),
    dvd_api_client: DvdApiClient = Depends(get_dvd_api_client),
):
    """Update an existing document in the current user's IDU_DVD index."""
    if not project_id and not scenario_id:
        raise AgentsInputException("project_id or scenario_id is required")
    return await dvd_api_client.update_user_document(
        name,
        file,
        project_id=project_id,
        scenario_id=scenario_id,
        version=version,
    )


@dvd_router.delete("/user-documents/{name}")
async def delete_user_document(
    name: str,
    project_id: str | None = Query(None),
    scenario_id: str | None = Query(None),
    version: str | None = Query(None),
    dvd_api_client: DvdApiClient = Depends(get_dvd_api_client),
):
    """Delete an existing document from the current user's IDU_DVD index."""
    if not project_id and not scenario_id:
        raise AgentsInputException("project_id or scenario_id is required")
    return await dvd_api_client.delete_user_document(
        name,
        project_id=project_id,
        scenario_id=scenario_id,
        version=version,
    )


@dvd_router.get("/user-documents/jobs/{job_id}/stream")
async def stream_user_document_job(
    job_id: str,
    request: Request,
    dvd_api_client: DvdApiClient = Depends(get_dvd_api_client),
):
    """Stream the current user's document-ingestion progress as SSE snapshots."""
    initial = await dvd_api_client.get_user_document_job(job_id)
    return StreamingResponse(
        _stream_document_job_snapshots(request, dvd_api_client, job_id, initial),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
