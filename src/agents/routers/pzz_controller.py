from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import get_pzz_mcp_client, get_pzz_service
from src.agents.dto.norms_request_dto import NormsQaRequestDTO
from src.agents.dto.pzz_request_dto import PzzRequestDTO
from src.agents.mcp_clients.pzz_mcp_client import PzzMcpClient
from src.agents.schema.pzz_response import PzzResponse
from src.agents.services.pzz.pzz_service import PzzService

pzz_router = APIRouter(prefix="/pzz", tags=["pzz"])


@pzz_router.post("/check/stream", response_class=EventSourceResponse)
async def stream_pzz_check(
    request: Request,
    user_request: PzzRequestDTO,
    token: str = Depends(verify_bearer_token),
    pzz_mcp_client: PzzMcpClient = Depends(get_pzz_mcp_client),
    pzz_service: PzzService = Depends(get_pzz_service),
):
    async for chunk in stream_with_error_handling(
        pzz_service.run_pzz_pipeline,
        request,
        pzz_service,
        user_request.model,
        rerun=False,
        pzz_mcp_client=pzz_mcp_client,
        pzz_api_client=pzz_mcp_client.api_client,
        token=token,
        user_query=user_request.request,
        temperature=user_request.temperature,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        inputs=user_request.inputs,
    ):
        yield PzzResponse(**chunk)


@pzz_router.get("/check/stream", response_class=EventSourceResponse)
async def stream_pzz_query(
    request: Request,
    user_request: Annotated[NormsQaRequestDTO, Depends(NormsQaRequestDTO)],
    token: str = Depends(verify_bearer_token),
    pzz_mcp_client: PzzMcpClient = Depends(get_pzz_mcp_client),
    pzz_service: PzzService = Depends(get_pzz_service),
):
    async for event in stream_pzz_check(
        request,
        PzzRequestDTO(**user_request.model_dump()),
        token,
        pzz_mcp_client,
        pzz_service,
    ):
        yield event


@pzz_router.post("/uploads", status_code=201)
async def upload_pzz_file(
    file: UploadFile = File(...),
    kind: Literal["layer", "zone_descriptions", "classifier"] = "layer",
    pzz_mcp_client: PzzMcpClient = Depends(get_pzz_mcp_client),
):
    from fastapi import HTTPException

    if pzz_mcp_client.api_client is None:
        raise HTTPException(503, "PZZ_API_URL is not configured")
    content = await file.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "PZZ upload exceeds 50 MiB")
    return await pzz_mcp_client.api_client.upload(
        file.filename or "layer.geojson",
        content,
        file.content_type or "application/octet-stream",
        kind=kind,
    )
