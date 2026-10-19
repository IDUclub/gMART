from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.sse import EventSourceResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import get_project_files_service
from src.agents.dto.norms_request_dto import NormsQaRequestDTO
from src.agents.dto.project_request_dto import ProjectFromFilesRequestDTO, ProjectInputs
from src.agents.schema.project_response import ProjectResponse
from src.agents.services.project_files.project_files_service import (
    ProjectFilesService,
)

project_router = APIRouter(prefix="/projects", tags=["projects"])


@project_router.post("/from-files/stream", response_class=EventSourceResponse)
async def stream_project_from_files(
    request: Request,
    user_request: ProjectFromFilesRequestDTO,
    token: str = Depends(verify_bearer_token),
    service: ProjectFilesService = Depends(get_project_files_service),
):
    """Create a project and its scenario in Urban API from uploaded files.

    Upload the layers with ``POST /compliance/uploads`` and pass ``upload_ids``.
    The agent maps the objects to Urban API types, finds the region and the profile
    and asks only what it cannot settle (``clarification`` events): answer with the
    next message of the same ``chat_id``.
    """
    async for chunk in stream_with_error_handling(
        service.run_project_pipeline,
        request,
        service,
        user_request.model,
        rerun=False,
        token=token,
        temperature=user_request.temperature,
        user_query=user_request.request,
        chat_id=user_request.chat_id,
        request_id=user_request.request_id,
        upload_ids=user_request.upload_ids,
        inputs=user_request.inputs,
    ):
        yield ProjectResponse(**chunk)


@project_router.get("/from-files/stream", response_class=EventSourceResponse)
async def stream_project_from_files_query(
    request: Request,
    user_request: Annotated[NormsQaRequestDTO, Depends(NormsQaRequestDTO)],
    upload_ids: list[str] | None = Query(default=None),
    project_name: str | None = Query(default=None, max_length=200),
    scenario_name: str | None = Query(default=None, max_length=200),
    territory_id: int | None = Query(default=None, gt=0),
    functional_zone_type_id: int | None = Query(default=None, gt=0),
    existing_objects: Literal["replace_same", "delete_all", "keep"] | None = Query(
        default=None
    ),
    token: str = Depends(verify_bearer_token),
    service: ProjectFilesService = Depends(get_project_files_service),
):
    """The same stream for EventSource clients: parameters in the query string."""
    body = ProjectFromFilesRequestDTO(
        **user_request.model_dump(),
        upload_ids=upload_ids,
        inputs=ProjectInputs(
            project_name=project_name,
            scenario_name=scenario_name,
            territory_id=territory_id,
            functional_zone_type_id=functional_zone_type_id,
            existing_objects=existing_objects,
        ),
    )
    async for event in stream_project_from_files(request, body, token, service):
        yield event
