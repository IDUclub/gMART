from collections.abc import AsyncIterable
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.executors.sse_executors import stream_with_error_handling
from src.agents.dependencies.dependencies import (
    get_compliance_upload_service,
    get_idu_mcp_client,
    get_optional_dvd_mcp_client,
    get_optional_normgraph_mcp_client,
    get_restriction_parser_service,
)
from src.agents.dto.compliance_request_dto import ComplianceRequestDTO, UploadRolesDTO
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.schema.file_event import file_sse_event
from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_uploads import (
    MAX_UPLOAD_BYTES,
    ComplianceUploadService,
)
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.common.service_auth import user_id_from_jwt

compliance_router = APIRouter(prefix="/compliance", tags=["compliance"])


def _owner(token: str) -> str:
    try:
        return user_id_from_jwt(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@compliance_router.post("/uploads", status_code=201)
async def upload_compliance_files(
    files: list[UploadFile] = File(
        ...,
        description=(
            "GeoJSON (.geojson/.json), ZIP of GeoJSON or Shapefiles, or GeoPackage. "
            "Each file, archive member and GeoPackage layer becomes one layer."
        ),
    ),
    role: Literal["project", "context", "border"] | None = Form(
        default=None,
        description=(
            "Role of every layer of this upload; omit it to detect roles from layer "
            "names and contents."
        ),
    ),
    token: str = Depends(verify_bearer_token),
    uploads: ComplianceUploadService = Depends(get_compliance_upload_service),
) -> dict:
    """Upload the layers of a design to check them against the norms.

    Returns ``upload_id`` (kept for 24 hours, readable only by its author) and the
    parsed layers with their roles. Pass the id in ``upload_ids`` of
    ``/compliance/check/stream`` or of the orchestrator.
    """
    owner = _owner(token)
    contents: list[tuple[str, bytes]] = []
    total = 0
    for item in files:
        content = await item.read(MAX_UPLOAD_BYTES + 1)
        total += len(content)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB",
            )
        contents.append((item.filename or "layer.geojson", content))
    return await uploads.create(owner, contents, role=role)


@compliance_router.get("/uploads/{upload_id}")
async def get_compliance_upload(
    upload_id: str,
    token: str = Depends(verify_bearer_token),
    uploads: ComplianceUploadService = Depends(get_compliance_upload_service),
) -> dict:
    """Layers of an upload with their roles and object counts."""
    return uploads.summary(await uploads.get(upload_id, _owner(token)))


@compliance_router.patch("/uploads/{upload_id}")
async def set_compliance_upload_roles(
    upload_id: str,
    body: UploadRolesDTO,
    token: str = Depends(verify_bearer_token),
    uploads: ComplianceUploadService = Depends(get_compliance_upload_service),
) -> dict:
    """Override the detected roles of an upload's layers by layer name."""
    return await uploads.set_roles(upload_id, _owner(token), body.roles)


@compliance_router.delete("/uploads/{upload_id}", status_code=204)
async def delete_compliance_upload(
    upload_id: str,
    token: str = Depends(verify_bearer_token),
    uploads: ComplianceUploadService = Depends(get_compliance_upload_service),
) -> Response:
    await uploads.delete(upload_id, _owner(token))
    return Response(status_code=204)


class CheckPlanReviewDTO(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["approve", "reject", "replace"]
    plan: dict | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def replace_requires_plan(self) -> "CheckPlanReviewDTO":
        if (self.action == "replace") != (self.plan is not None):
            raise ValueError("plan is required exactly for replace")
        return self


@compliance_router.get("/check-plans/review")
async def pending_check_plans(
    limit: int = 100,
    _token: str = Depends(verify_bearer_token),
    client: NormGraphMcpClient | None = Depends(get_optional_normgraph_mcp_client),
) -> list[dict]:
    if client is None:
        raise HTTPException(status_code=503, detail="NormGraph is not configured")
    return await client.pending_check_plans(max(1, min(limit, 500)))


@compliance_router.post("/check-plans/{restriction_id}/review")
async def review_check_plan(
    restriction_id: str,
    body: CheckPlanReviewDTO,
    _token: str = Depends(verify_bearer_token),
    client: NormGraphMcpClient | None = Depends(get_optional_normgraph_mcp_client),
) -> dict:
    if client is None:
        raise HTTPException(status_code=503, detail="NormGraph is not configured")
    result = await client.review_check_plan(
        restriction_id,
        body.action,
        plan=body.plan,
        reason=body.reason,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="CheckPlan not found")
    return result


def _query_request(
    request: str,
    model: str | None = None,
    temperature: float = 1.0,
    scenario_id: int | None = None,
    chat_id: str | None = None,
    request_id: str | None = None,
    upload_ids: list[str] | None = Query(
        default=None, description="Uploads (POST /compliance/uploads) to check"
    ),
) -> ComplianceRequestDTO:
    try:
        return ComplianceRequestDTO(
            request=request,
            model=model,
            temperature=temperature,
            scenario_id=scenario_id,
            chat_id=chat_id,
            request_id=request_id,
            upload_ids=upload_ids or [],
        )
    except ValidationError as exc:
        detail = [
            {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]
        raise HTTPException(status_code=422, detail=detail) from exc


@compliance_router.get("/check/stream", response_class=EventSourceResponse)
async def check_compliance(
    request: Request,
    user_request: Annotated[ComplianceRequestDTO, Depends(_query_request)],
    token: str = Depends(verify_bearer_token),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    normgraph_mcp_client: NormGraphMcpClient | None = Depends(
        get_optional_normgraph_mcp_client
    ),
    dvd_mcp_client: DvdMcpClient | None = Depends(get_optional_dvd_mcp_client),
    restriction_service: RestrictionParserService = Depends(
        get_restriction_parser_service
    ),
) -> AsyncIterable[RestrictionsResponse | ServerSentEvent]:
    """Check scenario layers against the norms in force on its territory.

    A question about which restrictions apply («какие ограничения есть на
    территории проекта?») returns the areas the norms govern instead of verdicts.
    With ``upload_ids`` the uploaded files are checked instead: over the scenario
    when ``scenario_id`` is given, on their own otherwise.
    """

    async for chunk in stream_with_error_handling(
        restriction_service.run_compliance_pipeline,
        request,
        restriction_service,
        user_request.model,
        rerun=False,
        mcp_client=idu_mcp_client,
        token=token,
        normgraph_mcp_client=normgraph_mcp_client,
        dvd_mcp_client=dvd_mcp_client,
        user_query=user_request.request,
        scenario_id=user_request.scenario_id,
        chat_id=user_request.chat_id,
        temperature=user_request.temperature,
        request_id=user_request.request_id,
        upload_ids=user_request.upload_ids or None,
    ):
        yield file_sse_event(chunk) or RestrictionsResponse(**chunk)


@compliance_router.post("/check/stream", response_class=EventSourceResponse)
async def check_compliance_body(
    request: Request,
    user_request: ComplianceRequestDTO,
    token: str = Depends(verify_bearer_token),
    idu_mcp_client: IduMcpClient = Depends(get_idu_mcp_client),
    normgraph_mcp_client: NormGraphMcpClient | None = Depends(
        get_optional_normgraph_mcp_client
    ),
    dvd_mcp_client: DvdMcpClient | None = Depends(get_optional_dvd_mcp_client),
    restriction_service: RestrictionParserService = Depends(
        get_restriction_parser_service
    ),
) -> AsyncIterable[RestrictionsResponse | ServerSentEvent]:
    """The same check with a JSON body — convenient for a list of upload_ids."""

    async for chunk in check_compliance(
        request,
        user_request,
        token,
        idu_mcp_client,
        normgraph_mcp_client,
        dvd_mcp_client,
        restriction_service,
    ):
        yield chunk
