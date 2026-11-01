import json
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import app_deps
from src.agents.dto.norms_request_dto import NormsQaRequestDTO
from src.agents.dto.parks_request_dto import ParksInputs, ParksRequestDTO

parks_router = APIRouter(prefix="/parks", tags=["parks"])


def get_parks_service():
    service = app_deps["parks_service"]
    if service.api is None:
        raise HTTPException(503, "PARKS_API_URL is not configured")
    return service


@parks_router.get("/generate/stream", response_class=StreamingResponse)
async def stream_parks_query(
    request: Request,
    user_request: Annotated[NormsQaRequestDTO, Depends(NormsQaRequestDTO)],
    token: str = Depends(verify_bearer_token),
    service=Depends(get_parks_service),
    inputs: str | None = Query(
        default=None, description="JSON ParksInputs, GeoJSON in WGS84"
    ),
    conversation_id: str | None = Query(default=None, max_length=100),
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    try:
        parsed = ParksInputs.model_validate_json(inputs) if inputs else ParksInputs()
    except ValidationError as exc:
        raise HTTPException(422, "Invalid parks inputs") from exc
    return await _stream(
        ParksRequestDTO(**user_request.model_dump(), inputs=parsed),
        token,
        service,
        conversation_id,
        after,
        last_event_id,
    )


@parks_router.post("/generate/stream", response_class=StreamingResponse)
async def stream_parks_body(
    user_request: ParksRequestDTO,
    token: str = Depends(verify_bearer_token),
    service=Depends(get_parks_service),
    conversation_id: str | None = Query(default=None, max_length=100),
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    return await _stream(
        user_request, token, service, conversation_id, after, last_event_id
    )


async def _stream(payload, token, service, conversation_id, after, last_event_id):
    try:
        after = int(last_event_id) if last_event_id is not None else after
        if after < 0:
            raise ValueError()
    except ValueError as exc:
        raise HTTPException(422, "Last-Event-ID must be a nonnegative integer") from exc
    if payload.request_id:
        try:
            await service.get_run(payload.request_id, token)
        except LookupError as exc:
            raise HTTPException(404, "Park request not found") from exc

    async def frames():
        async for event in service.run_parks_pipeline(
            token=token,
            model=payload.model,
            temperature=payload.temperature,
            user_query=payload.request,
            chat_id=payload.chat_id,
            scenario_id=payload.scenario_id,
            request_id=payload.request_id,
            inputs=payload.inputs,
            conversation_key=conversation_id,
            after=after,
        ):
            event_id = event["content"].get("event_id")
            prefix = f"id: {event_id}\n" if event_id is not None else ""
            yield prefix + "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@parks_router.get("/requests/{request_id}")
async def parks_request_state(
    request_id: str,
    token: str = Depends(verify_bearer_token),
    service=Depends(get_parks_service),
):
    try:
        state = await service.get_run(request_id, token)
    except LookupError as exc:
        raise HTTPException(404, "Park request not found") from exc
    return {
        key: state.get(key)
        for key in (
            "request_id",
            "chat_id",
            "status",
            "phase",
            "task_id",
            "event_seq",
            "conversation",
        )
    }
