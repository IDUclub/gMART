from __future__ import annotations

import json

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.auth.synapse_auth import SynapseCallerVerifier
from src.agents.dependencies.dependencies import (
    get_app_config,
    get_synapse_caller_verifier,
    get_synapse_gateway_service,
)
from src.agents.dto.synapse_request_dto import SynapseRunRequestDTO
from src.agents.schema.synapse_response import (
    SynapseRunResponse,
    SynapseRunStateResponse,
)
from src.agents.services.synapse_gateway_service import (
    SynapseGatewayConflict,
    SynapseGatewayService,
    SynapseRunNotFound,
)
from src.agents.services.synapse_run_store import SynapseIdempotencyConflict

synapse_router = APIRouter(prefix="/synapse", tags=["synapse"])


@synapse_router.get("/available")
async def synapse_available() -> dict[str, bool]:
    return {"enabled": get_app_config().SYNAPSE_ENABLED}


async def get_synapse_user_id(
    token: str = Depends(verify_bearer_token),
    verifier: SynapseCallerVerifier = Depends(get_synapse_caller_verifier),
) -> str:
    return await verifier.verify_user(token)


def _response(state: dict) -> SynapseRunResponse:
    request_id = str(state["request_id"])
    return SynapseRunResponse(
        request_id=request_id,
        chat_id=state.get("chat_id"),
        synapse_project_id=state.get("synapse_project_id"),
        run_id=state.get("run_id"),
        status=state["status"],
        events_url=f"/synapse/runs/{request_id}/events",
    )


@synapse_router.post(
    "/runs", response_model=SynapseRunResponse, status_code=status.HTTP_202_ACCEPTED
)
async def start_synapse_run(
    payload: SynapseRunRequestDTO = Body(...),
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1),
    user_id: str = Depends(get_synapse_user_id),
    service: SynapseGatewayService = Depends(get_synapse_gateway_service),
) -> SynapseRunResponse:
    try:
        state = await service.start_run(
            user_id=user_id,
            idempotency_key=idempotency_key,
            payload=payload,
        )
    except (SynapseIdempotencyConflict, SynapseGatewayConflict) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="Synapse run could not be started"
        ) from exc
    return _response(state)


@synapse_router.get("/runs/{request_id}", response_model=SynapseRunStateResponse)
async def get_synapse_run(
    request_id: str,
    user_id: str = Depends(get_synapse_user_id),
    service: SynapseGatewayService = Depends(get_synapse_gateway_service),
) -> SynapseRunStateResponse:
    try:
        state = await service.get_state_for_user(request_id, user_id)
    except SynapseRunNotFound as exc:
        raise HTTPException(status_code=404, detail="Synapse run not found") from exc
    base = _response(state).model_dump()
    return SynapseRunStateResponse(
        **base,
        last_event_id=state.get("last_event_id"),
        last_stream_id=state.get("last_stream_id"),
        error=state.get("error"),
        started_at=state.get("started_at"),
        finished_at=state.get("finished_at"),
    )


@synapse_router.get("/runs/{request_id}/events")
async def stream_synapse_run_events(
    request_id: str,
    after: str = Query("0-0"),
    user_id: str = Depends(get_synapse_user_id),
    service: SynapseGatewayService = Depends(get_synapse_gateway_service),
):
    try:
        await service.get_state_for_user(request_id, user_id)
    except SynapseRunNotFound as exc:
        raise HTTPException(status_code=404, detail="Synapse run not found") from exc

    async def frames():
        async for stream_id, event in service.store.read_events(
            request_id, after=after
        ):
            if not stream_id:
                yield ": heartbeat\n\n"
                continue
            event["stream_id"] = stream_id
            yield (
                f"id: {stream_id}\n"
                "event: synapse_event\n"
                f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            )

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@synapse_router.post(
    "/runs/{request_id}/cancel", response_model=SynapseRunStateResponse
)
async def cancel_synapse_run(
    request_id: str,
    user_id: str = Depends(get_synapse_user_id),
    service: SynapseGatewayService = Depends(get_synapse_gateway_service),
) -> SynapseRunStateResponse:
    try:
        await service.stop_run(request_id, user_id)
        return await get_synapse_run(request_id, user_id, service)
    except SynapseRunNotFound as exc:
        raise HTTPException(status_code=404, detail="Synapse run not found") from exc
