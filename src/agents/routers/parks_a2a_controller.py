import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import app_deps, resolve_a2a_caller
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.routers.parks_controller import get_parks_service

parks_a2a_router = APIRouter(prefix="/parks", tags=["parks", "a2a"])


@parks_a2a_router.get("/.well-known/agent-card.json")
@parks_a2a_router.get("/agent.json", include_in_schema=False)
async def parks_agent_card(request: Request):
    return app_deps["parks_a2a_service"].get_agent_card(str(request.base_url))


@parks_a2a_router.post("/a2a")
async def parks_a2a(
    payload: A2AJsonRpcPayloadDTO,
    token: str = Depends(verify_bearer_token),
    _service=Depends(get_parks_service),
):
    data = (
        [part.model_dump(mode="json", exclude_none=True) for part in payload]
        if isinstance(payload, list)
        else payload.model_dump(mode="json", exclude_none=True)
    )
    caller = await resolve_a2a_caller(data, token)
    service = app_deps["parks_a2a_service"]
    if service.is_streaming_request(data):

        async def frames():
            async for event in service.stream_json_rpc(data, caller.pipeline_token):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

        return StreamingResponse(
            frames(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await service.handle_json_rpc(data, caller.pipeline_token)
