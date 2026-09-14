"""Independent REST/SSE and A2A routes for each planning specialist."""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.dependencies.dependencies import (
    a2a_urban_mcp_client,
    app_deps,
    get_optional_urban_mcp_client,
    resolve_a2a_caller,
)
from src.agents.dto.a2a_dto import A2AJsonRpcPayloadDTO
from src.agents.services.planning.a2a import PlanningA2AService, PlanningAgentCard
from src.agents.services.planning.profiles import PROFILES
from src.common.service_auth import user_id_from_jwt


class PlanningRequest(BaseModel):
    request: str = Field(min_length=1)
    scenario_id: int | None = Field(default=None, gt=0)
    model: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    input_artifacts: dict[str, Any] = Field(default_factory=dict)


def planning_router(key):
    router = APIRouter(prefix="/" + key, tags=[key, "a2a"])
    # Separate stores for each verified caller; task IDs are never bearer capabilities.
    callers = {}

    def service():
        result = app_deps["planning_services"][key]
        if not result.mcp_url:
            raise HTTPException(503, f"{key.upper()}_MCP_SERVER is not configured")
        return result

    @router.get("/.well-known/agent-card.json", include_in_schema=False)
    @router.get("/agent.json", include_in_schema=False)
    async def agent_card(request: Request):
        return PlanningAgentCard(PROFILES[key]).get_agent_card(str(request.base_url))

    @router.post("/run/stream")
    async def run(
        body: PlanningRequest,
        token: str = Depends(verify_bearer_token),
        urban=Depends(get_optional_urban_mcp_client),
    ):
        specialist = service()

        async def events():
            async for event in specialist.run(
                token=token,
                user_query=body.request,
                scenario_id=body.scenario_id,
                model=body.model,
                temperature=body.temperature,
                input_artifacts=body.input_artifacts,
                urban_mcp_client=urban,
            ):
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @router.post("/a2a")
    async def a2a(
        payload: A2AJsonRpcPayloadDTO,
        token: str = Depends(verify_bearer_token),
        urban=Depends(get_optional_urban_mcp_client),
    ):
        data = (
            [v.model_dump(mode="json", exclude_none=True) for v in payload]
            if isinstance(payload, list)
            else payload.model_dump(mode="json", exclude_none=True)
        )
        caller = await resolve_a2a_caller(data, token)
        if caller.is_synapse:
            urban = await a2a_urban_mcp_client(caller.user_id)
        uid = caller.user_id or user_id_from_jwt(token)
        if uid not in callers:
            callers[uid] = PlanningA2AService(service())
        api = callers[uid]
        if api.is_streaming_request(data):

            async def events():
                async for event in api.stream_json_rpc(
                    data, urban, caller.pipeline_token
                ):
                    yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")
        return await api.handle_json_rpc(data, urban, caller.pipeline_token)

    return router


planning_routers = [planning_router(key) for key in PROFILES]
