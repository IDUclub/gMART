from fastapi import APIRouter, Depends

from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.exceptions.base_exceptions import AgentsNotFound
from src.agents.dependencies.dependencies import get_pipeline_state_store
from src.agents.services.pipeline_state import PipelineStateStore

token_refresh_router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@token_refresh_router.post(
    "/{request_id}/token",
    summary="Provide a refreshed token to resume a suspended pipeline",
    response_description="Confirmation that the token was delivered to the waiting pipeline",
)
async def update_token(
    request_id: str,
    token: str = Depends(verify_bearer_token),
    store: PipelineStateStore = Depends(get_pipeline_state_store),
) -> dict:
    """
    Deliver a fresh bearer token to a pipeline that is suspended waiting for token refresh.

    Applies to **any** pipeline type (restriction, provision, etc.).
    Call this endpoint after receiving a ``token_expired`` SSE event from any pipeline stream.
    The pipeline will resume from the step that failed with 401.

    If no pipeline is currently waiting for ``request_id``, a 404 is returned.
    """

    subscribers = await store.provide_token(request_id, token)
    if subscribers == 0:
        raise AgentsNotFound(
            f"No pipeline is currently waiting for request_id={request_id!r}",
            error_input={"request_id": request_id},
        )
    return {"status": "ok", "request_id": request_id}
