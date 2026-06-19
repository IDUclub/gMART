from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from src.agents.dependencies.dependencies import get_system_service
from src.agents.schema.app_config_request import AppConfigRequest
from src.agents.schema.app_config_response import AppConfigResponse
from src.agents.services.system_service import SystemService

system_router = APIRouter(prefix="/system", tags=["system"])


@system_router.get("/logs")
async def get_system_logs(
    system_service: SystemService = Depends(get_system_service),
):
    """
    Get FastAPI APP custom logs from last startup.
    """

    return FileResponse(
        path=system_service.log_path,
        filename=f"idu-agents-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
    )


@system_router.post("/config")
async def get_app_config(
    request: AppConfigRequest,
    system_service: SystemService = Depends(get_system_service),
) -> AppConfigResponse:
    """
    Get the current agents service runtime configuration.
    Requires the system password in the request body.
    """

    return system_service.get_app_config(request.password)
