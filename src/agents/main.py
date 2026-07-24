from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles

from src.agents.__version__ import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from src.agents.common.logging.log_config import config_logger
from src.agents.common.middlewares.exception_handler import (
    ExceptionHandlerMiddleware,
)
from src.agents.dependencies.dependencies import app_deps
from src.agents.routers.a2a_controller import a2a_router, restriction_a2a_router
from src.agents.routers.auth_controller import auth_router
from src.agents.routers.dvd_a2a_controller import dvd_a2a_router
from src.agents.routers.dvd_controller import dvd_router
from src.agents.routers.norms_a2a_controller import norms_a2a_router
from src.agents.routers.norms_controller import norms_router
from src.agents.routers.orchestrator_controller import orchestrator_router
from src.agents.routers.provision_a2a_controller import provision_a2a_router
from src.agents.routers.provision_controller import provision_router
from src.agents.routers.restriction_parser_controller import restriction_router
from src.agents.routers.simple_llm_controller import llm_router
from src.agents.routers.system_controller import system_router
from src.agents.routers.token_refresh_controller import token_refresh_router
from src.agents.routers.urban_data_a2a_controller import urban_data_a2a_router
from src.agents.routers.urban_data_controller import urban_data_router

config_logger()

UI_DIST_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"loaded dependencies {app_deps}")
    yield


app = FastAPI(
    version=APP_VERSION,
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=100)


@app.get("/", include_in_schema=False)
async def read_root():
    return (
        RedirectResponse("/ui/") if UI_DIST_DIR.exists() else RedirectResponse("/docs")
    )


@app.get("/ping")
async def ping_server():
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(llm_router)
app.include_router(restriction_router)
app.include_router(provision_router)
app.include_router(dvd_router)
app.include_router(norms_router)
app.include_router(urban_data_router)
app.include_router(orchestrator_router)
app.include_router(token_refresh_router)
app.include_router(restriction_a2a_router)
app.include_router(provision_a2a_router)
app.include_router(dvd_a2a_router)
app.include_router(norms_a2a_router)
app.include_router(urban_data_a2a_router)
app.include_router(a2a_router)
app.include_router(system_router)

if UI_DIST_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIST_DIR, html=True), name="ui")
else:
    logger.info("gMART UI is not mounted: build it with `npm run build` in frontend")
