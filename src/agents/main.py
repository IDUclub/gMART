from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from loguru import logger
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from src.agents.__version__ import APP_DESCRIPTION, APP_TITLE, APP_VERSION
from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware
from src.agents.dependencies.dependencies import app_deps
from src.agents.routers.simple_llm_controller import llm_router


@asynccontextmanager
async def lifespan(app):
    logger.info(f"loaded dependencies {app_deps}")
    yield


app = FastAPI(
    version=APP_VERSION, title=APP_TITLE, description=APP_DESCRIPTION, lifespan=lifespan
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
    return RedirectResponse("/docs")


@app.get("/ping")
async def ping_server():
    return {"status": "ok"}


app.include_router(llm_router)
