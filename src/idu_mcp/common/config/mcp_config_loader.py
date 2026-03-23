import os

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from .mcp_config import IduFastMcpConfig

ENV_EXTENSIONS = [
    "",
    "idu_mcp",
    "idu_mcp.dev",
    "idu_mcp.develop",
    "idu_mcp.development",
    "idu_mcp.prod",
    "idu_mcp.production",
    "idu_mcp.example",
]


def try_load(env_file_extension: str):

    before = dict(os.environ)
    find_res = find_dotenv(f".env.{env_file_extension}")
    load_dotenv(find_res, override=True)
    return {
        k: (before.get(k), os.environ.get(k))
        for k in os.environ
        if before.get(k) != os.environ.get(k)
    }


def load_config() -> IduFastMcpConfig:

    for extension in ENV_EXTENSIONS:
        if try_load(extension):
            return IduFastMcpConfig(
                urban_api_url=os.getenv("URBAN_API_URL"), workers=os.getenv("WORKERS")
            )

    logger.warning("No new configurations found in env file or no env file foound")
    try:
        return IduFastMcpConfig(
            urban_api_url=os.getenv("URBAN_API_URL"), workers=os.getenv("WORKERS")
        )
    except Exception as e:
        raise Exception(
            "Failed to initialize app configuration. Error: {}".format(repr(e))
        ) from e
