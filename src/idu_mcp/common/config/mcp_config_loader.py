import os

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src.common.auth.auth_config import AuthConfig
from src.idu_mcp.common.config.mcp_config import IduFastMcpConfig

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


def _build_auth_config() -> AuthConfig:
    audiences_raw = os.getenv("AUTH_VALID_AUDIENCES", "")
    valid_audiences = [a.strip() for a in audiences_raw.split(",") if a.strip()]
    return AuthConfig(
        verify=os.getenv("AUTH_VERIFY", "true").lower() == "true",
        server_url=os.getenv("AUTH_SERVER_URL", ""),
        client_id=os.getenv("AUTH_CLIENT_ID", ""),
        verify_aud=os.getenv("AUTH_VERIFY_AUD", "false").lower() == "true",
        valid_audiences=valid_audiences,
    )


def load_config() -> IduFastMcpConfig:

    for extension in ENV_EXTENSIONS:
        if try_load(extension):
            return IduFastMcpConfig(
                urban_api_url=os.getenv("URBAN_API_URL"),
                workers=os.getenv("WORKERS", "1"),
                auth_config=_build_auth_config(),
            )

    logger.warning("No new configurations found in env file or no env file found")
    try:
        return IduFastMcpConfig(
            urban_api_url=os.getenv("URBAN_API_URL"),
            workers=os.getenv("WORKERS", "1"),
            auth_config=_build_auth_config(),
        )
    except Exception as e:
        raise Exception(
            f"Failed to initialize app configuration. Error: {repr(e)}"
        ) from e
