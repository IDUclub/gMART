import os

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src.agents.common.config.app_config import AgentsAppConfig
from src.common.auth.auth_config import AuthConfig

ENV_EXTENSIONS = [
    "agents",
    "agents.dev",
    "agents.develop",
    "agents.development",
    "agents.prod",
    "agents.production",
    "agents.example",
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


def load_config() -> AgentsAppConfig:

    for extension in ENV_EXTENSIONS:
        if try_load(extension):
            return AgentsAppConfig(
                ollama_api_url=os.getenv("OLLAMA_API_URL"),
                idu_mcp_url=os.getenv("IDU_MCP_SERVER"),
                effects_mcp_url=os.getenv("OBJECTS_EFFECTS_MCP_SERVER"),
                chat_storage_url=os.getenv("CHAT_STORAGE"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                auth_config=_build_auth_config(),
            )

    logger.warning("No config file found from: {}".format(", ".join(ENV_EXTENSIONS)))
    try:
        return AgentsAppConfig(
            ollama_api_url=os.getenv("OLLAMA_API_URL"),
            idu_mcp_url=os.getenv("IDU_MCP_SERVER"),
            effects_mcp_url=os.getenv("OBJECTS_EFFECTS_MCP_SERVER"),
            chat_storage_url=os.getenv("CHAT_STORAGE"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            auth_config=_build_auth_config(),
        )
    except ValueError:
        raise
    except Exception as e:
        logger.exception(e)
        raise ValueError("No configuration found in environment variables") from e
