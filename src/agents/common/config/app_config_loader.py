import os

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from src.agents.common.config.app_config import AgentsAppConfig

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


def load_config() -> AgentsAppConfig:

    def synapse_settings() -> dict:
        return {
            "synapse_enabled": os.getenv("SYNAPSE_ENABLED", "false").lower()
            in {"1", "true", "yes", "on"},
            "synapse_api_url": os.getenv("SYNAPSE_API_URL"),
            "synapse_service_email": os.getenv("SYNAPSE_SERVICE_EMAIL"),
            "synapse_service_password": os.getenv("SYNAPSE_SERVICE_PASSWORD"),
            "synapse_workflow_id": os.getenv("SYNAPSE_WORKFLOW_ID"),
            "synapse_run_config_id": os.getenv("SYNAPSE_RUN_CONFIG_ID"),
            "synapse_approval_mode": os.getenv("SYNAPSE_APPROVAL_MODE", "auto"),
            "synapse_http_timeout": float(os.getenv("SYNAPSE_HTTP_TIMEOUT", "30")),
            "synapse_sse_reconnect_max_seconds": float(
                os.getenv("SYNAPSE_SSE_RECONNECT_MAX_SECONDS", "30")
            ),
            "synapse_run_ttl_seconds": int(
                os.getenv("SYNAPSE_RUN_TTL_SECONDS", "86400")
            ),
            "synapse_a2a_client_id": os.getenv("SYNAPSE_A2A_CLIENT_ID", "synapse"),
            "synapse_auth_audience": os.getenv("SYNAPSE_AUTH_AUDIENCE"),
        }

    for extension in ENV_EXTENSIONS:
        if try_load(extension):
            return AgentsAppConfig(
                ollama_api_url=os.getenv("OLLAMA_API_URL"),
                idu_mcp_url=os.getenv("IDU_MCP_SERVER"),
                effects_mcp_url=os.getenv("OBJECTS_EFFECTS_MCP_SERVER"),
                chat_storage_url=os.getenv("CHAT_STORAGE"),
                urban_api_url=os.getenv("URBAN_API_URL"),
                dvd_mcp_url=os.getenv("DVD_MCP_SERVER"),
                dvd_api_url=os.getenv("DVD_API_URL"),
                norm_graph_mcp_url=os.getenv("NORM_GRAPH_MCP_SERVER"),
                urban_mcp_url=os.getenv("URBAN_MCP_SERVER"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                system_password=os.getenv("SYSTEM_PASSWORD"),
                auth_helper_url=os.getenv("AUTH_HELPER_URL"),
                auth_helper_api_key=os.getenv("AUTH_HELPER_API_KEY"),
                llm_backend=os.getenv("LLM_BACKEND"),
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
                scenario_data_linear_workflow_enabled=os.getenv(
                    "SCENARIO_DATA_LINEAR_WORKFLOW_ENABLED", "false"
                ).lower()
                in {"1", "true", "yes", "on"},
                scenario_data_workspace_enabled=os.getenv(
                    "SCENARIO_DATA_WORKSPACE_ENABLED", "false"
                ).lower()
                in {"1", "true", "yes", "on"},
                **synapse_settings(),
            )
    logger.warning("No config file found from: {}".format(", ".join(ENV_EXTENSIONS)))
    try:
        return AgentsAppConfig(
            ollama_api_url=os.getenv("OLLAMA_API_URL"),
            idu_mcp_url=os.getenv("IDU_MCP_SERVER"),
            effects_mcp_url=os.getenv("OBJECTS_EFFECTS_MCP_SERVER"),
            chat_storage_url=os.getenv("CHAT_STORAGE"),
            urban_api_url=os.getenv("URBAN_API_URL"),
            dvd_mcp_url=os.getenv("DVD_MCP_SERVER"),
            dvd_api_url=os.getenv("DVD_API_URL"),
            norm_graph_mcp_url=os.getenv("NORM_GRAPH_MCP_SERVER"),
            urban_mcp_url=os.getenv("URBAN_MCP_SERVER"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            system_password=os.getenv("SYSTEM_PASSWORD"),
            auth_helper_url=os.getenv("AUTH_HELPER_URL"),
            auth_helper_api_key=os.getenv("AUTH_HELPER_API_KEY"),
            llm_backend=os.getenv("LLM_BACKEND"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            scenario_data_linear_workflow_enabled=os.getenv(
                "SCENARIO_DATA_LINEAR_WORKFLOW_ENABLED", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            scenario_data_workspace_enabled=os.getenv(
                "SCENARIO_DATA_WORKSPACE_ENABLED", "false"
            ).lower()
            in {"1", "true", "yes", "on"},
            **synapse_settings(),
        )
    except ValueError:
        raise
    except Exception as e:
        logger.exception(e)
        raise ValueError("No configuration found in environment variables") from e
