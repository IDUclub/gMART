from src.agents.model_clients.endpoint_policy import (
    reject_forbidden_llm_host,
    require_local_ollama_url,
)


class AgentsAppConfig:
    """
    Fast API rest agents service configuration class.
    Attributes:
        OLLAMA_URL (str): Ollama URL (also the OpenAI backend's fallback base URL).
        LLM_BACKEND (str): Which LLM backend the agents use: "openai" (default) for
            any OpenAI-compatible server such as vLLM, or "ollama" for the native
            Ollama client.
        OPENAI_BASE_URL (str | None): Base URL of that server when LLM_BACKEND=openai.
        IDU_MCP_URL (str): IDU MCP URL.
        EFFECTS_MCP_URL (str): Object Effects MCP URL.
        DVD_MCP_URL (str | None): IDU_DVD document vector-DB MCP URL (optional).
        DVD_API_URL (str | None): IDU_DVD REST URL used for user-document uploads.
        NORM_GRAPH_MCP_URL (str | None): NormGraph normative-restrictions graph MCP URL (optional).
        URBAN_MCP_URL (str | None): Base URL of the external Urban MCP server.
        CHAT_STORAGE_URL (str): Chat Storage service URL.
        URBAN_API_URL (str): Urban API URL.
        REDIS_URL (str): Redis URL (used for pipeline state and pub/sub).
        SYSTEM_PASSWORD (str | None): Optional password guarding system config retrieval.
        AUTH_HELPER_URL (str | None): IDU auth helper base URL (optional; enables /auth/token).
        AUTH_HELPER_API_KEY (str | None): API key for the auth helper /api/token endpoint.
            Secret — kept server-side only, never exposed via /system/config or logs.
        SYNAPSE_ENABLED (bool): Enable the optional Synapse gateway.
    """

    OLLAMA_URL: str
    LLM_BACKEND: str
    OPENAI_BASE_URL: str | None
    IDU_MCP_URL: str
    EFFECTS_MCP_URL: str
    DVD_MCP_URL: str | None
    DVD_API_URL: str | None
    NORM_GRAPH_MCP_URL: str | None
    URBAN_MCP_URL: str | None
    CHAT_STORAGE_URL: str
    URBAN_API_URL: str
    REDIS_URL: str
    SYSTEM_PASSWORD: str | None
    AUTH_HELPER_URL: str | None
    AUTH_HELPER_API_KEY: str | None
    SYNAPSE_ENABLED: bool
    SYNAPSE_API_URL: str | None
    SYNAPSE_SERVICE_EMAIL: str | None
    SYNAPSE_SERVICE_PASSWORD: str | None
    SYNAPSE_WORKFLOW_ID: str | None
    SYNAPSE_RUN_CONFIG_ID: str | None
    SYNAPSE_APPROVAL_MODE: str
    SYNAPSE_HTTP_TIMEOUT: float
    SYNAPSE_SSE_RECONNECT_MAX_SECONDS: float
    SYNAPSE_RUN_TTL_SECONDS: int
    SYNAPSE_A2A_CLIENT_ID: str
    SYNAPSE_AUTH_AUDIENCE: str | None

    def __init__(
        self,
        ollama_api_url: str,
        idu_mcp_url: str,
        effects_mcp_url: str,
        chat_storage_url: str,
        urban_api_url: str,
        dvd_mcp_url: str | None = None,
        dvd_api_url: str | None = None,
        norm_graph_mcp_url: str | None = None,
        urban_mcp_url: str | None = None,
        redis_url: str = "redis://localhost:6379",
        system_password: str | None = None,
        auth_helper_url: str | None = None,
        auth_helper_api_key: str | None = None,
        llm_backend: str | None = None,
        openai_base_url: str | None = None,
        scenario_data_linear_workflow_enabled: bool = False,
        scenario_data_workspace_enabled: bool = False,
        synapse_enabled: bool = False,
        synapse_api_url: str | None = None,
        synapse_service_email: str | None = None,
        synapse_service_password: str | None = None,
        synapse_workflow_id: str | None = None,
        synapse_run_config_id: str | None = None,
        synapse_approval_mode: str = "auto",
        synapse_http_timeout: float = 30.0,
        synapse_sse_reconnect_max_seconds: float = 30.0,
        synapse_run_ttl_seconds: int = 86400,
        synapse_a2a_client_id: str = "synapse",
        synapse_auth_audience: str | None = None,
    ) -> None:

        if not ollama_api_url:
            raise ValueError("OLLAMA_API_URL must be set")
        require_local_ollama_url(ollama_api_url, "OLLAMA_API_URL")
        self.OLLAMA_URL = ollama_api_url
        if not idu_mcp_url:
            raise ValueError("IDU_MCP_URL must be set")
        self.IDU_MCP_URL = idu_mcp_url
        if not effects_mcp_url:
            raise ValueError("OBJECTS_EFFECTS_MCP_SERVER must be set")
        self.EFFECTS_MCP_URL = effects_mcp_url
        if not self.EFFECTS_MCP_URL:
            raise ValueError("OBJECTS_EFFECTS_MCP_SERVER must be set")
        # Optional: only required by the document-QA (RAG) agent. Kept optional so
        # existing deployments without DVD_MCP_SERVER still start; the DVD endpoints
        # raise a clear error if it is unset (see dependencies.get_dvd_mcp_client).
        self.DVD_MCP_URL = dvd_mcp_url or None
        clean_dvd_mcp_url = dvd_mcp_url.rstrip("/") if dvd_mcp_url else ""
        derived_dvd_api_url = (
            clean_dvd_mcp_url[: -len("/mcp")]
            if clean_dvd_mcp_url.endswith("/mcp")
            else clean_dvd_mcp_url
        )
        self.DVD_API_URL = (
            dvd_api_url.rstrip("/") if dvd_api_url else derived_dvd_api_url or None
        )
        # Optional: only required by the norms-QA (NormGraph graph-RAG) agent. Kept optional
        # so existing deployments without NORM_GRAPH_MCP_SERVER still start; the /norms
        # endpoints raise a clear error if it is unset (see dependencies.get_normgraph_mcp_client).
        self.NORM_GRAPH_MCP_URL = norm_graph_mcp_url or None
        # Optional so existing deployments keep starting. The scenario-data agent
        # is hidden from the orchestrator until this URL is configured.
        self.URBAN_MCP_URL = urban_mcp_url.rstrip("/") if urban_mcp_url else None
        if not chat_storage_url:
            raise ValueError("CHAT_STORAGE_URL must be set")
        self.CHAT_STORAGE_URL = chat_storage_url
        if not urban_api_url:
            raise ValueError("URBAN_API_URL must be set")
        self.URBAN_API_URL = urban_api_url
        self.REDIS_URL = redis_url
        self.SYSTEM_PASSWORD = system_password
        # Optional: only required by the /auth/token proxy (UI login through the IDU
        # auth helper). Both must be set to enable it; the API key stays server-side.
        self.AUTH_HELPER_URL = auth_helper_url or None
        self.AUTH_HELPER_API_KEY = auth_helper_api_key or None
        # Backend selection lives here for visibility in /system/config; the client
        # factory reads the same variables from the environment.
        self.LLM_BACKEND = (llm_backend or "openai").strip().lower()
        if self.LLM_BACKEND not in ("ollama", "openai"):
            raise ValueError("LLM_BACKEND must be 'ollama' or 'openai'")
        self.OPENAI_BASE_URL = openai_base_url or None
        reject_forbidden_llm_host(self.OPENAI_BASE_URL, "OPENAI_BASE_URL")
        self.SCENARIO_DATA_LINEAR_WORKFLOW_ENABLED = (
            scenario_data_linear_workflow_enabled
        )
        self.SCENARIO_DATA_WORKSPACE_ENABLED = scenario_data_workspace_enabled
        self.SYNAPSE_ENABLED = synapse_enabled
        self.SYNAPSE_API_URL = synapse_api_url.rstrip("/") if synapse_api_url else None
        self.SYNAPSE_SERVICE_EMAIL = synapse_service_email or None
        self.SYNAPSE_SERVICE_PASSWORD = synapse_service_password or None
        self.SYNAPSE_WORKFLOW_ID = synapse_workflow_id or None
        self.SYNAPSE_RUN_CONFIG_ID = synapse_run_config_id or None
        self.SYNAPSE_APPROVAL_MODE = synapse_approval_mode
        self.SYNAPSE_HTTP_TIMEOUT = synapse_http_timeout
        self.SYNAPSE_SSE_RECONNECT_MAX_SECONDS = synapse_sse_reconnect_max_seconds
        self.SYNAPSE_RUN_TTL_SECONDS = synapse_run_ttl_seconds
        self.SYNAPSE_A2A_CLIENT_ID = synapse_a2a_client_id
        self.SYNAPSE_AUTH_AUDIENCE = synapse_auth_audience or None
        if self.SYNAPSE_ENABLED:
            required_synapse = {
                "SYNAPSE_API_URL": self.SYNAPSE_API_URL,
                "SYNAPSE_SERVICE_EMAIL": self.SYNAPSE_SERVICE_EMAIL,
                "SYNAPSE_SERVICE_PASSWORD": self.SYNAPSE_SERVICE_PASSWORD,
                "SYNAPSE_WORKFLOW_ID": self.SYNAPSE_WORKFLOW_ID,
                "SYNAPSE_RUN_CONFIG_ID": self.SYNAPSE_RUN_CONFIG_ID,
            }
            missing_synapse = [
                name for name, value in required_synapse.items() if not value
            ]
            if missing_synapse:
                raise ValueError(
                    "Missing mandatory Synapse variables: " + ", ".join(missing_synapse)
                )
        if self.LLM_BACKEND == "openai" and not (
            self.OPENAI_BASE_URL or self.OLLAMA_URL
        ):
            raise ValueError("LLM_BACKEND=openai requires OPENAI_BASE_URL")

    def to_dict(self) -> dict[str, str]:

        return {
            "OLLAMA_URL": self.OLLAMA_URL,
            "LLM_BACKEND": self.LLM_BACKEND,
            # OPENAI_API_KEY is deliberately absent here, like AUTH_HELPER_API_KEY.
            "OPENAI_BASE_URL": self.OPENAI_BASE_URL or "",
            "IDU_MCP_URL": self.IDU_MCP_URL,
            "EFFECTS_MCP_URL": self.EFFECTS_MCP_URL,
            "DVD_MCP_URL": self.DVD_MCP_URL or "",
            "DVD_API_URL": self.DVD_API_URL or "",
            "NORM_GRAPH_MCP_URL": self.NORM_GRAPH_MCP_URL or "",
            "URBAN_MCP_URL": self.URBAN_MCP_URL or "",
            "CHAT_STORAGE_URL": self.CHAT_STORAGE_URL,
            "URBAN_API_URL": self.URBAN_API_URL,
            "REDIS_URL": self.REDIS_URL,
            # AUTH_HELPER_API_KEY is deliberately omitted: to_dict feeds
            # /system/config and __repr__, and the key must not leak there.
            "AUTH_HELPER_URL": self.AUTH_HELPER_URL or "",
            "SCENARIO_DATA_LINEAR_WORKFLOW_ENABLED": str(
                self.SCENARIO_DATA_LINEAR_WORKFLOW_ENABLED
            ),
            "SCENARIO_DATA_WORKSPACE_ENABLED": str(
                self.SCENARIO_DATA_WORKSPACE_ENABLED
            ),
            "SYNAPSE_ENABLED": str(self.SYNAPSE_ENABLED),
            "SYNAPSE_API_URL": self.SYNAPSE_API_URL or "",
            "SYNAPSE_WORKFLOW_ID": self.SYNAPSE_WORKFLOW_ID or "",
            "SYNAPSE_RUN_CONFIG_ID": self.SYNAPSE_RUN_CONFIG_ID or "",
            "SYNAPSE_APPROVAL_MODE": self.SYNAPSE_APPROVAL_MODE,
            "SYNAPSE_A2A_CLIENT_ID": self.SYNAPSE_A2A_CLIENT_ID,
            "SYNAPSE_AUTH_AUDIENCE": self.SYNAPSE_AUTH_AUDIENCE or "",
        }

    def __repr__(self) -> str:

        return str(self.to_dict())
