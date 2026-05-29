from src.common.auth.auth_config import AuthConfig


class IduFastMcpConfig:
    """
    IDU Fast MCP Server configuration.
    Attributes:
        URBAN_API_URL (str): Urban API URL.
        APP_WORKERS (int): Number of uvicorn workers.
        AUTH_CONFIG (AuthConfig): Keycloak auth settings.
    """

    URBAN_API_URL: str
    APP_WORKERS: int = 1
    AUTH_CONFIG: AuthConfig

    def __init__(
        self,
        urban_api_url: str,
        auth_config: AuthConfig,
        workers: str | int = 1,
    ) -> None:

        if not urban_api_url:
            raise ValueError("URBAN_API_URL must be set")
        if isinstance(workers, str):
            if not workers.isdigit():
                raise ValueError(
                    "Number of workers must be a positive integer greater than 0"
                )
            workers = int(workers)
            if workers < 1:
                raise ValueError(
                    "Number of workers must be a positive integer greater than 0"
                )
        self.URBAN_API_URL = urban_api_url
        self.APP_WORKERS = workers
        self.AUTH_CONFIG = auth_config
