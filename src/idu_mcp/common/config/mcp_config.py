import os


class IduFastMcpConfig:
    """
    IDU Fast MCP Server configuration class
    Attributes:
        URBAN_API_URL (str): Urban API URl for urban api instruments
        APP_WORKERS (int): Number of workers to use in idu_mcp http app instance
    """

    URBAN_API_URL: str
    APP_WORKERS: int = 1

    def __init__(self, urban_api_url: str, workers: str | int = 1) -> None:
        """
        Initialization function for fast idu_mcp server configuration class.
        Args:
            urban_api_url (str): Urban API URL for urban api instruments.
            workers (int | str): Number of workers to use in idu_mcp http app instance
        """

        if not urban_api_url:
            raise ValueError("URBAN_API_URL must be set")
        if isinstance(workers, str):
            if not workers.isdigit():
                raise ValueError(
                    "Number of workers must be a positive integer grater then 0"
                )
            workers = int(workers)
            if workers < 1:
                raise ValueError(
                    "Number of workers must be a positive integer grater then 0"
                )
        self.URBAN_API_URL = urban_api_url
        self.APP_WORKERS = workers
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.WORKSPACE_ENABLED = os.getenv(
            "IDU_MCP_WORKSPACE_ENABLED", "false"
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.WORKSPACE_DIR = os.getenv(
            "IDU_MCP_WORKSPACE_DIR", "/dev/shm/idu-workspace"
        )
        self.WORKSPACE_TTL_SECONDS = int(
            os.getenv("IDU_MCP_WORKSPACE_TTL_SECONDS", "3600")
        )
        self.WORKSPACE_MAX_DATASET_BYTES = int(
            os.getenv("IDU_MCP_WORKSPACE_MAX_DATASET_BYTES", str(128 * 1024 * 1024))
        )
        self.WORKSPACE_MAX_TOTAL_BYTES = int(
            os.getenv("IDU_MCP_WORKSPACE_MAX_TOTAL_BYTES", str(2 * 1024 * 1024 * 1024))
        )
