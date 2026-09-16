from pathlib import Path

from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.schema.app_config_response import AppConfigResponse


class SystemService:
    def __init__(self, log_path: Path, app_config: AgentsAppConfig):

        self.log_path: Path = log_path
        self.app_config: AgentsAppConfig = app_config

    def get_app_config(self) -> AppConfigResponse:
        """Build the public view of the current application configuration."""
        return AppConfigResponse(**self.app_config.to_dict())
