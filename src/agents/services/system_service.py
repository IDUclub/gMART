import secrets
from pathlib import Path

from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.base_exceptions import AgentsUnauthorizedException
from src.agents.schema.app_config_response import AppConfigResponse


class SystemService:
    def __init__(self, log_path: Path, app_config: AgentsAppConfig):

        self.log_path: Path = log_path
        self.app_config: AgentsAppConfig = app_config

    def get_app_config(self, password: str | None) -> AppConfigResponse:
        """
        Build a response model with the current application configuration.
        Access is gated by the system password and fails closed: if no password
        is configured on the service, access is always denied.
        Args:
            password (str | None): System password from the request body.
        Returns:
            AppConfigResponse: current agents service configuration.
        Raises:
            AgentsUnauthorizedException: If the password is unset, missing or invalid (401).
        """

        configured = self.app_config.SYSTEM_PASSWORD
        if (
            not configured
            or not password
            or not secrets.compare_digest(password, configured)
        ):
            raise AgentsUnauthorizedException("Invalid system password")
        return AppConfigResponse(**self.app_config.to_dict())
