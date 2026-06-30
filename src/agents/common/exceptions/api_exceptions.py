from typing import Any

from src.agents.common.exceptions.base_exceptions import AgentsBaseException


class DownstreamServiceError(AgentsBaseException):
    """
    Raised when a downstream REST service (Urban API, ChatStorage, ...) returns an
    unexpected, non-retryable status or stays unavailable after all retries.

    Surfaced to the client as 502 Bad Gateway — the agents service itself is fine,
    but a service it depends on failed.
    Attributes:
        status_code (int): HTTP status code returned to the caller. Default to 502.
        service (str): Base URL of the failing downstream service.
        downstream_status (int | None): Status code returned by the downstream
            service, or None when the failure was a network/transport error.
        message (str): Human-readable error message.
        error_input (Any): Downstream response body or error repr.
    """

    status_code = 502

    def __init__(
        self,
        service: str,
        downstream_status: int | None,
        message: str,
        error_input: Any = None,
    ):
        """
        Args:
            service (str): Base URL of the failing downstream service.
            downstream_status (int | None): Status code returned downstream, or None
                for transport-level failures.
            message (str): Error message.
            error_input (Any): Error input. Default to None.
        """

        self.service = service
        self.downstream_status = downstream_status
        super().__init__(message, error_input)
