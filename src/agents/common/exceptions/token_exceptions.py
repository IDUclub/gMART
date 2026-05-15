class TokenExpiredError(Exception):
    """Raised when a downstream service returns HTTP 401 / token expired."""


class PipelineSuspendedError(Exception):
    """Raised when the pipeline is suspended due to token refresh timeout."""
