from typing import Any


class AgentsBaseException(Exception):
    """
    Agents APP base exception. Carries enough information for the
    ExceptionHandlerMiddleware to build the HTTP response — no FastAPI
    dependency here.
    Attributes:
        status_code (int): HTTP status code. Default to 500.
        message (str): Human-readable error message.
        error_input (Any): The value / payload that caused the error.
    """

    status_code = 500

    def __init__(self, message: str, error_input: Any = None):
        """
        Args:
            message (str): Error message.
            error_input (Any): Error input.
        """

        super().__init__(message)
        self.message = message
        self.error_input = error_input

    def __str__(self) -> str:
        return f"Exception: {self.message} \n {self.error_input}"


class AgentsInputException(AgentsBaseException):
    """
    Raised when input is invalid (400 Bad Request).
    Attributes:
        status_code (int): HTTP status code. Default to 400.
        message (str): Error message.
        error_input (Any): Error input. Default to None.
    """

    status_code = 400

    def __init__(self, message: str, error_input: Any = None):
        """
        Args:
            message (str): Error message.
            error_input (Any): Error input. Default to None.
        """

        super().__init__(message, error_input)


class AgentsUnauthorizedException(AgentsBaseException):
    """
    Raised when a request is unauthorized (401 Unauthorized).
    Attributes:
        status_code (int): HTTP status code. Default to 401.
        message (str): Error message.
        error_input (Any): Error input. Default to None.
    """

    status_code = 401

    def __init__(self, message: str, error_input: Any = None):
        """
        Args:
            message (str): Error message.
            error_input (Any): Error input. Default to None.
        """

        super().__init__(message, error_input)


class AgentsNotFound(AgentsBaseException):
    """
    Raised when an entity is not found in the database or a connected
    service (404 Not Found).
    Attributes:
        status_code (int): HTTP status code. Default to 404.
        message (str): Error message.
        error_input (Any): Error input. Default to None.
    """

    status_code = 404

    def __init__(self, message: str, error_input: Any = None):
        """
        Args:
            message (str): Error message.
            error_input (Any): Error input. Default to None.
        """

        super().__init__(message, error_input)
