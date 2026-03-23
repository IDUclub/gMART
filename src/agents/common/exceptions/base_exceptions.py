from typing import Any

from fastapi import HTTPException


class AgentsBaseException(Exception):
    """
    Agents APP base exception.
    Attributes:
        status_code (int): http status code. Default to 500.
        message (str): error message.
        error_input (Any): error input.
    """

    status_code = 500

    def __init__(self, message: str, error_input: Any = None):
        """
        AgentsNotFound initialization function.
        Args:
            message (str): error message
            error_input (Any): error input
        """

        super().__init__(message)
        self.message = message
        self.error_input = error_input

    def __str__(self) -> str:
        return f"Exception: {self.message} \n {self.error_input}"

    def http_repr(self, headers: dict | None = None) -> HTTPException:
        """
        Function creates http representation of error.
        Args:
            headers (dict): headers for response.
        Returns:
            HTTPException: fast api serializable http exception.
        """

        return HTTPException(
            status_code=self.status_code,
            detail={
                "message": self.message,
                "input": self.error_input,
            },
            headers=headers,
        )


class AgentsInputException(AgentsBaseException):
    """
    Raised when input is invalid.
    Attributes:
        status_code (int): http status code. Default to 400.
        message (str): error message
        error_input (Any): error input. Default to None.
    """

    status_code = 400

    def __init__(self, message: str, error_input: Any = None):
        """
        AgentsInputException initialization function.
        Args:
            message (str): error message
            error_input (Any): error input. Default to None.
        """

        super().__init__(message, error_input)

    def http_repr(self, headers: dict | None = None) -> HTTPException:

        return HTTPException(
            status_code=self.status_code,
        )


class AgentsNotFound(AgentsBaseException):
    """
    Raised when entity is not found in the database or connected service
    Attributes:
        status_code (int): http status code. Default to 404.
        message (str): error message
        error_input (Any): error input. Default to None.
    """

    status_code = 404

    def __init__(self, message: str, error_input: Any = None):
        """
        AgentsNotFound initialization function.
        Args:
            message (str): error message
            error_input (Any): error input. Default to None.
        """

        super().__init__(message, error_input)

    def http_repr(self, headers: dict | None = None) -> HTTPException:
        """
        Function creates http representation of error.
        Args:
            headers (dict): headers for response.
        Returns:
            HTTPException: fast api serializable http exception.
        """

        return HTTPException(
            status_code=self.status_code,
            detail={
                "message": self.message,
                "input": self.error_input,
            },
            headers=headers,
        )
