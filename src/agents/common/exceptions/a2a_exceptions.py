from typing import Any


class A2AJsonRpcError(Exception):
    """
    Base A2A JSON-RPC error.
    Attributes:
        code (int): JSON-RPC error code.
        message (str): Error message.
        data (Any | None): Additional error data.
    """

    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        """
        A2AJsonRpcError initialization function.
        Args:
            code (int): JSON-RPC error code.
            message (str): Error message.
            data (Any | None): Additional error data.
        """

        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class A2AInvalidRequestError(A2AJsonRpcError):
    """
    Raised when A2A JSON-RPC request is invalid.
    """

    def __init__(self, message: str = "Invalid Request") -> None:
        """
        A2AInvalidRequestError initialization function.
        Args:
            message (str): Error message.
        """

        super().__init__(-32600, message)


class A2AInvalidParamsError(A2AJsonRpcError):
    """
    Raised when A2A JSON-RPC params are invalid.
    """

    def __init__(self, message: str = "Invalid params") -> None:
        """
        A2AInvalidParamsError initialization function.
        Args:
            message (str): Error message.
        """

        super().__init__(-32602, message)


class A2AMethodNotFoundError(A2AJsonRpcError):
    """
    Raised when A2A method is not supported.
    """

    def __init__(self, method: str | None) -> None:
        """
        A2AMethodNotFoundError initialization function.
        Args:
            method (str | None): Requested A2A method name.
        """

        super().__init__(-32601, f"Method not found: {method}")


class A2AStreamingEndpointRequiredError(A2AJsonRpcError):
    """
    Raised when streaming A2A method is sent to non-streaming handler.
    """

    def __init__(self) -> None:
        """
        A2AStreamingEndpointRequiredError initialization function.
        """

        super().__init__(-32001, "Use the streaming endpoint for SendStreamingMessage")


class A2ATaskNotFoundError(A2AJsonRpcError):
    """
    Raised when requested A2A task is not found.
    """

    def __init__(self, task_id: str) -> None:
        """
        A2ATaskNotFoundError initialization function.
        Args:
            task_id (str): A2A task id.
        """

        super().__init__(-32004, f"Task not found: {task_id}")
