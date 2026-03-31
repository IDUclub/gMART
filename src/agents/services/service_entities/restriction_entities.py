from dataclasses import dataclass

@dataclass(frozen=True)
class GeometryToolCallResult:
    """
    Dataclass for collecting tool result.
    Attributes:
        tool_result (dict): Result data from tool call.
        tool_calls (list[dict]): List of called tools params.
        messages (list[dict]): List of provided messages to ollama model via ollama client.
    """

    tool_result: dict
    tool_calls: list[dict]
    messages: list[dict]
