from typing import Any, Literal

from pydantic import BaseModel


class TextPayload(BaseModel):
    """
    Pydantic model for simple text payload
    Attributes:
        text (str): Text to post.
    """

    text: str


class TextPartRequest(BaseModel):
    """
    Request pydantic model for text part in request.
    Attributes:
        kind (Literal["text"]): Type equals to "text".
        payload (textPayload): Text payload to post
    """

    kind: Literal["text"]
    payload: TextPayload


class StatusPayload(BaseModel):
    """
    Pydantic model for status part payload.
    Attributes:
        status (str): Status name.
        text (str): Additional message to status.
    """

    status: str
    text: str


class StatusPartRequest(BaseModel):
    """
    Request pydantic model for status part in request.
    Attributes:
        kind (Literal["status"]): Type equals to "status".
        payload (StatusPayload): Status payload to post.
    """

    kind: Literal["status"]
    payload: StatusPayload


class TableColumn(BaseModel):
    """
    Pydantic model for a strict table column.
    Attributes:
        key (str): Machine-readable column key, stable across requests.
        label (str): Human-readable column label (Russian).
    """

    key: str
    label: str


class TablePayload(BaseModel):
    """
    Pydantic model for table part payload.
    Attributes:
        name (str): Machine-readable table identifier (e.g. "provision_summary").
        title (str): Human-readable table title.
        columns (list[TableColumn]): Ordered column contract.
        rows (list[dict[str, Any]]): Row values keyed by column key.
    """

    name: str
    title: str
    columns: list[TableColumn]
    rows: list[dict[str, Any]]


class TablePartRequest(BaseModel):
    """
    Request pydantic model for table part in request.
    Attributes:
        kind (Literal["table"]): Type equals to "table".
        payload (TablePayload): Table payload to post.
    """

    kind: Literal["table"]
    payload: TablePayload


class ToolCall(BaseModel):
    """
    Tool Call pydantic model.
    Attributes:
        step (int): Tool call step in request.
        tool_name (str): Extracted tool name.
        arguments (dict[str, Any]): Tool call used arguments.
    """

    step: int
    tool_name: str
    arguments: dict[str, Any]


class ToolCallPayload(BaseModel):
    """
    Pydantic model for tool call payload.
    Attributes:
        execution_mode (str): Execution mode for tool.
        calls (list[ToolCall]): List of called tools.
    """

    execution_mode: str
    calls: list[ToolCall]


class ToolCallPartRequest(BaseModel):
    """
    Request pydantic model for tool call part.
    Attributes:
        kind (Literal["tool_call"]): type equals to "tool_call".
        payload (ToolCallPayload): ToolCall payload to post.
        mcp_source (str | None): MCP server identifier in {NAME}_MCP_URL format.
    """

    kind: Literal["tool_call"]
    payload: ToolCallPayload
    mcp_source: str | None = None
