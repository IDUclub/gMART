from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class A2APartDTO(BaseModel):
    """
    A2A message part.
    Attributes:
        type (str): Part type, usually text or data.
        text (str | None): Text part payload.
        data (dict | None): Structured data part payload.
        mediaType (str | None): Optional media type for data/file parts.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(default="text", examples=["text", "data"])
    text: str | None = Field(
        default=None,
        examples=["Построй зону ограничения вокруг школ 200 метров"],
        description="Text payload for text parts.",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Structured payload for data parts.",
    )
    mediaType: str | None = Field(
        default=None,
        examples=["application/json", "application/vnd.geo+json"],
        description="Media type for non-text payloads.",
    )


class A2AMessageDTO(BaseModel):
    """
    A2A user message.
    Attributes:
        role (str): Message role.
        parts (list[A2APartDTO]): A2A message parts.
        metadata (dict | None): Agent metadata such as scenario_id, model and temperature.
    """

    model_config = ConfigDict(extra="allow")

    role: str = Field(default="user", examples=["user"])
    parts: list[A2APartDTO] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "type": "text",
                    "text": "Построй зону ограничения вокруг школ 200 метров",
                }
            ]
        ],
        description="Message content parts.",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        examples=[{"scenario_id": 772, "model": "gpt-oss:20b", "temperature": 0.7}],
        description="Business metadata hidden from the agent prompt.",
    )


class A2AParamsDTO(BaseModel):
    """
    A2A JSON-RPC params.
    Attributes:
        message (A2AMessageDTO | None): User message for send methods.
        id (str | int | None): Task id for task methods.
        taskId (str | int | None): Alternative task id field for task methods.
        contextId (str | None): Optional A2A context id.
        metadata (dict | None): Request metadata.
        includeArtifacts (bool | None): Whether task listing should include artifacts.
    """

    model_config = ConfigDict(extra="allow")

    message: A2AMessageDTO | None = Field(default=None)
    id: str | int | None = Field(
        default=None, description="Task id for task operations."
    )
    taskId: str | int | None = Field(default=None, description="Alternative task id.")
    contextId: str | None = Field(default=None, description="A2A context id.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        examples=[{"scenario_id": 772, "model": "gpt-oss:20b", "temperature": 0.7}],
        description="Request metadata.",
    )
    includeArtifacts: bool | None = Field(
        default=True,
        description="Return task artifacts in ListTasks response.",
    )


class A2AJsonRpcRequestDTO(BaseModel):
    """
    A2A JSON-RPC request.
    Attributes:
        jsonrpc (str): JSON-RPC protocol version.
        id (str | int | None): JSON-RPC request id.
        method (str | None): A2A method name.
        params (A2AParamsDTO): A2A method params.
    """

    model_config = ConfigDict(extra="allow")

    jsonrpc: str = Field(default="2.0", examples=["2.0"])
    id: str | int | None = Field(default=None, examples=["check-1"])
    method: str | None = Field(
        default=None,
        examples=[
            "SendMessage",
            "SendStreamingMessage",
            "GetTask",
            "ListTasks",
        ],
        description="A2A JSON-RPC method.",
    )
    params: A2AParamsDTO = Field(
        default_factory=A2AParamsDTO,
        examples=[
            {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Построй зону ограничения вокруг школ 200 метров",
                        }
                    ],
                    "metadata": {
                        "scenario_id": 772,
                        "model": "gpt-oss:20b",
                        "temperature": 0.7,
                    },
                }
            }
        ],
    )


A2AJsonRpcPayloadDTO: TypeAlias = A2AJsonRpcRequestDTO | list[A2AJsonRpcRequestDTO]
