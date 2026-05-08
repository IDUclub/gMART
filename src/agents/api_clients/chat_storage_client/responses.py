"""
Module for converting data from ChatStorage service to python objects.
"""

from dataclasses import dataclass

from src.agents.api_clients.chat_storage_client.entities import (
    MessageUploadType,
)


@dataclass(frozen=True)
class ChatCreated:
    """
    Dataclass for new created chat.
    Attributes:
        chat_id (str): String repr of created chat uuid.
        title (str): Chat title.
    """

    chat_id: str
    title: str


@dataclass(frozen=True)
class MessageAdded:
    """
    Dataclass for new added message to chat,
    Attributes:
        chat_id (str): String representation of chat uuid.
        message_id (str): String representation of message uuid.
        message_type (MessageUploadType): MessageUploadType enum value. Available values: "TEXT", "PARTS".
    """

    chat_id: str
    message_id: str
    message_type: MessageUploadType


@dataclass(frozen=True)
class ChatHistory:
    """
    Dataclass for parsing chat history response from ChatStorage API.
    Attributes:
        chat_id (str): String representation of chat uuid.
        title (str): Chat title.
        created_at (str): Chat creation date-time.
        updated_at (str): Chat last update date-time.
        messages (list[dict]): Chat messages history.
        scenario_id (int | None): Scenario ID from Urban API to which chat. Default to None.
        metadata (dict | None): Chat metadata.
    """

    chat_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[dict]
    scenario_id: str | None = None
    metadata: dict | None = None
