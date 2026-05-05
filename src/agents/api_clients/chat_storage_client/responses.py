"""
Module for converting data from ChatStorage service to python objects.
"""

from dataclasses import dataclass

from src.agents.api_clients.chat_storage_client.entities import MessageUploadType


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
