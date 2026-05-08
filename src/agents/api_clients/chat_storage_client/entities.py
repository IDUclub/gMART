"""
Module is aimed for enum entities for ChatStorage.
"""

from enum import StrEnum


class BaseChatStorageStringEnum(StrEnum):
    @classmethod
    def parse(cls, value: str):
        """
        Function parses value to RoleEnum (self) class.
        Args:
            value (str): Value to parse as string.
        Returns:
            RoleEnum: Role enum instance with provided value.
        """

        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unknown layer type: {value!r}") from exc


class RoleEnum(BaseChatStorageStringEnum):
    """
    String enum class for role.
    Attributes:
        USER (str) = "user": User role.
        SYSTEM (str) = "system": System role.
        ASSISTANT (str) = "assistant": Assistant role.
    """

    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class MessageUploadType(BaseChatStorageStringEnum):
    """
    String enum class for uploaded message type.
    Attributes:
        TEXT (str) = "text": Text message, is single uploaded message.
        PARTS (str) = "parts": Parts of messages, is list of messages with different types, e.g. "text", "tool call" and so on.
    """

    TEXT = "text"
    PARTS = "parts"
