"""ChatStorage request payloads stay aligned with the current API contract."""

from unittest.mock import AsyncMock

import pytest

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)


@pytest.mark.asyncio
async def test_create_chat_sends_agent_metadata():
    handler = AsyncMock()
    handler.post.return_value = {"chat_id": "chat-1", "title": "Городские данные"}
    client = ChatStorageApiClient(handler)

    await client.create_chat(
        "token",
        "Городские данные",
        scenario_id=772,
        agent_id="scenario_data",
    )

    handler.post.assert_awaited_once_with(
        endpoint="/api/v1/chat_history/create_chat",
        auth_token="token",
        data={
            "title": "Городские данные",
            "scenario_id": 772,
            "metadata": {"agent_id": "scenario_data"},
        },
    )


@pytest.mark.asyncio
async def test_add_message_uses_metadata_field():
    handler = AsyncMock()
    handler.post.return_value = {"chat_id": "chat-1", "message_id": "message-1"}
    client = ChatStorageApiClient(handler)

    await client.add_single_message(
        "token",
        "chat-1",
        "assistant",
        "Ответ",
        model="gpt-oss-20b",
    )

    handler.post.assert_awaited_once_with(
        endpoint="/api/v1/chat_history/chat-1/message",
        auth_token="token",
        data={
            "role": "assistant",
            "content": "Ответ",
            "metadata": {"model": "gpt-oss-20b"},
        },
    )
