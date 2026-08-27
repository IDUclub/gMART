from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis

from src.agents.api_clients.synapse_client import SynapseUnavailableError
from src.agents.dto.synapse_request_dto import SynapseRunRequestDTO
from src.agents.services.synapse_gateway_service import (
    SynapseGatewayService,
    SynapseStartUnknownError,
)
from src.agents.services.synapse_run_store import SynapseRunStore


@pytest.mark.asyncio
async def test_new_run_persists_user_project_and_run_mapping() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)
    client = AsyncMock()
    client.create_project.return_value = {
        "project_id": "synapse-project",
        "status": "running",
        "title": "Result",
    }
    client.get_project.return_value = {
        "project_id": "synapse-project",
        "current_run_id": "synapse-run",
        "created_at": "2026-08-27T10:00:00Z",
    }
    chat_storage = AsyncMock()
    chat_storage.create_chat.return_value.chat_id = "chat-id"
    service = SynapseGatewayService(client, store, chat_storage, workflow_id="workflow")
    service.start_relay = lambda request_id: None
    payload = SynapseRunRequestDTO(
        request="Проверь ограничения", scenario_id=772, project_id=42
    )

    state = await service.start_run(
        user_id="user-id", idempotency_key="key-1", payload=payload
    )

    assert state["status"] == "running"
    assert state["synapse_project_id"] == "synapse-project"
    assert state["run_id"] == "synapse-run"
    assert (
        await store.resolve_a2a_user(project_id="synapse-project", run_id="synapse-run")
        == "user-id"
    )
    prompt = client.create_project.await_args.args[0]
    assert "scenario_id=772" in prompt
    assert "[USER_REQUEST]" in prompt
    assert "Bearer" not in prompt
    await redis.aclose()


def test_prompt_contains_only_allowlisted_context() -> None:
    payload = SynapseRunRequestDTO(
        request="Запрос",
        scenario_id=1,
        metadata={
            "selected_object_ids": [2],
            "selected_layer_ids": [3],
            "Authorization": "secret",
        },
    )

    prompt = SynapseGatewayService.build_prompt("request-id", payload)

    assert "selected_object_ids=[2]" in prompt
    assert "Authorization" not in prompt
    assert "secret" not in prompt


@pytest.mark.asyncio
async def test_follow_up_reuses_project_run_and_chat() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)
    client = AsyncMock()
    client.get_project.return_value = {
        "project_id": "synapse-project",
        "current_run_id": "synapse-run",
        "created_at": "2026-01-01T00:00:00Z",
    }
    chat_storage = AsyncMock()
    chat_storage.get_chat.return_value = SimpleNamespace(
        metadata={"synapse_project_id": "synapse-project"}
    )
    service = SynapseGatewayService(client, store, chat_storage, workflow_id="workflow")
    service.start_relay = lambda request_id: None
    payload = SynapseRunRequestDTO(
        request="Продолжи проверку",
        chat_id="chat-id",
        scenario_id=772,
    )

    state = await service.start_run(
        user_id="user-id", idempotency_key="follow-up-1", payload=payload
    )

    assert state["chat_id"] == "chat-id"
    assert state["run_id"] == "synapse-run"
    client.send_message.assert_awaited_once()
    assert client.send_message.await_args.kwargs == {
        "run_id": "synapse-run",
        "metadata": {"request_id": state["request_id"]},
    }
    assert state["started_at"] != "2026-01-01T00:00:00Z"
    await redis.aclose()


@pytest.mark.asyncio
async def test_unknown_project_start_is_reconciled_without_second_create() -> None:
    redis = FakeRedis(decode_responses=True)
    store = SynapseRunStore(redis)
    client = AsyncMock()
    client.create_project.side_effect = SynapseUnavailableError("timeout")
    client.find_projects.return_value = []
    client.get_project.return_value = {
        "project_id": "synapse-project",
        "current_run_id": "synapse-run",
    }
    chat_storage = AsyncMock()
    chat_storage.create_chat.return_value.chat_id = "chat-id"
    service = SynapseGatewayService(client, store, chat_storage, workflow_id="workflow")
    service.start_relay = lambda request_id: None
    payload = SynapseRunRequestDTO(request="Запрос", scenario_id=772)

    with pytest.raises(SynapseStartUnknownError, match="result is unknown"):
        await service.start_run(
            user_id="user-id", idempotency_key="stable-key", payload=payload
        )

    client.find_projects.return_value = [
        {"project_id": "synapse-project", "title": "Recovered"}
    ]
    recovered = await service.start_run(
        user_id="user-id", idempotency_key="stable-key", payload=payload
    )

    assert recovered["status"] == "running"
    assert client.create_project.await_count == 1
    await redis.aclose()
