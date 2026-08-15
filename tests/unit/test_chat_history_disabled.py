"""DISABLE_CHAT_HISTORY takes ChatStorage out of the restrictions pipeline.

Benchmark runs send every query on its own and never read the history back, so
the writes are pure overhead — and they make a ChatStorage (or MongoDB) outage
fail a run that does not need the service. The flag is off by default, so normal
deployments keep persisting.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.agents.services.base_llm_service import chat_history_disabled


@pytest.fixture
def service(monkeypatch, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **k: Mock(),
    )
    from src.agents.services.restriction_parser_service import RestrictionParserService

    svc = RestrictionParserService("http://ollama", Mock(), Mock(), state_store)
    svc._schedule_add_message_parts_to_chat = Mock()

    async def fake_pipeline(**kwargs):
        svc.calls.append(kwargs)
        yield {"type": "chunk", "content": {"text": "ответ", "done": True}}

    svc.calls = []
    svc._run_restriction_execution_pipline = fake_pipeline
    return svc


async def _run(svc):
    async for _ in svc.run_restriction_execution_pipline(
        mcp_client=Mock(),
        temperature=0.0,
        model="gemma-3-27b",
        user_query="запрос",
        scenario_id=772,
    ):
        pass


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("no", False),
    ],
)
def test_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("DISABLE_CHAT_HISTORY", value)
    assert chat_history_disabled() is expected


def test_flag_is_off_without_the_env_var(monkeypatch):
    monkeypatch.delenv("DISABLE_CHAT_HISTORY", raising=False)
    assert chat_history_disabled() is False


async def test_history_is_persisted_by_default(monkeypatch, service):
    monkeypatch.delenv("DISABLE_CHAT_HISTORY", raising=False)

    await _run(service)

    assert service.calls[0]["persist_history"] is True
    service._schedule_add_message_parts_to_chat.assert_called_once()


async def test_flag_stops_every_chat_storage_write(monkeypatch, service):
    monkeypatch.setenv("DISABLE_CHAT_HISTORY", "1")

    await _run(service)

    assert service.calls[0]["persist_history"] is False
    service._schedule_add_message_parts_to_chat.assert_not_called()
