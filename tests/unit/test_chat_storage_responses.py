"""Parsing ChatStorage responses — unknown/new fields must not break history loading."""

from __future__ import annotations

from src.agents.api_clients.chat_storage_client.responses import ChatHistory

_BASE = {
    "chat_id": "c1",
    "title": "Тестовый чат",
    "created_at": "2026-07-14T00:00:00Z",
    "updated_at": "2026-07-14T00:00:00Z",
    "messages": [],
}


def test_parses_known_fields():
    history = ChatHistory.from_response({**_BASE, "scenario_id": "772"})
    assert history.chat_id == "c1"
    assert history.scenario_id == "772"
    assert history.project_id is None


def test_parses_project_id():
    history = ChatHistory.from_response({**_BASE, "project_id": 5})
    assert history.project_id == 5


def test_ignores_unknown_fields():
    history = ChatHistory.from_response(
        {**_BASE, "project_id": None, "brand_new_field": {"x": 1}}
    )
    assert history.title == "Тестовый чат"
    assert not hasattr(history, "brand_new_field")
