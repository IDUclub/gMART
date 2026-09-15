from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd.context_reducer import (
    DvdContextReducer,
    current_context_window,
)
from src.agents.services.dvd.dvd_reasoning import _request_json
from src.agents.services.dvd.request_budget import (
    check_request,
    request_chars,
    request_limit,
)
from src.agents.services.service_entities.dvd_plan import CriticVerdict


def test_full_unicode_request_includes_history_sources_and_schema(monkeypatch):
    monkeypatch.delenv("DVD_REQUEST_MAX_CHARS", raising=False)
    messages = [
        {"role": "system", "content": "Источники: " + "Я" * 15000},
        {"role": "assistant", "content": "история"},
        {"role": "user", "content": "вопрос"},
    ]
    assert request_limit() == 32000
    assert request_chars(messages) < 16000
    check_request(messages)
    with pytest.raises(ValueError, match="character_limit"):
        check_request(messages, {"description": "С" * 18000})


async def test_oversized_planning_is_rejected_before_model_call():
    llm = AsyncMock()
    with pytest.raises(ValueError, match="character_limit"):
        await _request_json(
            llm, "m", [{"role": "user", "content": "я" * 32000}], CriticVerdict
        )
    llm.chat.assert_not_called()


async def test_fallback_window_increased_but_server_cap_is_respected(monkeypatch):
    monkeypatch.delenv("DVD_CONTEXT_WINDOW_TOKENS", raising=False)
    assert current_context_window() == 65536
    llm = AsyncMock()
    llm.model_context_window.return_value = 16384
    reducer = DvdContextReducer(llm)
    async with reducer.model_window("m"):
        assert current_context_window() == 16384
    assert current_context_window() == 65536


@pytest.mark.parametrize("method", ["_summarize", "_select_evidence"])
async def test_evidence_calls_enforce_whole_request_limit(method):
    llm = AsyncMock()
    reducer = DvdContextReducer(llm, window_tokens=131072)
    with pytest.raises(ValueError, match="character_limit"):
        await getattr(reducer, method)("m", "q", "[1] Source\n" + "x" * 32000, 40000)
    llm.chat.assert_not_called()


async def test_answer_checks_history_and_instructions_before_sending():
    from src.agents.services.dvd.answer_generation import (
        AnswerGenerationError,
        DvdAnswerGenerator,
    )

    llm = AsyncMock()
    generator = DvdAnswerGenerator(DvdContextReducer(llm, window_tokens=131072))
    with pytest.raises(AnswerGenerationError, match="character_room"):
        await generator.generate(
            "m",
            "q",
            "[1] source",
            0,
            lambda evidence: [
                {"role": "system", "content": "x" * 32000 + evidence},
                {"role": "user", "content": "q"},
            ],
            iteration=1,
        )
    llm.chat.assert_not_called()
