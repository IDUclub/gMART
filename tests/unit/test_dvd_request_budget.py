from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd.context_reducer import (
    DvdContextReducer,
    current_context_window,
)
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner, _request_json
from src.agents.services.service_entities.dvd_plan import CriticVerdict
from tests.helpers import plan_json, verdict_json


async def test_legacy_character_limit_no_longer_rejects_model_input(monkeypatch):
    monkeypatch.setenv("DVD_REQUEST_MAX_CHARS", "4096")
    monkeypatch.delenv("DVD_CONTEXT_WINDOW_TOKENS", raising=False)
    llm = AsyncMock()
    llm.chat.return_value = {"message": {"content": verdict_json()}}
    await _request_json(
        llm, "m", [{"role": "user", "content": "Я" * 5000}], CriticVerdict
    )
    call = llm.chat.call_args.kwargs
    # Byte estimate of the input (10000 + framing), half of it plus the floor.
    assert call["options"]["num_predict"] == 4096 + (10000 + 64 + 256) // 2
    assert call["options"]["num_ctx"] == 32000


async def test_oversized_planning_is_rejected_before_model_call():
    llm = AsyncMock()
    with pytest.raises(ValueError, match="context window"):
        await _request_json(
            llm, "m", [{"role": "user", "content": "я" * 32000}], CriticVerdict
        )
    llm.chat.assert_not_called()


@pytest.mark.parametrize("server_window", [None, 16384, 65536])
@pytest.mark.parametrize("configured", [None, "8192", "131072"])
async def test_window_cap_respects_smaller_server_and_configuration(
    monkeypatch, server_window, configured
):
    monkeypatch.delenv("DVD_CONTEXT_WINDOW_TOKENS", raising=False)
    if configured:
        monkeypatch.setenv("DVD_CONTEXT_WINDOW_TOKENS", configured)
    llm = AsyncMock()
    llm.model_context_window.return_value = server_window
    reducer = DvdContextReducer(llm)
    target = int(configured or 100000)
    # Unverified by the server, a window never exceeds 32000.
    default = min(target, 32000)
    assert current_context_window() == default
    async with reducer.model_window("m"):
        expected = min(target, server_window) if server_window else default
        assert current_context_window() == reducer.window == expected
    assert current_context_window() == default


@pytest.mark.parametrize("method", ["_summarize", "_select_evidence"])
async def test_evidence_calls_enforce_context_window(method):
    llm = AsyncMock()
    reducer = DvdContextReducer(llm, window_tokens=131072)
    with pytest.raises(ValueError, match="context_budget_exhausted"):
        await getattr(reducer, method)("m", "q", "[1] Source\n" + "x" * 32000, 40000)
    llm.chat.assert_not_called()


async def test_answer_checks_history_and_instructions_before_sending():
    from src.agents.services.dvd.answer_generation import (
        AnswerGenerationError,
        DvdAnswerGenerator,
    )

    llm = AsyncMock()
    generator = DvdAnswerGenerator(DvdContextReducer(llm, window_tokens=131072))
    with pytest.raises(AnswerGenerationError, match="context_room"):
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


async def test_planner_output_follows_input_and_ignores_legacy_caps(monkeypatch):
    monkeypatch.setenv("DVD_PLANNER_MAX_TOKENS", "1024")
    monkeypatch.setenv("DVD_REQUEST_MAX_CHARS", "32000")
    monkeypatch.delenv("DVD_CONTEXT_WINDOW_TOKENS", raising=False)
    llm = AsyncMock()
    llm.model_input_tokens.return_value = 8000
    llm.chat.return_value = {"message": {"content": plan_json()}}
    await RetrievalPlanner(llm).build_plan(
        "gpt-oss-20b",
        "Что написано в СП 55 пункт 3?",
        history=[{"role": "user", "content": "История " * 5000}],
    )
    call = llm.chat.call_args.kwargs
    assert call["options"]["num_predict"] == 4096 + 8000 // 2
    assert call["options"]["num_ctx"] == 32000
    assert sum(len(m["content"]) for m in call["messages"]) > 32000


async def test_reasoning_output_is_bounded_by_remaining_window(monkeypatch):
    monkeypatch.setenv("DVD_CONTEXT_WINDOW_TOKENS", "8192")
    llm = AsyncMock()
    llm.chat.return_value = {"message": {"content": verdict_json()}}
    await _request_json(
        llm,
        "m",
        [{"role": "user", "content": "x" * 6000}],
        CriticVerdict,
    )
    assert (
        128 <= llm.chat.call_args.kwargs["options"]["num_predict"] <= 8192 - 6000 - 256
    )


async def test_runaway_structured_reply_is_cut_at_the_proportional_limit():
    llm = AsyncMock()
    llm.model_input_tokens.return_value = 2000
    llm.chat.side_effect = [
        {"message": {"content": "{"}, "done_reason": "length"},
        {"message": {"content": verdict_json()}},
    ]
    verdict = await _request_json(
        llm, "m", [{"role": "user", "content": "q"}], CriticVerdict
    )
    first, second = (
        c.kwargs["options"]["num_predict"] for c in llm.chat.call_args_list
    )
    # The first request stops a runaway reply early; a legitimately long reply
    # is retried with a doubled limit, still well inside the window.
    assert first == 4096 + 1000
    assert second == 2 * first < 32000 - 2000 - 256
    assert verdict.satisfied


async def test_reply_truncated_by_the_window_itself_is_not_retried(monkeypatch):
    monkeypatch.setenv("DVD_CONTEXT_WINDOW_TOKENS", "8192")
    llm = AsyncMock()
    llm.model_input_tokens.return_value = 6000
    llm.chat.return_value = {"message": {"content": "{"}, "done_reason": "length"}
    with pytest.raises(ValueError, match="exhausted_context_window"):
        await _request_json(llm, "m", [{"role": "user", "content": "q"}], CriticVerdict)
    assert llm.chat.await_count == 1


async def test_adapter_truncation_error_is_retried_with_a_wider_limit():
    from src.agents.model_clients.llm_base import LlmResponseError

    llm = AsyncMock()
    llm.model_input_tokens.return_value = 2000
    llm.chat.side_effect = [
        LlmResponseError(
            "Incomplete structured answer", 502, reason="output_truncated"
        ),
        {"message": {"content": verdict_json()}},
    ]
    verdict = await _request_json(
        llm, "m", [{"role": "user", "content": "q"}], CriticVerdict
    )
    first, second = (
        c.kwargs["options"]["num_predict"] for c in llm.chat.call_args_list
    )
    assert verdict.satisfied and second == 2 * first
