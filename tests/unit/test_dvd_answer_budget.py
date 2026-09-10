from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd.answer_generation import (
    DvdAnswerGenerator,
    append_continuation,
    message_cost,
)
from src.agents.services.dvd.context_reducer import DvdContextReducer, PreparedContext
from src.agents.services.dvd.dvd_rag_service import DvdRagService


class InterruptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    async def chat(self, model, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        text, reason = next(self.replies)

        async def stream():
            yield SimpleNamespace(
                message=SimpleNamespace(content=text), done_reason=reason
            )

        return stream()


def service_for(replies):
    service = DvdRagService.__new__(DvdRagService)
    service.llm_client = InterruptedModel(replies)
    service.context_reducer = DvdContextReducer(service.llm_client, window_tokens=65536)
    return service


async def answer(service):
    return [
        event
        async for event in service._generate_answer(
            "gpt-oss-20b",
            "Какое расстояние?",
            "[1] Источник\nНе менее 15 м.",
            0,
            [],
            1,
        )
    ]


async def test_length_continues_visible_prefix_with_larger_budget():
    service = service_for([("Не менее ", "length"), ("Не менее 15 м [1].", "stop")])
    events = await answer(service)
    assert "".join(e["content"]["text"] for e in events) == "Не менее 15 м [1]."
    first, second = service.llm_client.calls
    assert second["options"]["num_predict"] > first["options"]["num_predict"]
    assert {"role": "assistant", "content": "Не менее "} in second["messages"]
    assert all(not e["content"]["done"] for e in events)


async def test_reasoning_only_length_restarts_without_empty_assistant_message():
    service = service_for([("", "length"), ("Не менее 15 м [1].", "stop")])
    assert (await answer(service))[0]["content"]["text"] == "Не менее 15 м [1]."
    first, second = service.llm_client.calls
    assert first["messages"] == second["messages"]
    assert second["options"]["num_predict"] > first["options"]["num_predict"]


async def test_exhausted_retries_do_not_emit_partial_answer():
    service = service_for([("Черновик ", "length")] * 10)
    events = []
    with pytest.raises(ValueError, match="answer_generation_incomplete"):
        async for event in service._generate_answer("m", "q", "source", 0, [], 1):
            events.append(event)
    assert events == []
    assert len(service.llm_client.calls) == 3


async def test_content_filter_is_not_retried_as_token_exhaustion():
    service = service_for([("blocked", "content_filter")])
    with pytest.raises(ValueError, match="answer_generation_filtered"):
        await answer(service)
    assert len(service.llm_client.calls) == 1


def test_dynamic_budget_grows_with_evidence_and_respects_explicit_cap(monkeypatch):
    generator = DvdAnswerGenerator(DvdContextReducer(None))
    assert generator.initial_budget("short", "q") < generator.initial_budget(
        "x" * 32000, "q"
    )
    assert generator.initial_budget("x" * 200000, "q") == generator.maximum
    monkeypatch.setenv("DVD_ANSWER_MAX_TOKENS", "1536")
    generator = DvdAnswerGenerator(DvdContextReducer(None))
    assert generator.initial_budget("x" * 32000, "q") == 1536


async def test_retry_reduces_evidence_to_reserve_larger_output_and_keeps_history():
    service = service_for([("Rule: ", "length"), ("Rule: 15 m [1].", "stop")])
    service.context_reducer.configured_window = 16384
    service.context_reducer.prepare = AsyncMock(
        return_value=PreparedContext("[1] 15 m")
    )
    history = {"role": "user", "content": "Earlier question"}

    def messages(context):
        return [
            {"role": "system", "content": context},
            history,
            {"role": "user", "content": "q"},
        ]

    result = await DvdAnswerGenerator(service.context_reducer).generate(
        "m",
        "q",
        "x" * 8000,
        0,
        messages,
        iteration=1,
    )
    assert result == "Rule: 15 m [1]."
    service.context_reducer.prepare.assert_awaited_once()
    first, second = service.llm_client.calls
    assert second["options"]["num_predict"] > first["options"]["num_predict"]
    assert second["messages"][0]["content"] == "[1] 15 m"
    assert history in second["messages"]
    for call in service.llm_client.calls:
        assert message_cost(call["messages"]) + call["options"]["num_predict"] <= 16384


async def test_no_room_for_history_fails_before_calling_model():
    service = service_for([])
    with pytest.raises(ValueError, match="no_context_room"):
        await DvdAnswerGenerator(service.context_reducer).generate(
            "m",
            "q",
            "source",
            0,
            lambda context: [{"role": "user", "content": "Я" * 40000}],
            iteration=1,
        )
    assert not service.llm_client.calls


async def test_failed_reduction_never_uses_incomplete_evidence():
    service = service_for([])
    service.context_reducer.prepare = AsyncMock(
        return_value=PreparedContext("partial", failed_parts=["source 2 failed"])
    )
    with pytest.raises(ValueError, match="context_incomplete"):
        await DvdAnswerGenerator(service.context_reducer).generate(
            "m",
            "q",
            "x" * 70000,
            0,
            lambda context: [{"role": "system", "content": context}],
            iteration=1,
        )
    assert not service.llm_client.calls


async def test_reasoning_only_failure_stops_at_hard_cap(monkeypatch):
    monkeypatch.setenv("DVD_ANSWER_MAX_TOKENS", "1536")
    service = service_for([("", "length")] * 10)
    with pytest.raises(ValueError, match="no_budget_growth"):
        await answer(service)
    assert len(service.llm_client.calls) == 1


async def test_eof_without_terminal_event_is_not_a_complete_answer():
    service = service_for([("Не менее 15 м [1].", None)])
    with pytest.raises(ValueError, match="missing_terminal"):
        await answer(service)


def test_exact_repeated_prefix_is_not_duplicated():
    prefix = "Согласно источнику минимальное расстояние — "
    assert append_continuation(prefix, prefix + "15 м.") == prefix + "15 м."
    assert (
        append_continuation("Ответ. " + prefix, prefix + "15 м.")
        == "Ответ. " + prefix + "15 м."
    )
    assert append_continuation("Число: 1", "15.") == "Число: 115."


async def test_complete_answer_with_no_text_is_not_accepted():
    service = service_for([("", "stop")])
    with pytest.raises(ValueError, match="empty_completion"):
        await answer(service)


async def test_continuation_repeats_last_line_without_losing_word_separator():
    service = service_for(
        [
            ("First complete rule.\nFor test object", "length"),
            ("For test object ITEM-75, distance is 85 m.", "stop"),
        ]
    )
    assert (await answer(service))[0]["content"]["text"] == (
        "First complete rule.\nFor test object ITEM-75, distance is 85 m."
    )


async def test_bad_overlap_is_retried_without_incorporating_guessed_text():
    service = service_for(
        [
            ("First rule.\nFor test object", "length"),
            ("ITEM-75, wrong boundary.", "stop"),
            ("For test object ITEM-75, distance is 85 m.", "stop"),
        ]
    )
    result = (await answer(service))[0]["content"]["text"]
    assert result == "First rule.\nFor test object ITEM-75, distance is 85 m."
    assert len(service.llm_client.calls) == 3


async def test_openai_eof_is_not_fabricated_as_success():
    from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter

    async def raw():
        yield SimpleNamespace(
            model="m",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="Partial", role="assistant"),
                    finish_reason=None,
                )
            ],
        )

    parts = [part async for part in OpenAiCompatAdapter._as_stream(raw())]
    assert parts[-1].done and parts[-1].done_reason == "incomplete"


async def test_context_reduction_error_is_a_controlled_generation_failure():
    service = service_for([])
    service.context_reducer.prepare = AsyncMock(
        side_effect=ValueError("did not converge")
    )
    with pytest.raises(ValueError, match="answer_generation_context_reduction_failed"):
        await DvdAnswerGenerator(service.context_reducer).generate(
            "m",
            "q",
            "x" * 70000,
            0,
            lambda context: [{"role": "system", "content": context}],
            iteration=1,
        )
    assert not service.llm_client.calls


async def test_repeated_full_prefix_takes_precedence_over_similar_tail():
    service = service_for([("Rule\nRule", "length"), ("Rule\nRule complete.", "stop")])
    assert (await answer(service))[0]["content"]["text"] == "Rule\nRule complete."
