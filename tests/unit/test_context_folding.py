"""An oversized restrictions context is folded, not dropped.

The summary table grows with the scenario: on the largest ones the final
answer's prompt reached 380 000 tokens against a 16 384 window, the server
answered 400 and the whole pipeline row was lost. Oversized contexts are now
summarised part by part, and the streamed answer is written from those
summaries.
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.model_clients.model_limits import context_budget_chars, context_tokens
from src.agents.services.restriction_context import RestrictionContextBuilder


def _objects(n: int) -> dict:
    """n distinct restriction names — the part that has no bound."""

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30.3 + i * 1e-4, 59.9]},
                "properties": {
                    "restriction_name": f"Ограничение по объекту номер {i}",
                    "restriction_description": "Описание нормативного ограничения " * 3,
                    "source_layer": "Жилой дом",
                    "object_ref": {"id": f"feature/{i}", "name": f"дом {i}"},
                },
            }
            for i in range(n)
        ],
    }


def _generators() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [30.3, 59.9],
                            [30.4, 59.9],
                            [30.4, 60.0],
                            [30.3, 60.0],
                            [30.3, 59.9],
                        ]
                    ],
                },
                "properties": {"source_layer": "Источники ограничений"},
            }
        ],
    }


def test_context_window_comes_from_configuration(monkeypatch):
    monkeypatch.setenv("MODEL_CONTEXT_TOKENS", "gpt-oss-20b=16384,gemma-3-27b=32768")

    assert context_tokens("gemma-3-27b") == 32768
    assert context_tokens("gpt-oss-20b") == 16384
    # an unlisted model falls back rather than assuming the largest window
    assert context_tokens("mystery-model") == 16384
    assert context_budget_chars("gemma-3-27b") > context_budget_chars("gpt-oss-20b")


def test_budget_leaves_room_for_the_answer(monkeypatch):
    monkeypatch.setenv("MODEL_CONTEXT_TOKENS", "m=10000")
    monkeypatch.setenv("MODEL_CONTEXT_RESERVE_TOKENS", "3000")
    monkeypatch.setenv("MODEL_CHARS_PER_TOKEN", "3")

    assert context_budget_chars("m") == (10000 - 3000) * 3


@pytest.mark.asyncio
async def test_a_context_that_fits_is_one_unsplit_part():
    builder = RestrictionContextBuilder()

    chunks = await builder.generate_restrictions_context_chunks(
        _generators(), _objects(3), budget_chars=1_000_000
    )
    whole = await builder.generate_restrictions_context(_generators(), _objects(3))

    assert chunks == [whole]


@pytest.mark.asyncio
async def test_an_oversized_context_is_split_into_parts_within_budget():
    builder = RestrictionContextBuilder()
    budget = 4000

    chunks = await builder.generate_restrictions_context_chunks(
        _generators(), _objects(400), budget_chars=budget
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= budget for chunk in chunks)
    # every part stands on its own: it says which part it is and carries the
    # generators block, so it can be summarised without the others
    for number, chunk in enumerate(chunks, start=1):
        assert f"часть {number} из {len(chunks)}" in chunk
        assert "Генераторы ограничений" in chunk


@pytest.mark.asyncio
async def test_every_restriction_survives_the_split():
    builder = RestrictionContextBuilder()

    chunks = await builder.generate_restrictions_context_chunks(
        _generators(), _objects(120), budget_chars=4000
    )

    seen = sum(chunk.count("Ограничение по объекту номер") for chunk in chunks)
    assert seen == 120


@pytest.mark.asyncio
async def test_empty_objects_do_not_split():
    builder = RestrictionContextBuilder()
    empty = {"type": "FeatureCollection", "features": []}

    chunks = await builder.generate_restrictions_context_chunks(
        _generators(), empty, budget_chars=10
    )

    assert len(chunks) == 1


def test_overflow_is_recognised_from_the_server_message():
    from src.agents.services.restriction_parser_service import RestrictionParserService

    overflow = LlmResponseError(
        "Error code: 400 - Input length (381519) exceeds model's maximum "
        "context length (16384).",
        400,
    )

    assert RestrictionParserService._is_context_overflow(overflow)
    assert not RestrictionParserService._is_context_overflow(
        LlmResponseError("Error code: 500 - internal", 500)
    )


def test_folded_context_says_it_is_a_fold():
    from src.agents.services.restriction_parser_service import RestrictionParserService

    folded = RestrictionParserService._folded_context(["первая", "вторая"])

    assert "по частям (2)" in folded
    assert "первая" in folded and "вторая" in folded
    assert "GeoJSON" in folded


@pytest.mark.asyncio
async def test_oversized_context_is_summarised_then_streamed(monkeypatch, state_store):
    """The user sees one answer, written from per-part summaries."""

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter", lambda *a, **k: Mock()
    )
    from src.agents.services.restriction_parser_service import RestrictionParserService

    service = RestrictionParserService("http://ollama", Mock(), Mock(), state_store)
    summarised: list[str] = []
    finals: list[str] = []

    async def fake_summary(model, user_query, part, temperature):
        summarised.append(part)
        return f"выжимка {len(summarised)}"

    async def fake_final(model, user_query, context, temperature, history=None):
        finals.append(context)
        yield {"type": "chunk", "content": {"text": "итог", "done": True}}

    service._summarize_context_part = fake_summary
    service.generate_final_response = fake_final
    monkeypatch.setenv("MODEL_CONTEXT_TOKENS", "m=4000")
    monkeypatch.setenv("MODEL_CONTEXT_RESERVE_TOKENS", "3000")
    monkeypatch.setenv("MODEL_CHARS_PER_TOKEN", "1")

    events = [
        event
        async for event in service._final_response_events(
            "m", "запрос", _generators(), _objects(300), 0.0
        )
    ]

    assert len(summarised) > 1, "the context should have been split"
    assert len(finals) == 1, "the answer is written once, from the summaries"
    assert "выжимка 1" in finals[0]
    statuses = [e for e in events if e.get("type") == "status"]
    assert len(statuses) == len(summarised), "each part is announced while it runs"
    assert events[-1]["content"]["text"] == "итог"


@pytest.mark.asyncio
async def test_a_stale_window_setting_refolds_once(monkeypatch, state_store):
    """The configured window can lag the server; an overflow that still gets
    through is refolded rather than lost — but only before anything streamed."""

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter", lambda *a, **k: Mock()
    )
    from src.agents.services.restriction_parser_service import RestrictionParserService

    service = RestrictionParserService("http://ollama", Mock(), Mock(), state_store)
    attempts: list[int] = []

    async def fake_summary(model, user_query, part, temperature):
        return "выжимка"

    async def fake_final(model, user_query, context, temperature, history=None):
        attempts.append(len(context))
        if len(attempts) == 1:
            raise LlmResponseError(
                "Error code: 400 - Input length (99999) exceeds model's maximum "
                "context length (16384).",
                400,
            )
        yield {"type": "chunk", "content": {"text": "итог", "done": True}}

    service._summarize_context_part = fake_summary
    service.generate_final_response = fake_final
    monkeypatch.setenv("MODEL_CONTEXT_TOKENS", "m=100000")

    events = [
        event
        async for event in service._final_response_events(
            "m", "запрос", _generators(), _objects(200), 0.0
        )
    ]

    assert len(attempts) == 2
    assert events[-1]["content"]["text"] == "итог"


@pytest.mark.asyncio
async def test_an_unrelated_failure_is_not_retried(monkeypatch, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter", lambda *a, **k: Mock()
    )
    from src.agents.services.restriction_parser_service import RestrictionParserService

    service = RestrictionParserService("http://ollama", Mock(), Mock(), state_store)
    attempts: list[int] = []

    async def fake_final(model, user_query, context, temperature, history=None):
        attempts.append(1)
        raise LlmResponseError("Error code: 503 - server overloaded", 503)
        yield  # pragma: no cover — generator marker

    service.generate_final_response = fake_final

    with pytest.raises(LlmResponseError):
        async for _ in service._final_response_events(
            "m", "запрос", _generators(), _objects(2), 0.0
        ):
            pass

    assert len(attempts) == 1


def test_parts_are_valid_json_payloads():
    """The model is handed a table it can parse, not a string cut in half."""

    rows = [{"Наименование ограничения": "а", "Количество объектов": 1}]
    items = [json.dumps(row, ensure_ascii=False) for row in rows]
    part = RestrictionContextBuilder._render_part("генераторы", items, 1, 2, 10)

    payload = part[part.index("[") : part.rindex("]") + 1]
    assert json.loads(payload) == rows


@pytest.mark.asyncio
async def test_summaries_are_folded_again_when_they_do_not_fit(
    monkeypatch, state_store
):
    """Enough parts and the summaries themselves overrun the window — which is
    how the scenarios this was meant to rescue kept failing anyway."""

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter", lambda *a, **k: Mock()
    )
    from src.agents.services.restriction_parser_service import RestrictionParserService

    service = RestrictionParserService("http://ollama", Mock(), Mock(), state_store)
    rounds: list[int] = []

    async def fake_summary(model, user_query, part, temperature):
        rounds.append(len(part))
        return "выжимка" * 20

    service._summarize_context_part = fake_summary

    folded = await service._reduce_summaries(
        "m", "запрос", ["выжимка" * 20] * 40, 0.0, budget=2000
    )

    assert len(folded) <= 2000
    assert rounds, "a second fold should have run"


@pytest.mark.asyncio
async def test_reduce_stops_instead_of_folding_for_ever(monkeypatch, state_store):
    """A summary that never shrinks must end the fold, not loop."""

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter", lambda *a, **k: Mock()
    )
    from src.agents.services.restriction_parser_service import RestrictionParserService

    service = RestrictionParserService("http://ollama", Mock(), Mock(), state_store)
    calls: list[int] = []

    async def stubborn(model, user_query, part, temperature):
        calls.append(1)
        return "х" * 5000

    service._summarize_context_part = stubborn

    folded = await service._reduce_summaries(
        "m", "запрос", ["х" * 5000] * 8, 0.0, budget=1000
    )

    assert len(folded) <= 1000
    assert len(calls) <= 8 * 3, "the depth cap should have stopped it"


def test_summaries_are_grouped_within_the_budget():
    from src.agents.services.restriction_parser_service import RestrictionParserService

    groups = RestrictionParserService._group_to_budget(["а" * 100] * 10, budget=300)

    assert all(
        sum(len(s) + 32 for s in group) <= 300 or len(group) == 1 for group in groups
    )
    assert sum(len(group) for group in groups) == 10


def _one_restriction_many_reasons(n_objects: int, reasons_per_object: int) -> dict:
    """The shape that defeated the first attempt: a single restriction name, with
    the size living in each object's evidence instead of in the summary table."""

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30.3 + i * 1e-4, 59.9]},
                "properties": {
                    "restriction_name": "Санитарно-защитная зона",
                    "restriction_description": "Одно и то же описание",
                    "source_layer": "Жилой дом",
                    "object_ref": {"id": f"feature/{i}", "name": f"дом {i}"},
                    "restriction_evidence": [
                        {
                            "generator": f"источник {j}",
                            "reason": "пересечение с зоной ограничения " * 4,
                        }
                        for j in range(reasons_per_object)
                    ],
                },
            }
            for i in range(n_objects)
        ],
    }


@pytest.mark.asyncio
async def test_a_single_restriction_with_heavy_evidence_is_still_split():
    """One distinct restriction meant nothing to divide by, so the whole context
    went to the model unsplit and came back 400 — 22 rows lost that way."""

    builder = RestrictionContextBuilder()
    objects = _one_restriction_many_reasons(n_objects=10, reasons_per_object=200)
    budget = 6000

    whole = await builder.generate_restrictions_context(_generators(), objects)
    chunks = await builder.generate_restrictions_context_chunks(
        _generators(), objects, budget_chars=budget
    )

    assert len(whole) > budget, "the fixture must actually overflow"
    assert len(chunks) > 1
    assert all(len(chunk) <= budget for chunk in chunks)


@pytest.mark.asyncio
async def test_one_object_larger_than_the_whole_budget_is_truncated():
    """A single object's evidence can exceed the budget on its own; the part has
    to stay within it rather than the row being lost."""

    builder = RestrictionContextBuilder()
    objects = _one_restriction_many_reasons(n_objects=2, reasons_per_object=2000)

    chunks = await builder.generate_restrictions_context_chunks(
        _generators(), objects, budget_chars=4000
    )

    assert all(len(chunk) <= 4000 for chunk in chunks)
