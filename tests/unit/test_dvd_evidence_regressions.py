import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.dvd.context_reducer import DvdContextReducer, SummaryError
from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.dvd.dvd_reasoning import AnswerCritic


def test_bibliography_is_not_a_retrieval_source():
    context = DvdContextBuilder().build_context(
        [
            {"name": "СП 55", "text": "См. [10].\n[11] Библиографическая ссылка."},
            {"name": "СП 42", "text": "Другой фрагмент."},
        ]
    )
    reducer = DvdContextReducer(None)
    assert reducer._sources(context) == {"[1]", "[2]"}
    parts = reducer._parts(context, 90)
    assert all(reducer._sources(p) <= {"[1]", "[2]"} for p in parts)


def test_extraction_parts_never_mix_independent_sources():
    context = DvdContextBuilder().build_context(
        [
            {"name": "Гостиницы", "text": "Требование к гостиницам."},
            {"name": "Школы", "text": "Другое требование к школам."},
        ]
    )
    reducer = DvdContextReducer(None)
    parts = reducer._parts(context, 32000)
    assert len(parts) == 2
    assert all(len(reducer._sources(part)) == 1 for part in parts)


async def test_quotes_receive_verified_source_labels_without_model_inline_citations():
    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "evidence": [
                        {
                            "source_id": "[4]",
                            "quotes": [
                                "Гостиницу размещают на неохраняемой территории."
                            ],
                        }
                    ],
                    "complete": True,
                }
            )
        }
    }
    reducer = DvdContextReducer(llm)
    source = "[4] СП 308, п. 31.6.1\nГостиницу размещают на неохраняемой территории."
    summary = await reducer._summarize("m", "Где размещают гостиницу?", source, 1500)
    assert "[4] СП 308, п. 31.6.1" in summary
    assert "Гостиницу размещают на неохраняемой территории." in summary


async def test_hotel_rule_cannot_be_rewritten_as_school_rule():
    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "evidence": [
                        {
                            "source_id": "[4]",
                            "quotes": ["Школу размещают на неохраняемой территории."],
                        }
                    ],
                    "complete": True,
                }
            )
        }
    }
    with pytest.raises(SummaryError, match="quote_not_in_source"):
        await DvdContextReducer(llm)._summarize(
            "m",
            "Школы?",
            "[4] СП 308\nГостиницу размещают на неохраняемой территории.",
            1500,
        )


async def test_reported_model_window_avoids_unnecessary_reduction(monkeypatch):
    monkeypatch.delenv("DVD_CONTEXT_WINDOW_TOKENS", raising=False)
    llm = SimpleNamespace(
        model_context_window=AsyncMock(return_value=65536), chat=AsyncMock()
    )
    reducer = DvdContextReducer(llm)
    source = "[1] Source\n" + "text " * 7000
    async with reducer.model_window("m"):
        result = await reducer.prepare("m", "q", source)
        assert reducer.window == 65536 and result.text == source
        llm.chat.assert_not_called()
    assert reducer.window == 8192


async def test_explicit_window_is_capped_by_server_and_is_task_local(monkeypatch):
    monkeypatch.setenv("DVD_CONTEXT_WINDOW_TOKENS", "16384")

    async def window(model):
        return {"small": 8192, "big": 65536}[model]

    reducer = DvdContextReducer(SimpleNamespace(model_context_window=window))

    async def probe(model):
        async with reducer.model_window(model):
            await asyncio.sleep(0)
            return reducer.window

    assert await asyncio.gather(probe("small"), probe("big")) == [8192, 16384]


async def test_openai_reads_deployed_window_from_models_metadata():
    adapter = OpenAiCompatAdapter("http://example.invalid/v1")
    adapter.client.models.list = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id="gpt-oss-20b", max_model_len=65536)]
        )
    )
    try:
        assert await adapter.model_context_window("gpt-oss-20b") == 65536
    finally:
        await adapter.client.close()


@pytest.mark.parametrize("window", [None, "65536", True, 1024])
async def test_invalid_metadata_uses_fallback_without_changing_explicit_cap(
    monkeypatch, window
):
    monkeypatch.delenv("DVD_CONTEXT_WINDOW_TOKENS", raising=False)
    reducer = DvdContextReducer(
        SimpleNamespace(model_context_window=AsyncMock(return_value=window))
    )
    async with reducer.model_window("m"):
        assert reducer.window == 8192
    reducer = DvdContextReducer(reducer.llm_client, window_tokens=16384)
    async with reducer.model_window("m"):
        assert reducer.window == 16384


async def test_openai_metadata_outage_does_not_fail_pipeline():
    import httpx
    from openai import APIConnectionError

    adapter = OpenAiCompatAdapter("http://example.invalid/v1")
    adapter.client.models.list = AsyncMock(
        side_effect=APIConnectionError(
            request=httpx.Request("GET", "http://example.invalid/v1/models")
        )
    )
    try:
        assert await adapter.model_context_window("m") is None
    finally:
        await adapter.client.close()


async def test_span_selection_copies_source_spelling_and_order():
    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "selections": {"[1]": [1, 0, 1]},
                    "complete": True,
                }
            )
        }
    }
    source = "[1] Test\nТ а б л и ц а 31.3\nКомнаты: 2–4-местные; 6,0 м2."
    result = await DvdContextReducer(llm)._select_evidence(
        "m", "Комнаты?", source, 1500
    )
    assert "Т а б л и ц а 31.3 Комнаты: 2–4-местные; 6,0 м2." in result
    assert result.count("Комнаты:") == 1


@pytest.mark.parametrize("indices", [[-1], [2]])
async def test_span_selection_rejects_out_of_range_indices(indices):
    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "selections": {"[1]": indices},
                    "complete": True,
                }
            )
        }
    }
    with pytest.raises(SummaryError, match="invalid_span_index"):
        await DvdContextReducer(llm)._select_evidence(
            "m", "q", "[1] Test\nOnly one span.", 1500
        )


async def test_quote_mismatch_retries_with_span_selection_and_audits_it():
    class Client:
        def __init__(self):
            self.calls = []

        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            if "selections" in kwargs["format"]["properties"]:
                data = {"selections": {"[1]": [0]}, "complete": True}
            else:
                data = {
                    "evidence": [{"source_id": "[1]", "quotes": ["Invented rule."]}],
                    "complete": True,
                }
            return {"message": {"content": json.dumps(data)}}

    llm = Client()
    reducer = DvdContextReducer(llm)
    result = await reducer.prepare(
        "m", "q", "[1] Test\nExact rule.\n" + "irrelevant. " * 500
    )
    assert not result.failed_parts
    assert "Exact rule." in result.text and "Invented" not in result.text
    selected_calls = [c for c in llm.calls if "selections" in c["format"]["properties"]]
    assert selected_calls
    assert any("draft_to_audit" in c["messages"][1]["content"] for c in selected_calls)


@pytest.mark.parametrize("field", ["unsupported_claims", "missing_requirements"])
async def test_evidence_defects_override_positive_model_verdict(field):
    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    field: ["Неверная ссылка на таблицу"],
                    "satisfied": True,
                }
            )
        }
    }
    verdict = await AnswerCritic(llm).review("m", "q", "source", "answer")
    assert not verdict.satisfied
    assert "Неверная ссылка" in verdict.critique
    schema = llm.chat.call_args.kwargs["format"]
    assert list(schema["properties"])[:2] == [
        "unsupported_claims",
        "missing_requirements",
    ]
    assert {"unsupported_claims", "missing_requirements"} <= set(schema["required"])


@pytest.mark.parametrize(
    "answer,source,rejected",
    [
        ("ЛИУ (лечебно-исцелительные учреждения)", "ЛИУ", True),
        (
            "ЛИУ (лечебно-исправительные учреждения)",
            "Лечебно-исправительные учреждения",
            False,
        ),
        ("СП (2017 г.)", "СП", False),
        ("Согласно таблице 27.3 [1]", "Пункт 27.3; таблица 31.3", True),
        ("Согласно таблице 31.3 [1]", "Т а б л и ц а 31.3", False),
        (
            "Таблица 27.3 в предоставленных фрагментах не приведена.",
            "Другой пункт",
            False,
        ),
        ("В источнике нет таблицы 27.3.", "Другой пункт", False),
    ],
)
def test_literal_citation_and_expansion_checks(answer, source, rejected):
    assert bool(AnswerCritic._literal_defects(source, answer)) is rejected


async def test_audited_irrelevant_parts_do_not_fill_window_with_repeated_headers():
    class Client:
        async def chat(self, **kwargs):
            schema = kwargs["format"]["properties"]["evidence"]
            assert schema["minItems"] == schema["maxItems"] == 1
            source_id = kwargs["format"]["$defs"]["SourceEvidence"]["properties"][
                "source_id"
            ]["enum"][0]
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "evidence": [{"source_id": source_id, "quotes": []}],
                            "complete": True,
                        }
                    )
                }
            }

    context = DvdContextBuilder().build_context(
        [{"name": "Long header " * 20, "text": "irrelevant " * 1000}]
    )
    result = await DvdContextReducer(Client(), window_tokens=8192).prepare(
        "m", "q", context
    )
    assert not result.failed_parts and result.processed_parts > 2
    assert result.reduction_rounds == 1
    assert "сведений для ответа не извлечено" in result.text
