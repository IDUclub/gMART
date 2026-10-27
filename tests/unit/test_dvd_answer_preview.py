"""Unverified sentence snapshots never become answer chunks or critic input."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.schema.dvd_response import DvdResponse
from src.agents.services.dvd import answer_preview, source_citations
from src.agents.services.dvd.answer_generation import DvdAnswerGenerator
from src.agents.services.dvd.answer_preview import AnswerPreview, complete_sentences
from src.agents.services.dvd.context_reducer import DvdContextReducer
from src.agents.services.dvd.run_metrics import DvdRunMetrics
from src.agents.services.service_entities.dvd_plan import Correction, CriticVerdict
from tests.helpers import answer_text, plan_json


class PieceModel:
    def __init__(self, replies, *, gate=None):
        self.replies = iter(replies)
        self.gate = gate
        self.closed = asyncio.Event()
        self.finished = False

    async def chat(self, *args, **kwargs):
        pieces, reason = next(self.replies)

        async def stream():
            try:
                for piece in pieces:
                    yield SimpleNamespace(message=SimpleNamespace(content=piece))
                    if self.gate is not None:
                        await self.gate.wait()
                self.finished = True
                yield SimpleNamespace(
                    message=SimpleNamespace(content=""), done_reason=reason
                )
            finally:
                self.closed.set()

        return stream()


async def generate(model, callback=None):
    reducer = DvdContextReducer(model, window_tokens=65536)
    return await DvdAnswerGenerator(reducer).generate(
        "m",
        "Вопрос",
        "[1] Источник",
        0,
        lambda context: [{"role": "user", "content": context}],
        iteration=1,
        progress_callback=callback,
    )


async def test_callback_gets_full_snapshots_and_no_callback_keeps_answer():
    replies = [(["Первое. ", "Второе", "."], "stop")]
    snapshots = []

    async def progress(text):
        snapshots.append(text)

    answer = await generate(PieceModel(replies), progress)
    assert snapshots == ["Первое. ", "Первое. Второе", "Первое. Второе."]
    assert answer == await generate(PieceModel(replies)) == snapshots[-1]


async def test_callback_continuation_snapshots_can_shrink_on_overlap():
    snapshots = []

    async def progress(text):
        snapshots.append(text)

    result = await generate(
        PieceModel([(["Начало\nФрагмент"], "length"), (["Фра", "гмент."], "stop")]),
        progress,
    )
    assert snapshots == [
        "Начало\nФрагмент",
        "Начало\nФра",
        "Начало\nФрагмент.",
    ]
    assert result == snapshots[-1]


async def test_callback_failure_is_logged_once_and_does_not_break_generation(
    monkeypatch,
):
    log = Mock()
    monkeypatch.setattr("src.agents.services.dvd.answer_generation.logger", log)
    callback = AsyncMock(side_effect=RuntimeError("UI failed"))
    assert (
        await generate(PieceModel([(["Раз. ", "Два."], "stop")]), callback)
        == "Раз. Два."
    )
    assert callback.await_count == 2
    log.warning.assert_called_once()


async def test_callback_keeps_receiving_snapshots_after_a_transient_failure():
    callback = AsyncMock(side_effect=[RuntimeError("UI failed"), None])
    assert (
        await generate(PieceModel([(["Раз. ", "Два."], "stop")]), callback)
        == "Раз. Два."
    )
    assert [call.args[0] for call in callback.await_args_list] == ["Раз. ", "Раз. Два."]


@pytest.mark.parametrize(
    "text",
    [
        "См. Пояснение",
        "По п. 2.1 требуется",
        "По ст. 51 требуется",
        "Размер 2.1 м. Далее",
        "В г. Москве",
        "На ул. Ленина, д. 2",
        "Это т.е. Условие",
        "Это и т. д. Условие",
        "По N 111 требуется 1.25 метра",
        "1. Первый пункт",
        "Норма. продолжение с маленькой буквы",
    ],
)
def test_cutter_keeps_abbreviations_addresses_decimals_and_list_markers(text):
    assert complete_sentences(text) == ""


@pytest.mark.parametrize(
    "next_sentence",
    [
        "Далее",
        "2. Пункт",
        "2) Пункт",
        "а) Пункт",
        "1.2. Пункт",
        "- Пункт",
        "* Пункт",
        "[1]",
    ],
)
@pytest.mark.parametrize("terminator", [".", "!", "?", "…"])
def test_cutter_recognizes_sentence_starts(terminator, next_sentence):
    sentence = "Требование" + terminator
    assert complete_sentences(sentence + " " + next_sentence) == sentence


def test_cutter_releases_complete_lines_and_final_sentence_only():
    assert (
        complete_sentences("- Без точки\n* Ещё строка\nХвост")
        == "- Без точки\n* Ещё строка"
    )
    assert complete_sentences("Готово.") == ""
    assert complete_sentences("Готово.", final=True) == "Готово."
    assert complete_sentences("Готово. Хвост", final=True) == "Готово."
    assert complete_sentences("Размер 2.1 м.", final=True) == ""


def test_interval_and_final_flush(monkeypatch):
    monkeypatch.setenv("DVD_ANSWER_PREVIEW_MIN_INTERVAL_MS", "700")
    clock = Mock(return_value=10.0)
    monkeypatch.setattr(answer_preview.time, "monotonic", clock)
    preview = AnswerPreview(None, 1)
    assert preview.event("Первая. Вторая") is not None
    clock.return_value = 10.1
    assert preview.event("Первая. Вторая. Третья") is None
    clock.return_value = 10.701
    assert preview.event("Первая. Вторая. Третья") is not None
    clock.return_value = 10.702
    assert preview.event("Первая. Вторая. Третья.", final=True) is not None
    assert preview.event("Первая. Вторая. Третья.", final=True) is None
    assert preview.event("Другая, даже более длинная версия. Хвост") is None
    next_iteration = AnswerPreview(None, 2, last_sent=preview.last_sent)
    assert next_iteration.event("Новая версия. Хвост") is None
    assert next_iteration.event("Новая версия.", final=True) is not None


@pytest.mark.parametrize("raw, expected", [("0", 0), ("-10", 0), ("bad", 0.7)])
def test_interval_configuration(monkeypatch, raw, expected):
    monkeypatch.setenv("DVD_ANSWER_PREVIEW_MIN_INTERVAL_MS", raw)
    assert answer_preview.min_interval() == expected


def pipeline(service, mcp):
    return service.run_document_qa_pipeline(
        dvd_mcp_client=mcp,
        token="tok",
        model="m",
        temperature=0,
        user_query="вопрос",
        chat_id="chat-1",
    )


async def test_pipeline_previews_are_live_readable_journaled_and_separate_from_ttft(
    service, fake_llm, fake_mcp, monkeypatch
):
    monkeypatch.setenv("DVD_ANSWER_PREVIEW", "true")
    monkeypatch.setenv("DVD_READABLE_CITATIONS", "true")
    monkeypatch.setenv("DVD_ANSWER_PREVIEW_MIN_INTERVAL_MS", "0")
    gate = asyncio.Event()
    pieces = ["Первое [1].\n", "Второе [1].\n", "Итог [1]."]
    model = PieceModel([(pieces, "stop")], gate=gate)
    service.llm_client = model
    fake_llm.json_responses = [plan_json()]
    service.critic.review = AsyncMock(return_value=CriticVerdict(satisfied=True))
    events, previews = [], []
    metrics = None
    async for event in pipeline(service, fake_mcp):
        events.append(event)
        if event["type"] == "status" and event["content"]["status"] == "answer_preview":
            previews.append(event["content"]["preview"])
            service.critic.review.assert_not_awaited()
            assert not any(e["type"] == "chunk" for e in events)
            request_id = events[0]["content"]["request_id"]
            metrics = service._active[request_id]["metrics"]
            assert metrics.ttft_ms is None
            assert metrics.first_preview_ms is not None
            assert event in await service.state_store.get_buffered_events(request_id)
            if len(previews) == 1:
                assert not model.finished
                gate.set()
    assert len(previews) == 3
    assert all(
        b.startswith(a) and len(b) > len(a) for a, b in zip(previews, previews[1:])
    )
    expected = source_citations.readable(
        "".join(pieces).strip(), source_citations.citations(fake_mcp.default_hits)
    )
    assert previews[-1] == answer_text(events) == expected
    assert "[1]" not in previews[0] and "СП 42.13330.2016" in previews[0]
    assert service.critic.review.call_args.args[3] == "".join(pieces).strip()
    assert metrics.ttft_ms >= metrics.first_preview_ms
    assert metrics.counters["preview_events"] == 3
    request_id = events[0]["content"]["request_id"]
    replay = [
        event
        async for event in service.run_document_qa_pipeline(
            dvd_mcp_client=fake_mcp,
            token="tok",
            model="m",
            temperature=0,
            user_query="вопрос",
            chat_id="chat-1",
            request_id=request_id,
        )
    ]
    assert replay == events


async def test_flag_off_matches_original_generation_event_sequence(
    service, fake_llm, fake_mcp, monkeypatch
):
    monkeypatch.setenv("DVD_ANSWER_PREVIEW", "false")
    fake_llm.json_responses = [plan_json()]
    fake_llm.answer_texts = ["Первое [1].\nВторое [1]."]
    service.critic.review = AsyncMock(return_value=CriticVerdict(satisfied=True))
    original = service._generate_answer
    calls = []

    async def generate_unchanged(*args, **kwargs):
        calls.append(kwargs)
        async for event in original(*args, **kwargs):
            yield event

    monkeypatch.setattr(service, "_generate_answer", generate_unchanged)
    events = [event async for event in pipeline(service, fake_mcp)]
    assert "progress_callback" not in calls[0]
    assert [(event["type"], event["content"].get("status")) for event in events] == [
        ("pipeline_started", None),
        ("status", "context_check"),
        ("status", "retrieval_planning"),
        ("status", "searching"),
        ("tool_call", None),
        ("status", "context_processing"),
        ("status", "answer_drafting"),
        ("status", "self_review"),
        ("chunk", None),
        ("status", "finalizing"),
        ("chunk", None),
    ]
    assert answer_text(events) == "Первое [1].\nВторое [1]."


@pytest.mark.parametrize("cancel_task", [False, True])
async def test_disconnect_cancels_generation_without_orphan_task(
    service, fake_llm, fake_mcp, monkeypatch, cancel_task
):
    monkeypatch.setenv("DVD_ANSWER_PREVIEW", "true")
    model = PieceModel(
        [(["Готовая строка.\n", "Не будет ответа."], "stop")], gate=asyncio.Event()
    )
    service.llm_client = model
    fake_llm.json_responses = [plan_json()]
    stream = pipeline(service, fake_mcp)
    if cancel_task:
        preview_seen = asyncio.Event()

        async def consume():
            try:
                async for event in stream:
                    if (
                        event["type"] == "status"
                        and event["content"]["status"] == "answer_preview"
                    ):
                        preview_seen.set()
            finally:
                await stream.aclose()

        waiting = asyncio.create_task(consume())
        await preview_seen.wait()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    else:
        async for event in stream:
            if (
                event["type"] == "status"
                and event["content"]["status"] == "answer_preview"
            ):
                break
        await stream.aclose()
    assert model.closed.is_set()
    assert not model.finished
    assert service._active == {}
    assert not any(
        task.get_name().startswith("dvd-answer-") for task in asyncio.all_tasks()
    )


def test_preview_metrics_keep_verified_ttft_separate(monkeypatch):
    metrics = DvdRunMetrics("request")
    elapsed = Mock(side_effect=[100, 200])
    monkeypatch.setattr(metrics, "_elapsed_ms", elapsed)
    metrics.preview()
    metrics.preview()
    assert metrics.ttft_ms is None
    metrics.first_answer()
    assert metrics.snapshot()["first_preview_ms"] == 100
    assert metrics.ttft_ms == 200
    assert metrics.counters["preview_events"] == 2


def test_sse_schema_preserves_preview_fields_and_existing_status_shape():
    event = {
        "type": "status",
        "content": {
            "status": "answer_preview",
            "text": "Предварительный анализ (ещё не проверен):\nГотово.",
            "preview": "Готово.",
            "iteration": 2,
        },
    }
    assert DvdResponse(**event).model_dump()["content"] == event["content"]
    status = {"status": "answer_drafting", "text": "Формирую ответ…"}
    assert DvdResponse(type="status", content=status).model_dump()["content"] == status


@pytest.mark.parametrize("repair", [False, True])
async def test_rejected_draft_replaces_iteration_and_targeted_repair_has_no_preview(
    service, fake_llm, fake_mcp, monkeypatch, repair
):
    monkeypatch.setenv("DVD_ANSWER_PREVIEW", "true")
    monkeypatch.setenv("DVD_ANSWER_PREVIEW_MIN_INTERVAL_MS", "0")
    fake_llm.json_responses = [plan_json()]
    model = PieceModel(
        [(["Первый черновик [1]."], "stop"), (["Новый ответ [1]."], "stop")]
    )
    service.llm_client = model
    service._revise_answer = AsyncMock(return_value="Новый ответ [1].")
    service.critic.review = AsyncMock(
        side_effect=[
            CriticVerdict(
                satisfied=False,
                critique="Исправить формулировку",
                corrections=(
                    [Correction(instruction="Исправить формулировку")] if repair else []
                ),
            ),
            CriticVerdict(satisfied=True),
        ]
    )
    events = [event async for event in pipeline(service, fake_mcp)]
    previews = [
        event["content"]
        for event in events
        if event["content"].get("status") == "answer_preview"
    ]
    assert [preview["iteration"] for preview in previews] == ([1] if repair else [1, 2])
    assert previews[0]["preview"] == "Первый черновик [1]."
    if not repair:
        assert previews[1]["preview"] == "Новый ответ [1]."
    assert answer_text(events) == "Новый ответ [1]."
    if repair:
        service._revise_answer.assert_awaited_once()


async def test_context_only_answer_does_not_preview(service, fake_llm, monkeypatch):
    from tests.unit.test_dvd_context_first import decision, seed
    from tests.unit.test_dvd_structured_retrieval import Pages, run

    monkeypatch.setenv("DVD_ANSWER_PREVIEW", "true")
    await seed(service)
    fake_llm.json_responses = [decision()]
    service.critic.review = AsyncMock(return_value=CriticVerdict(satisfied=True))
    no_search = Pages([])
    events = await run(service, no_search, "Расскажи вкраце о чём этот пункт")
    assert "Раздел определяет" in answer_text(events)
    assert not no_search.calls
    assert not any(e["content"].get("status") == "answer_preview" for e in events)


async def test_comparison_generation_previews_before_review(
    service, fake_llm, fake_mcp, monkeypatch
):
    from tests.unit.test_dvd_document_comparison import client, plan, response

    monkeypatch.setenv("DVD_ANSWER_PREVIEW", "true")
    fake_llm.json_responses = [plan().model_dump_json()]
    fake_llm.answer_texts = ["Число мест выросло с 10 до 20 [1] [2]."]
    service.critic.review = AsyncMock(return_value=CriticVerdict(satisfied=True))
    lookup = client()
    fake_mcp.find_document = lookup.find_document
    fake_mcp.document_versions = lookup.document_versions
    fake_mcp.compare_documents = AsyncMock(return_value=response())
    previews = []
    events = []
    async for event in service.run_document_qa_pipeline(
        dvd_mcp_client=fake_mcp,
        token="tok",
        model="m",
        temperature=0,
        user_query="Чем отличаются версии СП 42?",
        chat_id="chat-1",
    ):
        events.append(event)
        if event["content"].get("status") == "answer_preview":
            service.critic.review.assert_not_awaited()
            previews.append(event["content"]["preview"])
    assert previews == ["Число мест выросло с 10 до 20 [1] [2]."]
    assert previews[0] in answer_text(events)
