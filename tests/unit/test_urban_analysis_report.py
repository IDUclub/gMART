"""Grounding, fallback, file ownership and full report delivery."""

import base64
import json
from unittest.mock import AsyncMock

import pytest

from src.agents.common.files.temporary_file_store import TemporaryFileStore
from src.agents.services.urban_analysis.urban_analysis_report import (
    ReportResult,
    generate_report,
    report_file_event,
    report_system_prompt,
    report_user_prompt,
)


def steps():
    return [
        {
            "step": 1,
            "agent": "compliance",
            "status": "completed",
            "summary": "Нормы: 12",
        },
        {
            "step": 2,
            "agent": "provision",
            "status": "completed",
            "summary": "Сервис 82%",
        },
        {"step": 3, "agent": "pzz", "status": "failed", "summary": "Нет подключения"},
    ]


async def report(llm, data=None):
    result = ReportResult()
    emitted = [
        e
        async for e in generate_report(
            llm,
            model="m",
            temperature=0.1,
            steps=data or steps(),
            zones={"year": 2025, "source": "PZZ", "notes": ["Автовыбор"]},
            result=result,
        )
    ]
    return result.text, emitted


@pytest.mark.asyncio
async def test_report_prompt_evidence_limit_and_full_text(fake_llm):
    data = steps()
    data[0]["summary"] = "X" * 6001
    prompt = report_user_prompt(data, {"year": 2025, "source": "PZZ"})
    assert "X" * 6000 in prompt and "X" * 6001 not in prompt
    for value in (
        "completed",
        "failed",
        "Сервис 82%",
        "Нет подключения",
        "2025",
        "PZZ",
    ):
        assert value in prompt
    fake_llm.answer_texts = ["Полный отчёт " * 400]
    text, emitted = await report(fake_llm)
    assert len(text) > 1500
    assert (
        "".join(e["content"]["text"] for e in emitted if e["type"] == "chunk").strip()
        == text
    )
    assert fake_llm.chat_calls[0].request_kind == "urban_analysis.report"
    assert fake_llm.chat_calls[0].stream is True


@pytest.mark.asyncio
async def test_report_messages_separate_rules_from_evidence(fake_llm):
    await report(fake_llm)
    messages = fake_llm.chat_calls[0].messages
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == report_system_prompt()
    evidence = json.loads(messages[1]["content"])
    assert evidence["zones"] == {
        "year": 2025,
        "source": "PZZ",
        "notes": ["Автовыбор"],
    }
    for step, item in zip(steps(), evidence["steps"]):
        assert item == {
            "agent": step["agent"],
            "status": step["status"],
            "text": step["summary"],
        }
        assert step["summary"] not in messages[0]["content"]
    assert '"steps"' not in messages[0]["content"]
    assert '"zones"' not in messages[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_fallback_after_error_or_empty_response(fake_llm, empty):
    if not empty:
        fake_llm.chat = AsyncMock(side_effect=RuntimeError("down"))
    text, emitted = await report(fake_llm)
    assert "## Краткий вывод" in text and "## Ограничения анализа" in text
    assert "Нормы: 12" in text and "2025" in text
    assert any(e["type"] == "warning" for e in emitted)
    assert emitted[-1]["content"]["done"] is True


@pytest.mark.asyncio
async def test_interrupted_draft_is_replaced(fake_llm):
    async def broken():
        from types import SimpleNamespace

        yield SimpleNamespace(message=SimpleNamespace(content="Черновик"))
        raise RuntimeError("stream interrupted")

    fake_llm.chat = AsyncMock(return_value=broken())
    text, emitted = await report(fake_llm)
    assert "Черновик" not in text
    assert emitted[0]["content"]["iteration"] == 1
    assert emitted[-1]["content"]["iteration"] == 2


@pytest.mark.asyncio
async def test_file_contains_full_report_and_status_table(tmp_path):
    payload = (
        base64.urlsafe_b64encode(json.dumps({"sub": "user-17"}).encode())
        .decode()
        .rstrip("=")
    )
    token = f"header.{payload}.signature"
    store = TemporaryFileStore(tmp_path)
    text = "Полный текст отчёта " * 400
    event = await report_file_event(
        store, token=token, scenario_id=17, steps=steps(), text=text
    )
    assert event["content"]["filename"] == "urban-analysis-scenario-17.md"
    assert event["content"]["mime_type"] == "text/markdown"
    slot, file_id = event["content"]["url"].split("/")[-2:]
    assert slot == "urban_analysis_report"
    assert store.metadata(slot, file_id)["owner"] == "user-17"
    saved = (store.data_dir / slot / file_id).read_text("utf-8")
    assert "сценария 17" in saved and "| failed |" in saved
    assert text in saved


@pytest.mark.asyncio
async def test_no_owner_or_storage_means_no_file(tmp_path):
    store = TemporaryFileStore(tmp_path)
    for storage in (store, None):
        assert (
            await report_file_event(
                storage, token="invalid", scenario_id=17, steps=steps(), text="r"
            )
            is None
        )


@pytest.mark.asyncio
async def test_file_oserror_does_not_break_analysis(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.agents.services.urban_analysis.urban_analysis_report.user_id_from_jwt",
        lambda _: "u",
    )
    store = TemporaryFileStore(tmp_path)
    monkeypatch.setattr(
        store, "save", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk"))
    )
    assert (
        await report_file_event(
            store, token="t", scenario_id=17, steps=steps(), text="r"
        )
        is None
    )
