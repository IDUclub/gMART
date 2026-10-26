"""Figure evidence keeps ordinary citation checks and permanent file replay."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.common.files.temporary_file_store import TemporaryFileStore
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.schema.dvd_response import DvdResponse
from src.agents.schema.file_event import FileEventContent, file_sse_event
from src.agents.services.dvd import literal_audit, source_citations
from src.agents.services.dvd.conversation_evidence import compact_hits
from src.agents.services.dvd.document_comparison import (
    build_comparison_context,
    comparison_numbers_supported,
    literal_comparison,
    prepare_comparison_response,
)
from src.agents.services.dvd.dvd_context import DvdContextBuilder, source_records
from src.agents.services.dvd.dvd_reasoning import AnswerCritic
from src.agents.services.dvd.figures import cited_figures, figure_file
from src.agents.services.dvd.partial_answer import PartialAnswerEvidence
from src.agents.services.service_entities.dvd_plan import AuditedClaim
from tests.helpers import FakeDvdMcpClient, events_of_type, plan_json, verdict_json
from tests.unit.test_dvd_rag_service import _run


def figure(index=1, **updates):
    return {
        "id": f"figure-{index}",
        "name": "СП 42.13330.2016",
        "kind": "figure",
        "type": "figure",
        "element_type": "figure",
        "order": index,
        "figure": {
            "caption": f"Рисунок {index} — Схема",
            "description": "Расстояние между зданиями составляет 15 м.",
            "description_generated": True,
        },
        **updates,
    }


@pytest.mark.parametrize("compact", [False, True])
def test_figure_context_labels_machine_description_and_preserves_numbers(compact):
    hit = figure()
    hits = [hit, figure(2)] if compact else [hit]
    context = DvdContextBuilder().build_context(hits)
    assert "Рисунок: Рисунок 1 — Схема" in context
    assert "Описание изображения (составлено моделью по картинке):" in context
    assert "15 м" in source_records(context)["[1]"][1]
    answer = "Расстояние между зданиями составляет 15 м [1]."
    assert comparison_numbers_supported({}, context, answer)
    assert not comparison_numbers_supported({}, context, answer.replace("15", "99"))
    assert not comparison_numbers_supported(
        {}, context, answer.replace("15 м", "15 га")
    )
    assert not AnswerCritic._literal_defects(context, answer)
    assert literal_audit.audit([answer], context)[0].status == "supported"
    assert (
        literal_audit.audit([answer.replace("15", "99")], context)[0].status
        == "insufficient"
    )
    assert AnswerCritic._literal_defects(context, answer.replace("[1]", "[99]"))
    assert "составлено моделью" in DvdContextBuilder().full_quote(hits)


@pytest.mark.parametrize(
    "caption,address",
    [
        ("Рисунок 3 — Схема", "рисунок 3"),
        ("Рисунок А.3 — Схема", "рисунок А.3"),
        ("Схема улицы", "рисунок «Схема улицы»"),
        (None, "рисунок"),
    ],
)
def test_figure_readable_citation(monkeypatch, caption, address):
    monkeypatch.setenv("DVD_READABLE_CITATIONS", "true")
    hit = figure(figure={"caption": caption})
    refs = source_citations.citations([hit])
    assert source_citations.readable("Схема [1]", refs) == (
        f"Схема (СП 42.13330.2016, {address})"
    )


def test_mcp_and_conversation_keep_figure_metadata():
    hit = figure()
    normalized = DvdMcpClient._normalize(
        {
            "hits": [
                SimpleNamespace(**{**hit, "figure": SimpleNamespace(**hit["figure"])})
            ]
        }
    )
    assert normalized["hits"][0]["figure"] == hit["figure"]
    assert compact_hits(normalized["hits"])[0]["figure"] == hit["figure"]


def test_only_cited_figures_in_citation_order_with_cap_and_no_bibliography():
    hits = [figure(i) for i in range(1, 10)]
    selected = cited_figures("[8] [2] [8] [6] [4] [1] [7] [3]\n> см. [9]", hits)
    assert [h["id"] for h in selected] == [
        "figure-8",
        "figure-2",
        "figure-6",
        "figure-4",
        "figure-1",
        "figure-7",
    ]
    assert not cited_figures("> библиография [1]", hits)
    assert not cited_figures("Ответ [1]", [{"type": "text", "id": "text"}])


def test_figure_file_is_relative_safe_and_compatible_with_file_schema():
    content = figure_file(figure(name='Документ:/<>"|?*'), project_id=42)
    assert content["url"] == "/documents/figures/figure-1"
    assert content["download_url"] == content["url"] + "?download=1"
    assert content["filename"].endswith("-рисунок 1.png")
    assert not any(c in content["filename"] for c in '/\\<>:"|?*')
    FileEventContent.model_validate(content)
    event = {"type": "file", "content": content}
    DvdResponse.model_validate(event)
    assert file_sse_event(event).event == "file"
    user_file = figure_file(figure(user_id="user"), project_id=42)
    assert user_file["url"] == "/documents/figures/figure-1?project_id=42"
    assert user_file["download_url"].endswith("?project_id=42&download=1")


def test_figure_sse_uses_flat_file_frame_with_replay_id(monkeypatch):
    import json

    from tests.unit.test_dvd_routers import FakeRagService, _build_client

    content = figure_file(figure())

    async def stream(*args, **kwargs):
        yield {"type": "file", "content": content, "event_id": 7}

    monkeypatch.setattr("src.agents.routers.dvd_controller.stream_document_run", stream)
    with _build_client(None, FakeRagService()) as client:
        response = client.get("/documents/qa/stream?request=рисунок&model=test")
    assert response.status_code == 200
    assert "event: file" in response.text
    assert "id: 7" in response.text
    data = next(
        line[5:].strip()
        for line in response.text.splitlines()
        if line.startswith("data:")
    )
    assert json.loads(data) == content


@pytest.mark.parametrize("readable", [False, True])
async def test_accepted_answer_emits_only_cited_figures_and_replays(
    service, fake_llm, monkeypatch, readable
):
    monkeypatch.setenv("DVD_READABLE_CITATIONS", str(readable).lower())
    service.dvd_api_url = "http://dvd"
    fake_llm.json_responses = [plan_json(), verdict_json()]
    fake_llm.answer_texts = ["По машинному описанию расстояние — 15 м [2]."]
    mcp = FakeDvdMcpClient(default_hits=[figure(), figure(2)])
    events = await _run(service, mcp)
    files = events_of_type(events, "file")
    assert len(files) == 1
    assert files[0]["content"]["url"] == "/documents/figures/figure-2"
    position = events.index(files[0])
    assert events[position - 1]["type"] == "chunk"
    assert events[position - 1]["content"]["text"]
    assert events[-1]["content"]["done"]
    collected = service._schedule_persist_answer.call_args.args[2]
    assert collected["figure_files"] == [files[0]["content"]]
    request_id = events_of_type(events, "pipeline_started")[0]["content"]["request_id"]
    assert await _run(service, mcp, request_id=request_id) == events


@pytest.mark.parametrize("api_url,cited", [(None, True), ("http://dvd", False)])
async def test_no_figure_events_when_api_missing_or_figure_not_cited(
    service, fake_llm, api_url, cited
):
    service.dvd_api_url = api_url
    fake_llm.json_responses = [plan_json(), verdict_json()]
    fake_llm.answer_texts = ["Схема [1]." if cited else "Ответ без ссылок."]
    events = await _run(service, FakeDvdMcpClient(default_hits=[figure()]))
    assert not events_of_type(events, "file")


async def test_user_figure_event_resolves_project(service):
    service.dvd_api_url = "http://dvd"
    collected = {
        "citation_hits": [figure(user_id="u")],
        "figure_token": "tok",
        "figure_scenario_id": 772,
    }
    events = [e async for e in service._figure_events("r", collected, "Схема [1]")]
    assert events[0]["content"]["url"].endswith("?project_id=4242")
    assert service.urban_api_client.calls[-1] == ("tok", 772)


async def test_figure_events_are_capped_and_follow_first_citation(service):
    service.dvd_api_url = "http://dvd"
    collected = {"citation_hits": [figure(i) for i in range(1, 9)]}
    events = [
        e
        async for e in service._figure_events(
            "capped", collected, "[8] [3] [8] [1] [5] [2] [7] [4]"
        )
    ]
    assert [e["content"]["url"] for e in events] == [
        f"/documents/figures/figure-{i}" for i in (8, 3, 1, 5, 2, 7)
    ]


async def test_failed_project_lookup_keeps_scenario_scope(service):
    service.dvd_api_url = "http://dvd"
    service.urban_api_client.get_project_by_scenario = AsyncMock(
        side_effect=RuntimeError("offline")
    )
    collected = {
        "citation_hits": [figure(owner="user")],
        "figure_scenario_id": 772,
    }
    events = [e async for e in service._figure_events("r", collected, "[1]")]
    assert events[0]["content"]["url"].endswith("?scenario_id=772")


async def test_file_parts_persist_after_text(service):
    service.add_complex_message = AsyncMock()
    content = figure_file(figure())
    await service._persist_answer(
        "tok", "chat", {"final_answer": "Ответ", "figure_files": [content]}, None
    )
    parts = service.add_complex_message.call_args.args[3]
    assert [p.kind for p in parts] == ["text", "file"]
    assert parts[-1].payload == {
        k: v for k, v in content.items() if k not in {"role", "download_url"}
    }


async def test_partial_answer_keeps_figure_identity_across_source_renumbering(
    service, monkeypatch
):
    hit = figure()
    context = DvdContextBuilder().build_context([hit])
    evidence = PartialAnswerEvidence()
    claim = "Расстояние между зданиями составляет 15 м [1]."
    evidence.add(
        [
            AuditedClaim(
                text=claim,
                status="supported",
                evidence=[{"source_id": "[1]", "quote": hit["figure"]["description"]}],
            )
        ],
        claim,
        context,
        AnswerCritic._literal_defects,
        hits=[hit],
    )
    assert evidence.candidates()[0]["evidence"][0]["hit"] == hit
    monkeypatch.setenv("DVD_READABLE_CITATIONS", "true")
    service.dvd_api_url = "http://dvd"
    service.critic.select_partial = AsyncMock(return_value=[0])
    collected = {"tool_calls": [], "citation_hits": [figure(99)]}
    events = [
        e
        async for e in service._finish_partial_answer(
            "m", "Расстояние?", "partial", collected, 3, evidence
        )
    ]
    files = events_of_type(events, "file")
    assert len(files) == 1
    assert files[0]["content"]["url"] == "/documents/figures/figure-1"


async def test_reducer_preserves_figure_source_and_machine_description(
    service, fake_llm
):
    import json

    hit = figure()
    context = DvdContextBuilder().build_context([hit])
    fake_llm.json_responses = [
        json.dumps(
            {
                "evidence": [
                    {"source_id": "[1]", "quotes": [hit["figure"]["description"]]}
                ],
                "complete": True,
            }
        )
    ]
    reduced = await service.context_reducer._summarize(
        "m", "Расстояние?", context, 2000
    )
    header, body = source_records(reduced)["[1]"]
    assert "рисунок 1" in header
    assert "15 м" in body
    system = fake_llm.chat_calls[-1].messages[0]["content"]
    assert "машинного чтения" in system


def test_comparison_accepts_figure_without_plain_text():
    from tests.unit.test_dvd_document_comparison import response

    payload = response()
    payload["changes"][0]["left"] = figure()
    payload["changes"][0]["right"] = figure(2)
    _, context, refs = build_comparison_context(payload)
    assert "составлено моделью" in context
    assert any(address == "рисунок 1" for _, address in refs.values())
    assert "15 м" in literal_comparison(payload)
    prepare_comparison_response(payload, budget=300)


def test_temporary_file_links_are_relative(tmp_path):
    stored = TemporaryFileStore(tmp_path).save(
        "report", b"report", owner="u", filename="report.md", mime_type="text/markdown"
    )
    assert stored.url == f"/files/report/{stored.file_id}"
    assert stored.download_url == stored.url + "?download=1"
