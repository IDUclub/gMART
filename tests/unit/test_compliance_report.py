import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.files.temporary_file_store import (
    OwnedFilesApp,
    TemporaryFileStore,
)
from src.agents.schema.file_event import file_sse_event
from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_report import (
    MAX_VIOLATORS_PER_NORM,
    build_compliance_report,
)
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities.compliance import (
    ComplianceResult,
    ComplianceSummary,
    VerificationCoverage,
)
from src.common.service_auth import internal_user_context_jwt


def _source(rid, clause, text="Требование нормы"):
    return {
        "restriction_id": rid,
        "document_name": "СП 42.13330.2016 Градостроительство",
        "clause_number": clause,
        "extraction_text": text,
    }


def _plan(rid="r1", clause="7.1", distance=50):
    return {
        "schema_version": "1.0",
        "template": "distance_from_source",
        "template_version": 1,
        "params": {
            "source_layer": "schools",
            "targets": ["roads"],
            "geometry_mode": "buffered",
            "distance_m": distance,
            "predicate": "intersects",
            "violation_when": "matched",
            "result_mode": "both",
        },
        "declared_requirements": {
            "layers": [
                {"role": "schools", "entity": "школа", "entity_type": "service"},
                {
                    "role": "roads",
                    "entity": "дорога",
                    "entity_type": "physical_object",
                },
            ],
            "attributes": [],
        },
        "source": _source(rid, clause),
        "planner_status": "auto",
    }


def _evidence(index, violated=True):
    return {
        "object_ref": {"id": f"physical_object/{index}", "name": f"Дорога | {index}"},
        "generator_refs": [{"id": "service/1", "name": "Школа № 1"}],
        "measured_value": 1,
        "unit": "count",
        "threshold": 50,
        "operator": "matched",
        "radius_m": 50,
        "violated": violated,
    }


def _result(
    rid,
    compliance,
    verification="complete",
    violated=0,
    equivalent=(),
    evidence=(),
    clause="7.1",
):
    source = {**_source(rid, clause), "check_plan": _plan(rid, clause)}
    source["equivalent_sources"] = [_source(rid, clause), *equivalent]
    return {
        "restriction_id": rid,
        "template": "distance_from_source",
        "template_version": 1,
        "verification_status": verification,
        "compliance_status": compliance,
        "coverage": {
            "applicable_objects": 10,
            "checked_objects": 8 if verification == "partial" else 10,
            "unchecked_objects": 2 if verification == "partial" else 0,
            "fill_rate": 0.8 if verification == "partial" else 1.0,
        },
        "summary": {"violated_objects": violated, "passed_objects": 10 - violated},
        "resolved_requirements": [
            {
                "role": "roads",
                "requirement_type": "layer",
                "resolved": True,
                "layer": "Дорога",
            }
        ],
        "source": source,
        "evidence": list(evidence),
    }


def _summary(results, skipped=0):
    return {"results": results, "skipped_without_plan": skipped}


def test_report_groups_failed_passed_and_equivalent_norms():
    violations = [_evidence(index) for index in range(25)]
    report = build_compliance_report(
        _summary(
            [
                _result(
                    "r1",
                    "violated",
                    violated=25,
                    evidence=[*violations, _evidence(99, violated=False)],
                    equivalent=[_source("r9", "5.5", "Равнозначное требование")],
                ),
                _result("r2", "passed", verification="partial", clause="8.2"),
                _result("r3", "unknown", verification="unverifiable", clause="9.9"),
            ],
            skipped=4,
        )
    )

    failed, rest = report.split("## Прошли проверку")
    passed, equivalent = rest.split("## Проверены как эквивалентные")
    assert "| Норм проверено | 3 |" in report
    assert "| Не прошли проверку | 1 |" in report
    assert "| Прошли проверку | 1 |" in report
    assert "| Проверены как эквивалентные | 1 |" in report
    assert "| Не удалось проверить | 1 |" in report
    assert "| Пропущено без исполнимого плана | 4 |" in report
    # Unverified norms are counted but never listed.
    assert "п. 9.9" not in report

    assert "### 1. СП 42.13330.2016, п. 7.1" in failed
    assert "**Требование:** Требование нормы" in failed
    assert "Расстояние, м: 50" in failed
    assert "Источник: школа" in failed and "Проверяемые объекты: дорога" in failed
    assert "Данные «дорога»: Дорога" in failed
    assert "с нарушением — 25" in failed
    assert "Эквивалентные нормы:** СП 42.13330.2016, п. 5.5" in failed
    assert failed.count("| Дорога \\| ") == MAX_VIOLATORS_PER_NORM
    assert "…и ещё объектов с нарушением: 5" in failed
    assert "Дорога \\| 99" not in failed

    assert "### 2. СП 42.13330.2016, п. 8.2" in passed
    assert "проверена частично" in passed and "не проверено — 2" in passed

    assert "Группа нормы № 1: СП 42.13330.2016, п. 7.1" in equivalent
    assert "СП 42.13330.2016, п. 5.5 — Равнозначное требование" in equivalent
    assert "не прошла проверку" in equivalent


def test_merged_norms_with_the_same_label_are_counted_by_restriction_id():
    # Two unnumbered clauses of one document share the label "СП 42.13330.2016";
    # the merge happened and must be reported, not hidden as a duplicate label.
    same_label = {**_source("r9", ""), "extraction_text": "Другая формулировка"}
    result = _result("r1", "passed", clause="")
    result["source"]["equivalent_sources"] = [_source("r1", ""), same_label]
    report = build_compliance_report(_summary([result]))

    assert "| Норм проверено | 2 |" in report
    assert "| Проверены как эквивалентные | 1 |" in report
    assert "СП 42.13330.2016 — Другая формулировка" in report
    assert "Всего норм в группе: 2." in report


def test_norm_without_applicable_objects_is_a_formal_pass():
    real = _result("r1", "passed")
    empty = _result("r2", "passed", clause="8.2")
    empty["warnings"] = ["no_applicable_objects"]
    empty["coverage"] = {
        "applicable_objects": 0,
        "checked_objects": 0,
        "unchecked_objects": 0,
        "fill_rate": 1.0,
    }
    report = build_compliance_report(_summary([real, empty]))

    passed, vacuous = report.split("## Прошли без применимых объектов")
    assert "| Прошли проверку | 2 |" in report
    assert "| из них без применимых объектов | 1 |" in report
    assert "п. 7.1" in passed and "п. 8.2" not in passed
    assert "### 2. СП 42.13330.2016, п. 8.2" in vacuous
    assert "прошла формально" in vacuous
    # No "100% filled" claim for an empty object set.
    assert "заполненность" not in vacuous


def test_summary_text_separates_notes_and_formal_passes():
    empty = _result("r2", "passed", clause="8.2")
    empty["warnings"] = ["no_applicable_objects"]
    summary = {
        "total_norms": 2,
        "violated_norms": 1,
        "passed_norms": 1,
        "unverifiable_norms": 0,
        "unsupported_norms": 0,
        "partial_norms": 0,
        "results": [_result("r1", "violated", violated=3), empty],
    }
    text = RestrictionParserService._compliance_summary_text(summary)
    assert "Из них формально, без применимых объектов в сценарии: 1." in text


def test_report_is_not_built_without_checked_norms():
    assert build_compliance_report(_summary([])) is None
    assert (
        build_compliance_report(
            _summary([_result("r1", "unknown", verification="unsupported")])
        )
        is None
    )


def _service(file_store, compliance="violated"):
    service = object.__new__(RestrictionParserService)
    service.file_store = file_store
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(),
        save_checkpoint=AsyncMock(),
        set_status=AsyncMock(),
    )
    violated = compliance == "violated"
    result = ComplianceResult(
        restriction_id="r1",
        template="distance_from_source",
        template_version=1,
        verification_status="complete",
        compliance_status=compliance,
        coverage=VerificationCoverage(
            applicable_objects=1, checked_objects=1, unchecked_objects=0, fill_rate=1
        ),
        summary=ComplianceSummary(
            violated_objects=int(violated), passed_objects=int(not violated)
        ),
        source=_source("r1", "7.1"),
        evidence=[
            {
                "restriction_id": "r1",
                "template": "distance_from_source",
                "template_version": 1,
                "object_ref": {"id": "physical_object/1", "name": "Дорога"},
                "operation": "buffer+intersects",
                "violated": violated,
            }
        ],
    )
    service.compliance_executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(result=result, tool_calls=[], timings_ms={})
        )
    )
    return service


async def _run(service, owner="user-1"):
    return [
        event
        async for event in service._run_executable_compliance(
            mcp_client=object(),
            request_id="request-1",
            scenario_id=772,
            restrictions=[{"id": "r1", "check_plan": _plan()}],
            checkpoint={},
            owner=owner,
        )
    ]


async def test_pipeline_closes_stream_with_report_file_event(tmp_path):
    store = TemporaryFileStore(tmp_path, public_base_url="http://gmart.example/")
    events = await _run(_service(store))
    await asyncio.sleep(0)

    for event in events:
        RestrictionsResponse.model_validate(event)
    assert [event["type"] for event in events][-2:] == ["chunk", "file"]
    descriptor = events[-1]["content"]
    assert descriptor["name"] == "compliance_report"
    assert descriptor["role"] == "result"
    assert descriptor["mime_type"] == "text/markdown"
    assert descriptor["source_service"] == "gmart"
    assert descriptor["filename"].startswith("compliance_report_772_")
    assert descriptor["filename"].endswith(".md")
    assert descriptor["url"].startswith("http://gmart.example/files/compliance_report/")
    assert descriptor["download_url"] == descriptor["url"] + "?download=1"

    file_id = descriptor["url"].rsplit("/", 1)[1]
    assert store.metadata("compliance_report", file_id)["owner"] == "user-1"
    content = (store.data_dir / "compliance_report" / file_id).read_text("utf-8")
    assert content.startswith("# Отчёт о проверке соответствия нормам")


@pytest.mark.parametrize("with_store, owner", [(False, "user-1"), (True, None)])
async def test_report_needs_a_store_and_an_owner(tmp_path, with_store, owner):
    store = TemporaryFileStore(tmp_path) if with_store else None
    events = await _run(_service(store), owner=owner)
    assert "file" not in [event["type"] for event in events]


async def test_file_event_is_sent_as_a_flat_genbuilder_frame(tmp_path):
    events = await _run(_service(TemporaryFileStore(tmp_path)))
    frame = file_sse_event(events[-1])
    assert frame.event == "file"
    data = json.loads(frame.raw_data)
    assert "type" not in data and "content" not in data
    assert data["title"] == "Отчёт о проверке соответствия нормам"
    assert data["url"].startswith("/files/compliance_report/")
    assert file_sse_event({"type": "chunk", "content": {}}) is None


def test_history_keeps_file_link_without_role_and_download_url():
    part = RestrictionParserService._pipeline_item_to_chat_part(
        {
            "type": "file",
            "content": {
                "name": "compliance_report",
                "title": "Отчёт",
                "role": "result",
                "url": "http://gmart/files/compliance_report/id",
                "download_url": "http://gmart/files/compliance_report/id?download=1",
                "filename": "report.md",
                "mime_type": "text/markdown",
                "source_service": "gmart",
            },
        },
        text_only=True,
    )
    assert part.kind == "file"
    assert set(part.payload) == {
        "name",
        "title",
        "url",
        "filename",
        "mime_type",
        "source_service",
    }


@pytest.fixture
def files_client(tmp_path):
    store = TemporaryFileStore(tmp_path)
    app = FastAPI()
    app.mount("/files", OwnedFilesApp(store))
    stored = store.save(
        "compliance_report",
        "# Отчёт\n".encode(),
        owner="author",
        filename="compliance_report_1.md",
        mime_type="text/markdown",
    )
    return store, TestClient(app), stored


def _auth(user):
    return {"Authorization": f"Bearer {internal_user_context_jwt(user)}"}


def test_author_can_view_and_download_report(files_client):
    _, client, stored = files_client

    inline = client.get(stored.url, headers=_auth("author"))
    assert inline.status_code == 200
    assert inline.text == "# Отчёт\n"
    assert inline.headers["content-type"] == "text/markdown; charset=utf-8"
    assert inline.headers["content-disposition"].startswith("inline;")
    assert inline.headers["cache-control"] == "no-store"

    download = client.get(stored.download_url, headers=_auth("author"))
    assert download.status_code == 200
    assert download.headers["content-disposition"] == (
        "attachment; filename*=UTF-8''compliance_report_1.md"
    )


def test_report_is_hidden_from_other_users_and_anonymous_requests(files_client):
    _, client, stored = files_client
    assert client.get(stored.url, headers=_auth("someone-else")).status_code == 404
    assert client.get(stored.url).status_code == 401
    traversal = "/files/compliance_report/%2e%2e/%2e%2e/meta/compliance_report"
    assert client.get(traversal, headers=_auth("author")).status_code == 404
    assert (
        client.get("/files/compliance_report/unknown", headers=_auth("author"))
    ).status_code == 404


def test_expired_report_is_removed_and_unavailable(files_client, monkeypatch):
    store, client, stored = files_client
    expired = time.time() + store.ttl_seconds + 1
    monkeypatch.setattr(
        "src.agents.common.files.temporary_file_store.time.time", lambda: expired
    )

    assert client.get(stored.url, headers=_auth("author")).status_code == 404
    assert not (store.data_dir / "compliance_report" / stored.file_id).exists()


def test_purge_removes_only_expired_files(tmp_path, monkeypatch):
    store = TemporaryFileStore(tmp_path, ttl_seconds=10)

    def save(content):
        return store.save(
            "compliance_report",
            content,
            owner="u",
            filename="report.md",
            mime_type="text/markdown",
        )

    old = save(b"old")
    now = time.time()
    monkeypatch.setattr(
        "src.agents.common.files.temporary_file_store.time.time", lambda: now + 11
    )
    # Saving purges files whose TTL is over.
    fresh = save(b"new")

    assert not (store.data_dir / "compliance_report" / old.file_id).exists()
    assert store.metadata("compliance_report", old.file_id) is None
    assert store.metadata("compliance_report", fresh.file_id) is not None
    assert store.purge_expired() == 0
