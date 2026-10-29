import json
import time
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.common.files.temporary_file_store import (
    OwnedFilesApp,
    TemporaryFileStore,
)
from src.agents.schema.file_event import file_sse_event
from src.agents.services.compilance.compliance_archive import ComplianceArchive
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.common.service_auth import internal_user_context_jwt
from tests.unit.test_compliance_phase2 import IDS, Mcp, fc
from tests.unit.test_compliance_phase2 import plan as executable_plan
from tests.unit.test_compliance_phase2 import service as compliance_service


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
        "status": (
            "violated"
            if compliance == "violated"
            else "partial" if verification == "partial" else "passed"
        ),
        "status_reason": "Нарушено объектов: 2" if compliance == "violated" else None,
        "archive_path": f"{compliance}/{rid}",
        "check_plan": source["check_plan"],
        "check_type_en": "distance",
        "check_type_ru": "расстояние",
        "equivalent_sources": source["equivalent_sources"],
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


async def _report_pipeline(
    tmp_path, *, owner="author", with_store=True, plans=None, mcp=None
):
    service, _ = compliance_service(tmp_path)
    if not with_store:
        service.file_store = None
    plans = plans if plans is not None else [executable_plan(0)]
    events = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=mcp if mcp is not None else Mcp(),
            request_id="report-run",
            scenario_id=772,
            restrictions=[
                {"id": plan["source"]["restriction_id"], "check_plan": plan}
                for plan in plans
            ],
            checkpoint={},
            owner=owner,
            model="mock",
        )
    ]
    return service, events


@pytest.mark.parametrize(
    "with_store,owner", [(False, "author"), (True, None), (True, "author")]
)
async def test_report_needs_a_store_and_an_owner(tmp_path, with_store, owner):
    service, events = await _report_pipeline(
        tmp_path, owner=owner, with_store=with_store
    )
    files = [event["content"] for event in events if event["type"] == "file"]
    if not with_store or not owner:
        assert files == []
    else:
        assert [item["mime_type"] for item in files] == [
            "text/markdown",
            "application/zip",
        ]
        for item in files:
            metadata = service.file_store.metadata(
                item["name"], item["url"].rsplit("/", 1)[1]
            )
            assert metadata["owner"] == owner
    assert any(event["type"] == "compliance_summary" for event in events)
    assert any(event["type"] == "feature_collection" for event in events)


async def test_file_event_is_sent_as_a_flat_genbuilder_frame(tmp_path):
    _, events = await _report_pipeline(tmp_path)
    files = [event for event in events if event["type"] == "file"]
    assert len(files) == 2
    for item in files:
        frame = file_sse_event(item)
        assert frame.event == "file"
        data = json.loads(frame.raw_data)
        assert "type" not in data and "content" not in data
        assert data == item["content"]
        assert data["download_url"] == data["url"] + "?download=1"
    assert files[0]["content"]["filename"] == "summary.md"
    assert files[1]["content"]["filename"].startswith("compliance_772_")
    assert files[1]["content"]["filename"].endswith(".zip")
    assert events[-1] == files[-1]
    assert file_sse_event({"type": "chunk", "content": {}}) is None


@pytest.mark.parametrize("same_reference", [False, True])
async def test_summary_groups_failed_passed_and_equivalent_norms(
    tmp_path, same_reference
):
    class MixedVerdicts(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            response = await super().execute_tool(name, arguments, meta)
            if (
                name == "CheckDistanceFromSource"
                and arguments["restriction_id"] == IDS[0]
            ):
                response["summary"] = {"violated_objects": 0, "passed_objects": 1}
                layer = response["norm_layer"]
                layer["meta"].update(
                    norm_status="passed",
                    norm_status_ru="выполнено",
                    norm_status_reason=None,
                )
                layer["features"] = [
                    item
                    for item in layer["features"]
                    if item["properties"]["feature_kind"] != "violation"
                ]
                for item in layer["features"]:
                    item["properties"].update(
                        norm_status="passed",
                        norm_status_ru="выполнено",
                        norm_status_reason=None,
                    )
                    if item["properties"]["feature_kind"] == "zone":
                        item["properties"].update(
                            status="passed", status_ru="выполнено", status_reason=None
                        )
            return response

    passed, failed = executable_plan(0), executable_plan(1, radius=300)
    equivalent = deepcopy(failed)
    equivalent["source"].update(
        restriction_id="00000000-0000-4000-8000-000000000003",
        clause_number=failed["source"]["clause_number"] if same_reference else "7.3",
    )
    _, events = await _report_pipeline(
        tmp_path, plans=[passed, failed, equivalent], mcp=MixedVerdicts()
    )
    root = ComplianceArchive("report-run", 772).root
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    report = (root / "summary.md").read_text("utf-8")
    assert summary == next(
        event["content"] for event in events if event["type"] == "compliance_summary"
    )
    assert summary["total_norms"] == summary["executed_norms"] == 2
    assert summary["total_restrictions"] == 3
    assert summary["counts"] == {"passed": 1, "violated": 1}
    assert summary["duplicate_checks"] == 1
    assert "Проверено: 2. Выполнено: 1. Не выполнено: 1." in report
    assert "Эквивалентных: 1." in report
    assert report.index("## Не выполнено") < report.index("## Выполнено")
    equivalent_result = next(
        item for item in summary["results"] if item.get("equivalent_to")
    )
    assert equivalent_result["restriction_id"] == equivalent["source"]["restriction_id"]
    assert equivalent_result["equivalent_to"] == IDS[1]
    assert equivalent_result["status"] == "violated"
    assert equivalent_result["archive_path"] == f"violated/{IDS[1]}"
    assert not list(root.glob(f"*/{equivalent['source']['restriction_id']}"))
    executed = json.loads((root / f"violated/{IDS[1]}/result.json").read_text("utf-8"))
    assert len(executed["equivalent_sources"]) == 2


async def test_norm_without_applicable_objects_is_a_formal_pass(tmp_path):
    class NoTargets(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetServicesByGeometry" and arguments["services_names"] == [
                "Театр"
            ]:
                self.calls.append((name, deepcopy(arguments)))
                return {arguments["layer_name"]: fc(complete=True)}
            return await super().execute_tool(name, arguments, meta)

    service, events = await _report_pipeline(tmp_path, mcp=NoTargets())
    root = ComplianceArchive("report-run", 772).root
    summary = json.loads((root / "summary.json").read_text("utf-8"))
    report = (root / "summary.md").read_text("utf-8")
    result = summary["results"][0]
    assert result["status"] == "passed"
    assert result["formal_pass"] is True
    assert result["coverage"]["applicable_objects"] == 0
    assert "no_applicable_objects" in result["warnings"]
    assert summary["formal_passed_norms"] == 1
    assert summary["counts"] == {"passed": 1}
    assert "Из них формально, без применимых объектов: 1." in report
    assert "отсутствие объектов не подтверждает соответствие проекта" in report
    assert "Выполнено формально: применимых объектов нет" in report
    detail = (root / result["archive_path"] / "report.md").read_text("utf-8")
    assert "Норма выполнена формально" in detail
    assert (
        json.loads(service.llm_client.messages[-1]["content"])["summary"]["norms"][0][
            "formal_pass"
        ]
        is True
    )
    assert any(event["type"] == "feature_collection" for event in events)


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


def test_history_keeps_file_link_without_role_and_download_url():
    part = RestrictionParserService._pipeline_item_to_chat_part(
        {
            "type": "file",
            "content": {
                "name": "compliance_report",
                "title": "Отчёт",
                "role": "result",
                "url": "/files/compliance_report/id",
                "download_url": "/files/compliance_report/id?download=1",
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
    payload = internal_user_context_jwt(user).split(".")[1]
    # This transport fixture carries an external JWT shape, not an internal context.
    return {"Authorization": f"Bearer e30.{payload}.sig"}


def test_internal_context_is_rejected_by_file_http_edge(files_client):
    _, client, stored = files_client
    response = client.get(
        stored.url,
        headers={"Authorization": f"Bearer {internal_user_context_jwt('author')}"},
    )
    assert response.status_code == 401


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
