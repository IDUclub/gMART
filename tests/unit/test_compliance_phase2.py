"""Phase-2 contracts, with every MCP/LLM/Urban API boundary mocked."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZipFile

import pytest

from src.agents.a2a.compliance_executor import ComplianceAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.files.temporary_file_store import TemporaryFileStore
from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_archive import ComplianceArchive
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_inventory import (
    RestrictionZoneBuilder,
    source_radius,
)
from src.agents.services.compilance.compliance_progress import STAGES, progress_event
from src.agents.services.compilance.compliance_retrieval import intersection_type
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)

IDS = ["00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"]
POLYGON = {
    "type": "Polygon",
    "coordinates": [[[30, 60], [31, 60], [31, 61], [30, 61], [30, 60]]],
}


def feature(geometry=None, **properties):
    return {
        "type": "Feature",
        "geometry": geometry or POLYGON,
        "properties": properties,
    }


def fc(*features, **meta):
    return {"type": "FeatureCollection", "features": list(features), "meta": meta}


def plan(index, radius=100, predicate="intersects"):
    return {
        "schema_version": "1.0",
        "template": "distance_from_source",
        "template_version": 1,
        "params": {
            "source_layer": "source",
            "targets": ["target"],
            "geometry_mode": "buffered",
            "distance_m": radius,
            "predicate": predicate,
            "violation_when": "matched",
            "result_mode": "both",
        },
        "declared_requirements": {
            "layers": [
                {
                    "role": "source",
                    "entity": "Школа",
                    "entity_type": "service",
                    "geometry_types": ["Point"],
                },
                {
                    "role": "target",
                    "entity": "Театр",
                    "entity_type": "service",
                    "geometry_types": ["Point"],
                },
            ],
            "attributes": [],
        },
        "source": {
            "restriction_id": IDS[index],
            "document_name": "СП 42.13330.2016",
            "clause_number": f"7.{index + 1}",
            "extraction_text": "Полный исходный текст нормы, включая условия применения.",
        },
        "planner_status": "auto",
    }


class Mcp:
    def __init__(self):
        self.calls = []

    async def resolve_urban_entity_types(self, *, service_names, physical_object_names):
        return {
            "service": {
                name: {
                    "found": True,
                    "canonical_name": name,
                    "type_id": 1 if name == "Школа" else 2,
                }
                for name in service_names
            },
            "physical_object": {
                name: {"found": True, "canonical_name": name, "type_id": 1}
                for name in physical_object_names
            },
        }

    async def execute_tool(self, name, arguments, meta=None):
        self.calls.append((name, deepcopy(arguments)))
        if name == "GetProjectTerritory":
            return {"project_territory": fc(feature())}
        if name == "GetServicesByGeometry":
            # An out-of-project source is retained by MCP's buffered query.
            return {
                arguments["layer_name"]: fc(
                    feature(
                        {"type": "Point", "coordinates": [29.9999, 60]},
                        service_id=1,
                        name=arguments["services_names"][0],
                    ),
                    complete=True,
                )
            }
        if name == "CreateRestrictionZones":
            properties = arguments["properties"]
            common = {
                "norm_id": properties["restriction_id"],
                "norm_status": "not_checked",
                "norm_status_ru": "не проверялось",
                "norm_status_reason": None,
                "check_type_en": "distance",
                "check_type_ru": "расстояние",
                "norm_text": properties["provenance"]["extraction_text"],
                "feature_kind": "zone",
                "status": "not_checked",
                "status_ru": "не проверялось",
            }
            layer = fc(
                feature(**common),
                norm_id=common["norm_id"],
                norm_status="not_checked",
                group_by="norm_id",
                group_properties=list(common),
            )
            return {
                arguments["layer_name"]: fc(feature(**properties)),
                "norm_layer": layer,
            }
        if name == "CheckDistanceFromSource":
            assert "project_territory" in arguments["layers"]
            common = {
                "norm_id": arguments["restriction_id"],
                "norm_status": "violated",
                "norm_status_ru": "не выполнено",
                "norm_status_reason": "Нарушена одна зона.",
                "check_type_en": "distance",
                "check_type_ru": "расстояние",
            }
            layer = fc(
                feature(
                    **common,
                    feature_kind="zone",
                    status="violated",
                    status_ru="не выполнено",
                    status_reason="Объект в зоне.",
                ),
                feature(
                    **common,
                    feature_kind="violation",
                    explanation_ru="До школы 60 м, требуется ≥ 100 м, не хватает 40 м.",
                    measured_value=60,
                    required_value=100,
                    deficit=40,
                    violation_geometry_mode="intersection",
                ),
                feature(
                    **common,
                    feature_kind="object",
                    object_name="Театр",
                    zone_refs=[{"id": "z1"}],
                ),
                **common,
                group_by="norm_id",
                group_properties=list(common),
            )
            return {
                "coverage": {
                    "applicable_objects": 1,
                    "checked_objects": 1,
                    "unchecked_objects": 0,
                    "fill_rate": 1,
                },
                "summary": {"violated_objects": 1, "passed_objects": 0},
                "evidence": [],
                "norm_layer": layer,
            }
        raise AssertionError(name)


class Llm:
    def __init__(self, failure=False):
        self.failure, self.messages = failure, None

    async def chat(self, model, messages, **kwargs):
        self.messages = messages
        if self.failure:
            raise TimeoutError("LLM timeout")

        async def stream():
            for text in (
                "Проверка завершена. ",
                "Рассмотрите нарушенные нормы в первую очередь.",
            ):
                yield SimpleNamespace(message=SimpleNamespace(content=text))

        return stream()


def service(tmp_path, failure=False):
    svc = object.__new__(RestrictionParserService)
    checkpoints = {}

    async def save(request_id, step, value):
        checkpoints[step] = deepcopy(value)

    svc.state_store = SimpleNamespace(
        buffer_event=AsyncMock(),
        save_checkpoint=AsyncMock(side_effect=save),
        set_status=AsyncMock(),
    )
    svc.compliance_executor = ComplianceTemplateExecutor(effects_client_factory=None)
    svc.zone_builder = RestrictionZoneBuilder(svc.compliance_executor)
    svc.file_store = TemporaryFileStore(tmp_path / "files")
    svc.llm_client = Llm(failure)
    return svc, checkpoints


async def run(svc, mcp, checkpoint=None, inventory=False, plans=None):
    plans = plans or [plan(0), plan(1, radius=300)]
    method = (
        svc._run_restriction_inventory if inventory else svc._run_executable_compliance
    )
    kwargs = {} if inventory else {"model": "mock", "user_query": "Проверь проект"}
    return [
        item
        async for item in method(
            mcp_client=mcp,
            request_id="run-1",
            scenario_id=772,
            restrictions=[
                {"id": p["source"]["restriction_id"], "check_plan": p} for p in plans
            ],
            checkpoint=checkpoint or {},
            owner="author",
            **kwargs,
        )
    ]


async def test_two_phase_service_queries_use_max_radius_and_union_once(tmp_path):
    svc, _ = service(tmp_path)
    mcp = Mcp()
    events = await run(svc, mcp)
    queries = [args for name, args in mcp.calls if name == "GetServicesByGeometry"]
    assert len(queries) == 2
    source, target = queries
    assert source["services_names"] == ["Школа"]
    assert source["zones_layers"] == ["project_territory"]
    assert source["buffer_m"] == 300
    assert target["services_names"] == ["Театр"]
    assert len(target["zones_layers"]) == 2
    assert set(target["zones_layers"]) == set(target["layers"])
    assert all(name.startswith("Норма — ") for name in target["zones_layers"])
    assert target["intersection_type"] == "intersects"
    assert target["buffer_m"] == 0
    calls = [
        call
        for item in events
        if item["type"] == "tool_call"
        for call in item["content"]["tool_calls"]
    ]
    assert (
        sum(call["function"]["name"] == "GetServicesByGeometry" for call in calls) == 2
    )
    assert all("layers" not in call["function"]["arguments"] for call in calls)
    provided = {call["function"]["arguments"].get("layer_name") for call in calls}
    check = next(
        call["function"]["arguments"]
        for call in calls
        if call["function"]["name"].startswith("Check")
    )
    assert check["source_layer"] in provided
    assert set(check["targets"]) <= provided


async def test_target_union_excludes_norms_for_another_service_type(tmp_path):
    third = plan(1, radius=500)
    third["source"].update(
        restriction_id="00000000-0000-4000-8000-000000000003", clause_number="7.3"
    )
    third["declared_requirements"]["layers"][1]["entity"] = "Кинотеатр"
    svc, _ = service(tmp_path)
    mcp = Mcp()
    await run(svc, mcp, plans=[plan(0), plan(1, radius=300), third])
    targets = {
        args["services_names"][0]: args
        for name, args in mcp.calls
        if name == "GetServicesByGeometry" and args["layer_name"].startswith("Цели")
    }
    assert len(targets["Театр"]["zones_layers"]) == 2
    assert len(targets["Кинотеатр"]["zones_layers"]) == 1
    assert not set(targets["Кинотеатр"]["zones_layers"]) & set(
        targets["Театр"]["zones_layers"]
    )


@pytest.mark.parametrize(
    "template,params,expected",
    [
        ("distance_from_source", {"predicate": "within"}, "within"),
        ("distance_from_source", {"predicate": "contains"}, "intersects"),
        ("presence_within", {}, "intersects"),
        ("zonal_attribute_threshold", {"join_predicate": "within"}, "within"),
    ],
)
def test_intersection_derivation(template, params, expected):
    assert (
        intersection_type(SimpleNamespace(template=template, params=params)) == expected
    )


async def test_mixed_predicates_share_one_conservative_target_query(tmp_path):
    svc, _ = service(tmp_path)
    mcp = Mcp()
    await run(svc, mcp, plans=[plan(0, predicate="within"), plan(1)])
    target = [args for name, args in mcp.calls if name == "GetServicesByGeometry"][-1]
    assert target["intersection_type"] == "intersects"
    checks = [args for name, args in mcp.calls if name.startswith("Check")]
    assert [args["predicate"] for args in checks] == ["within", "intersects"]


async def test_per_norm_layers_order_progress_and_llm_input(tmp_path):
    svc, _ = service(tmp_path)
    events = await run(svc, Mcp())
    for event in events:
        RestrictionsResponse.model_validate(event)
        if event["type"] == "feature_collection":
            serialized = RestrictionsResponse.model_validate(event).model_dump(
                mode="json"
            )
            assert (
                serialized["content"]["feature_collection"]["meta"]["group_by"]
                == "norm_id"
            )
    types = [event["type"] for event in events]
    assert max(i for i, value in enumerate(types) if value == "chunk") < types.index(
        "feature_collection"
    )
    assert types[-2:] == ["file", "file"]
    assert events[-2]["content"]["filename"] == "summary.md"
    assert events[-1]["content"]["filename"].endswith(".zip")
    layers = [
        event["content"] for event in events if event["type"] == "feature_collection"
    ]
    assert len(layers) == 2 and all(
        layer["name"].startswith("Норма — ") for layer in layers
    )
    for layer in layers:
        collection = layer["feature_collection"]
        assert collection["meta"]["group_by"] == "norm_id"
        assert {f["properties"]["feature_kind"] for f in collection["features"]} == {
            "zone",
            "object",
            "violation",
        }
        assert all(
            f["properties"]["norm_text"] == plan(0)["source"]["extraction_text"]
            for f in collection["features"]
        )
        assert all(
            f["geometry"]["type"] != "GeometryCollection"
            for f in collection["features"]
        )
    progresses = [e["content"] for e in events if e["type"] == "progress"]
    assert [p["percent"] for p in progresses] == sorted(
        p["percent"] for p in progresses
    )
    assert progresses[-1]["percent"] == 100
    assert all(
        set(p) == {"stage", "stage_ru", "percent", "current", "total", "label", "bar"}
        for p in progresses
    )
    assert all(p["bar"].count("▰") + p["bar"].count("▱") == 10 for p in progresses)
    llm_data = json.loads(svc.llm_client.messages[-1]["content"])
    assert llm_data["query"] == "Проверь проект"
    assert "layers" not in llm_data["summary"] and "evidence" not in llm_data["summary"]
    assert len(llm_data["summary"]["norms"]) == 2


async def test_archive_layout_manifest_and_checkpoint(tmp_path):
    svc, checkpoints = service(tmp_path)
    mcp = Mcp()
    events = await run(svc, mcp)
    archive = ComplianceArchive("run-1", 772)
    archive_file = next(archive.root.glob("*.zip"))
    with ZipFile(archive_file) as zipped:
        names = set(zipped.namelist())
        assert {
            "README.md",
            "summary.md",
            "summary.json",
            "manifest.json",
            "layers.geojson",
        } <= names
        assert "inputs.json" not in names and "archive.json" not in names
        manifest = json.loads(zipped.read("manifest.json"))
        assert set(manifest["files"]) == names - {"manifest.json"}
        for name, checksum in manifest["files"].items():
            assert hashlib.sha256(zipped.read(name)).hexdigest() == checksum
        for rid in IDS:
            prefix = f"violated/{rid}/"
            assert {
                prefix + name for name in ("report.md", "result.json", "result.geojson")
            } <= names
            record = json.loads(zipped.read(prefix + "result.json"))
            assert record["check_plan"]["source"]["restriction_id"] == rid
            assert (
                record["input_layers"]
                and "timings_ms" in record
                and "input_revision" in record
            )
            assert any(name.startswith(prefix + "layers/source__") for name in names)
            assert any(name.startswith(prefix + "layers/target__") for name in names)
    first_bytes = archive_file.read_bytes()
    assert await run(svc, mcp, checkpoint=checkpoints) == []
    assert archive_file.read_bytes() == first_bytes
    # The same archive checkpoint never rebuilds the zip.
    archive.finalize({"request_id": "run-1", "results": []}, {})
    assert archive_file.read_bytes() == first_bytes
    descriptor = events[-1]["content"]
    file_id = descriptor["url"].rsplit("/", 1)[-1]
    assert svc.file_store.metadata("compliance_archive", file_id)["owner"] == "author"


async def test_inventory_emits_only_not_checked_zones_and_no_target_queries(tmp_path):
    svc, _ = service(tmp_path)
    mcp = Mcp()
    events = await run(svc, mcp, inventory=True)
    assert not any(name.startswith("Check") for name, _ in mcp.calls)
    queries = [args for name, args in mcp.calls if name == "GetServicesByGeometry"]
    assert len(queries) == 1 and queries[0]["buffer_m"] == 300
    layers = [
        e["content"]["feature_collection"]
        for e in events
        if e["type"] == "feature_collection"
    ]
    assert len(layers) == 2
    assert all(
        f["properties"]["status"] == "not_checked"
        and f["properties"]["status_ru"] == "не проверялось"
        for layer in layers
        for f in layer["features"]
    )
    for name, args in mcp.calls:
        if name == "CreateRestrictionZones":
            assert args["properties"]["template"] == "distance_from_source"
            assert (
                args["properties"]["provenance"]["extraction_text"]
                == plan(0)["source"]["extraction_text"]
            )


async def test_llm_failure_uses_deterministic_fallback(tmp_path):
    svc, _ = service(tmp_path, failure=True)
    events = await run(svc, Mcp())
    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    text = "".join(e["content"]["text"] for e in events if e["type"] == "chunk")
    assert text == svc._compliance_summary_text(summary)
    assert "Не выполнено: 2" in text


def test_progress_style_b():
    event = progress_event(
        "template_execution", 12, 30, "Проверка норм 12/30", minimum=40
    )
    assert event["content"]["bar"].endswith("· Проверка норм 12/30")
    assert set(STAGES) >= {"source_layers", "zones_building", "archive", "llm_summary"}


def test_source_radius_uses_max_band_and_accessibility_envelope():
    assert (
        source_radius(
            SimpleNamespace(template="distance_table"),
            SimpleNamespace(
                bands=[SimpleNamespace(distance_m=50), SimpleNamespace(distance_m=300)]
            ),
        )
        == 300
    )
    assert (
        source_radius(
            SimpleNamespace(template="accessibility_within"),
            SimpleNamespace(radius_m=lambda: 5000),
        )
        == 5000
    )


async def test_equivalent_norm_has_only_the_executed_folder(tmp_path):
    svc, _ = service(tmp_path)
    mcp = Mcp()
    events = await run(svc, mcp, plans=[plan(0), plan(1)])
    assert sum(name.startswith("Check") for name, _ in mcp.calls) == 1
    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    equivalent = next(item for item in summary["results"] if item.get("equivalent_to"))
    assert equivalent["status"] == "violated"
    with ZipFile(next(ComplianceArchive("run-1", 772).root.glob("*.zip"))) as archive:
        assert not any(f"/{IDS[1]}/" in name for name in archive.namelist())
        record = json.loads(archive.read(f"violated/{IDS[0]}/result.json"))
        assert len(record["equivalent_sources"]) == 2


async def test_one_failed_norm_does_not_erase_the_next_check(tmp_path):
    class FailingMcp(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name.startswith("Check") and arguments["restriction_id"] == IDS[0]:
                raise RuntimeError("Проверка недоступна")
            return await super().execute_tool(name, arguments, meta)

    svc, _ = service(tmp_path)
    events = await run(svc, FailingMcp())
    results = [e["content"] for e in events if e["type"] == "compliance_result"]
    assert [r["status"] for r in results] == ["not_checked", "violated"]
    assert results[0]["archive_path"] is None
    assert results[1]["archive_path"].startswith("violated/")
    assert events[-1]["content"]["mime_type"] == "application/zip"


async def test_passed_and_partial_norms_also_have_one_layer(tmp_path):
    class PassingMcp(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            response = await super().execute_tool(name, arguments, meta)
            if name.startswith("Check"):
                partial = arguments["restriction_id"] == IDS[1]
                response["summary"] = {"violated_objects": 0, "passed_objects": 1}
                if partial:
                    response["coverage"].update(
                        applicable_objects=2, unchecked_objects=1, fill_rate=0.5
                    )
                layer = response["norm_layer"]
                layer["meta"].update(
                    norm_status="partial" if partial else "passed",
                    norm_status_reason="Не хватает атрибутов" if partial else None,
                )
                layer["features"] = [
                    f
                    for f in layer["features"]
                    if f["properties"]["feature_kind"] != "violation"
                ]
            return response

    svc, _ = service(tmp_path)
    events = await run(svc, PassingMcp())
    layers = [
        e["content"]["feature_collection"]
        for e in events
        if e["type"] == "feature_collection"
    ]
    assert [layer["meta"]["norm_status"] for layer in layers] == ["passed", "partial"]
    assert len(layers) == 2


async def test_incomplete_delivery_reuses_inputs_results_and_zip(tmp_path):
    svc, checkpoints = service(tmp_path)
    mcp = Mcp()
    await run(svc, mcp)
    checkpoints.pop("compliance_completed")
    count = len(mcp.calls)
    archive = next(ComplianceArchive("run-1", 772).root.glob("*.zip"))
    before = archive.read_bytes()
    events = await run(svc, mcp, checkpoint=checkpoints)
    assert len(mcp.calls) == count
    assert archive.read_bytes() == before
    assert events[-1]["content"]["mime_type"] == "application/zip"


def test_a2a_archive_and_summary_use_distinct_artifacts():
    store = A2ATaskStore()
    store.create_task("task", "ctx", {}, {})
    executor = ComplianceAgentExecutor(object(), store)
    summary = executor._pipeline_item_to_event(
        "task",
        "ctx",
        {
            "type": "file",
            "content": {"filename": "summary.md", "mime_type": "text/markdown"},
        },
        0,
    )
    archive = executor._pipeline_item_to_event(
        "task",
        "ctx",
        {
            "type": "file",
            "content": {"filename": "result.zip", "mime_type": "application/zip"},
        },
        0,
    )
    assert summary["artifact"]["artifactId"] == "compliance-file"
    assert archive["artifact"]["artifactId"] == "compliance-archive"
