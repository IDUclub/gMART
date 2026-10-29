"""A compliance check of uploaded files, end to end with in-process geometry checks."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_entity_mapping import EntityMapper
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_inventory import RestrictionZoneBuilder
from src.agents.services.compilance.compliance_scope import (
    ComplianceScope,
    ScopeOutcome,
)
from src.agents.services.compilance.compliance_territory import TerritoryDocuments
from src.agents.services.compilance.compliance_uploads import ComplianceUploadService
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStep
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools

OWNER_TOKEN = "e30." + "eyJzdWIiOiJhbGljZSJ9" + ".sig"  # {} header  # {"sub": "alice"}


def square(x, y, size=0.0004):
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]
        ],
    }


def geojson(features):
    return json.dumps(
        {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        },
        ensure_ascii=False,
    ).encode("utf-8")


def feature(properties, geometry, feature_id):
    return {
        "type": "Feature",
        "id": feature_id,
        "geometry": geometry,
        "properties": properties,
    }


def design_files(extra_project=()):
    project = [
        # Next to the school in the surroundings: within 100 m.
        feature(
            {"type": "buildings", "building": "residential", "levels": "9"},
            square(37.4800, 55.6700),
            "near",
        ),
        # Far from it.
        feature(
            {"type": "buildings", "building": "residential", "levels": "5"},
            square(37.4900, 55.6760),
            "far",
        ),
        *extra_project,
    ]
    surroundings = [
        feature(
            {"type": "buildings", "building": "school", "amenity": "school"},
            square(37.4806, 55.6700),
            "school",
        ),
    ]
    border = [feature({"type": "border"}, square(37.479, 55.669, 0.012), "b")]
    return [
        ("336195_map_objects.geojson", geojson(project)),
        ("336195_osm_surroundings.geojson", geojson(surroundings)),
        ("336195_border.geojson", geojson(border)),
    ]


def distance_plan(restriction_id, target_entity, target_type="physical_object"):
    return {
        "id": restriction_id,
        "provenance": {"name": "СП 42.13330.2016", "numbering": "7.1"},
        "check_plan": {
            "schema_version": "1.0",
            "template": "distance_from_source",
            "template_version": 1,
            "params": {
                "source_layer": "schools",
                "targets": ["targets"],
                "geometry_mode": "buffered",
                "predicate": "intersects",
                "violation_when": "matched",
                "result_mode": "both",
                "distance_m": 100,
            },
            "declared_requirements": {
                "layers": [
                    {
                        "role": "schools",
                        "entity": "Школа",
                        "entity_type": "service",
                        "geometry_types": ["Polygon", "MultiPolygon"],
                        "required": True,
                    },
                    {
                        "role": "targets",
                        "entity": target_entity,
                        "entity_type": target_type,
                        "geometry_types": ["Polygon", "MultiPolygon"],
                        "required": True,
                    },
                ],
                "attributes": [],
            },
            "source": {
                "restriction_id": restriction_id,
                "document_name": "СП 42.13330.2016",
                "clause_number": "7.1",
                "extraction_text": f"Требование {restriction_id}",
            },
            "planner_status": "auto",
        },
    }


class InProcessMcp:
    """Urban catalogue lookups and IDU MCP Check* tools without a server."""

    def __init__(self):
        self.geometry = ComplianceGeometryTools()
        self.calls = []

    async def resolve_urban_entity_types(self, *, service_names, physical_object_names):
        def found(names):
            return {name: {"found": True, "canonical_name": name} for name in names}

        return {
            "service": found(service_names),
            "physical_object": found(physical_object_names),
        }

    async def execute_tool(self, name, arguments, meta=None):
        self.calls.append(name)
        if name == "CheckDistanceFromSource":
            return self.geometry.distance_from_source(**arguments)
        if name == "CreateRestrictionZones":
            zones = self.geometry.restriction_zones(
                **{
                    key: value
                    for key, value in arguments.items()
                    if key != "layer_name"
                }
            )
            norm_layer = zones.pop("norm_layer")
            return {arguments["layer_name"]: zones, "norm_layer": norm_layer}
        raise AssertionError(f"unexpected tool {name}")


class FakeLlm:
    def __init__(self, answers=()):
        self.answers = list(answers)

    async def chat(self, **kwargs):
        return {
            "message": {"content": json.dumps(self.answers.pop(0), ensure_ascii=False)}
        }


@pytest.fixture
async def store():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield PipelineStateStore(redis)
    await redis.aclose()


def service_for(store, llm=None):
    service = object.__new__(RestrictionParserService)
    service.state_store = store
    service.file_store = None
    service.compliance_executor = ComplianceTemplateExecutor(
        effects_client_factory=None
    )
    service.zone_builder = RestrictionZoneBuilder(service.compliance_executor)
    service.compliance_uploads = ComplianceUploadService(store)
    service.entity_mapper = EntityMapper(llm or FakeLlm())
    return service


async def run_check(service, files, restrictions, *, scope_key="chat-1"):
    await service.state_store.create(
        "req", chat_id=None, user_query="q", scenario_id=None, model="m", temperature=0
    )
    return [
        event
        async for event in service._run_executable_compliance(
            mcp_client=InProcessMcp(),
            request_id="req",
            scenario_id=None,
            restrictions=restrictions,
            checkpoint={},
            owner="alice",
            territory=TerritoryDocuments(
                status="ok", allowed=("СП 42.13330.2016",), place="files"
            ),
            user_files=files,
            model="m",
            scope_key=scope_key,
            user_query="Проверь мой проект",
        )
    ]


async def test_design_objects_are_checked_against_the_surroundings(store):
    service = service_for(store)
    upload = await service.compliance_uploads.create("alice", design_files())
    files = await service.compliance_uploads.load([upload["upload_id"]], "alice")

    events = await run_check(
        service,
        files,
        [
            distance_plan("r1", "Жилой дом"),
            distance_plan("r2", "Детский сад", "service"),
        ],
    )

    for event in events:  # every event survives the SSE schema
        RestrictionsResponse.model_validate(event)
    (result,) = [e["content"] for e in events if e["type"] == "compliance_result"]
    assert result["restriction_id"] == "r1"
    assert result["compliance_status"] == "violated"
    assert result["summary"] == {"violated_objects": 1, "passed_objects": 1}
    layers = {
        e["content"]["name"]: e["content"]["feature_collection"]
        for e in events
        if e["type"] == "feature_collection"
    }
    norm_features = layers["Норма — СП 42.13330.2016, п. 7.1"]["features"]
    violated = [
        f for f in norm_features if f["properties"]["feature_kind"] == "violation"
    ]
    assert [f["properties"]["object_ref"]["entity_id"] for f in violated] == [
        "user/336195_map_objects/near"
    ]
    assert all(f["properties"]["norm_id"] == "r1" for f in norm_features)
    assert all(name.startswith("Норма — ") for name in layers)
    # Calls over uploaded layers are not replayable from the chat history.
    assert not [e for e in events if e["type"] == "tool_call"]

    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    user_files = summary["user_files"]
    assert user_files["mode"] == "standalone"
    assert user_files["entities"]["Жилой дом"] == {"project": 2, "context": 0}
    (not_checked,) = user_files["not_checked"]
    assert not_checked["restriction_id"] == "r2"
    assert "Детский сад" in not_checked["reasons"][0]
    text = "".join(e["content"]["text"] for e in events if e["type"] == "chunk")
    assert "Проверены объекты загруженных файлов без сценария" in text
    assert "Не проверено из-за отсутствия нужных объектов в файлах: 1" in text
    mapping_event = next(e for e in events if e["type"] == "user_files")
    assert mapping_event["content"]["pending"] == []


async def test_doubtful_mapping_is_asked_and_the_answer_continues_the_check(store):
    calm_rest = feature({"type": "calm_rest"}, square(37.4801, 55.6701), "calm")
    llm = FakeLlm(
        [{"items": [{"id": 1, "entity": "Детская площадка", "confidence": 0.5}]}]
    )
    service = service_for(store, llm)
    upload = await service.compliance_uploads.create("alice", design_files([calm_rest]))
    files = await service.compliance_uploads.load([upload["upload_id"]], "alice")

    events = await run_check(
        service, files, [distance_plan("r3", "Детская площадка", "service")]
    )

    clarification = next(e for e in events if e["type"] == "clarification")
    serialized = RestrictionsResponse.model_validate(clarification).model_dump()
    assert serialized["content"]["kind"] == "entity_mapping"
    assert serialized["content"]["options"][0]["value"] == "Детская площадка"
    assert not [e for e in events if e["type"] == "compliance_result"]
    pending = await store.get_compliance_mapping("chat-1")
    assert pending["upload_ids"] == [upload["upload_id"]]
    assert pending["user_query"] == "Проверь мой проект"
    checkpoint = await store.get_checkpoint("req")
    assert checkpoint[PipelineStep.COMPLIANCE_MAPPING] == {"pending": True}

    # «да» confirms the proposal for the upload; a rerun asks nothing.
    assert await service._answer_mapping("m", "да", pending, OWNER_TOKEN) == "confirmed"
    reloaded = await service.compliance_uploads.load([upload["upload_id"]], "alice")
    assert reloaded.confirmed_mapping == {
        "type=calm_rest": {"entity": "Детская площадка", "entity_type": "service"}
    }
    await store.save_checkpoint("req", PipelineStep.COMPLIANCE_MAPPING, None)
    rerun = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=InProcessMcp(),
            request_id="req2",
            scenario_id=None,
            restrictions=[distance_plan("r3", "Детская площадка", "service")],
            checkpoint={},
            owner="alice",
            user_files=reloaded,
            model="m",
            scope_key="chat-1",
            user_query="Проверь мой проект",
        )
    ]
    (result,) = [e["content"] for e in rerun if e["type"] == "compliance_result"]
    assert result["compliance_status"] == "violated"


def pipeline_service(store, pending=None):
    service = service_for(store)
    service.compliance_territory = SimpleNamespace(
        resolve=AsyncMock(),
        resolve_geometry=AsyncMock(
            return_value=TerritoryDocuments(
                status="ok", allowed=("СП 42.13330.2016",), place="files"
            )
        ),
    )
    service.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    service.add_single_message = AsyncMock()
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *a: None
    )
    service.compliance_scope = SimpleNamespace(
        resolve=AsyncMock(return_value=ScopeOutcome(kind="scoped")),
        resolve_choice=AsyncMock(),
    )
    service.normgraph_retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=SimpleNamespace(
                restrictions=[{"id": "r1"}],
                unsupported_count=0,
                tool_call={"function": {"name": "list_restrictions", "arguments": {}}},
            )
        )
    )
    service.executed_with = None

    async def run(**kwargs):
        service.executed_with = kwargs
        yield {"type": "chunk", "content": {"text": "Готово.", "done": True}}

    service._run_executable_compliance = run
    return service


async def run_pipeline(service, query, upload_ids):
    return [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query=query,
            scenario_id=None,
            token_ref=[OWNER_TOKEN],
            chat_id="chat-1",
            persist_history=False,
            normgraph_mcp_client=object(),
            history_agent="compliance",
            upload_ids=upload_ids,
        )
    ]


async def test_files_without_a_scenario_locate_documents_by_their_boundary(store):
    service = pipeline_service(store)
    upload = await service.compliance_uploads.create("alice", design_files())

    await run_pipeline(service, "Проверь мой проект", [upload["upload_id"]])

    service.compliance_territory.resolve.assert_not_awaited()
    (geometry,) = service.compliance_territory.resolve_geometry.await_args.args[3:]
    assert geometry["type"] == "Polygon"
    files = service.executed_with["user_files"]
    assert [layer.role for layer in files.layers] == ["project", "context", "border"]
    assert service.executed_with["scenario_id"] is None


async def test_someone_elses_upload_is_not_checked(store):
    service = pipeline_service(store)
    upload = await service.compliance_uploads.create("bob", design_files())

    events = await run_pipeline(service, "Проверь", [upload["upload_id"]])

    assert "Проверка файлов не выполнена" in events[-1]["content"]["text"]
    assert service.executed_with is None


async def test_answer_to_a_pending_mapping_reruns_the_original_request(store):
    service = pipeline_service(store)
    upload = await service.compliance_uploads.create("alice", design_files())
    await store.set_compliance_mapping(
        "chat-1",
        {
            "upload_ids": [upload["upload_id"]],
            "user_query": "Проверь мой проект по СП 42",
            "scope": ComplianceScope(documents=("СП 42.13330.2016",)).to_dict(),
            "scenario_id": None,
            "mapping": {
                "proposals": [
                    {
                        "signature": "type=calm_rest",
                        "label": "type=calm_rest",
                        "count": 1,
                        "entity": "Детская площадка",
                        "entity_type": "service",
                        "confidence": 0.5,
                        "source": "llm",
                    }
                ],
                "pending": ["type=calm_rest"],
            },
            "candidates": [{"entity": "Детская площадка", "entity_type": "service"}],
        },
    )

    await run_pipeline(service, "1 — нет", None)

    assert await store.get_compliance_mapping("chat-1") is None
    assert service.executed_with["user_query"] == "Проверь мой проект по СП 42"
    assert service.executed_with["scope"].documents == ("СП 42.13330.2016",)
    service.compliance_scope.resolve.assert_not_awaited()
    files = service.executed_with["user_files"]
    assert files.confirmed_mapping == {"type=calm_rest": None}


def test_report_explains_norms_the_files_could_not_serve():
    from src.agents.services.compilance.compliance_archive import (
        build_summary_report,
        complete_summary,
    )

    raw = distance_plan("r1", "Детский сад", "service")
    summary = {
        "request_id": "req",
        "results": [],
        "user_files": {
            "layers": [{"name": "map_objects", "role": "project", "feature_count": 3}],
            "mapping": [
                {
                    "label": "type=calm_rest",
                    "count": 2,
                    "entity": "Детская площадка",
                    "source": "llm",
                    "confidence": 0.9,
                }
            ],
            "not_checked": [
                {
                    "restriction_id": "r1",
                    "label": "СП 42.13330.2016, п. 7.1",
                    "missing": ["user_files:no_project_objects:Детский сад"],
                    "reasons": ["В файлах нет объектов Детский сад"],
                }
            ],
        },
    }
    complete_summary(summary, [raw], {})
    report = build_summary_report(summary)
    assert "Не запущено норм: 1." in report
    assert "В файлах нет объектов Детский сад" in report
    assert "## Загруженные файлы" in report
    assert "| map_objects | проект | 3 |" in report
    assert "### Сопоставление с объектами норм" in report
    assert (
        "| type=calm_rest | 2 | Детская площадка | моделью, уверенность 90% |" in report
    )
    assert (
        report.index("## Не запущено")
        < report.index("В файлах нет объектов Детский сад")
        < report.index("## Загруженные файлы")
        < report.index("## Где смотреть подробно")
    )
    assert summary["results"][0]["status"] == "not_checked"
    assert "Полный список находится в summary.json" in report


def test_report_missing_file_entities_are_deduplicated_and_bounded():
    from src.agents.services.compilance.compliance_archive import build_summary_report

    names = [f"Тип {index}" for index in range(12)]
    report = build_summary_report(
        {
            "results": [],
            "user_files": {
                "not_checked": [
                    {
                        "label": "Норма",
                        "missing": [
                            "user_files:no_project_objects:" + ", ".join(names[:6]),
                            *[f"user_files:no_objects:{name}" for name in names[6:]],
                            "user_files:no_objects:Тип 0",
                            "catalog:Свободный текст",
                        ],
                        "reasons": ["Свободное пояснение с объектом Скрытый объект"],
                    }
                ]
            },
        }
    )
    sentence = next(
        line
        for line in report.splitlines()
        if line.startswith("В файлах нет объектов ")
    )
    assert sentence == "В файлах нет объектов " + ", ".join(names[:10]) + " и ещё 2."


async def test_chat_message_without_scenario_files_or_question_checks_nothing(store):
    service = pipeline_service(store)

    events = await run_pipeline(service, "да", None)

    assert "не выбран сценарий и не переданы файлы" in events[-1]["content"]["text"]
    assert service.executed_with is None
    service.compliance_territory.resolve.assert_not_awaited()


async def test_llm_answers_are_kept_so_a_rerun_asks_nothing_new(store):
    calm_rest = feature({"type": "calm_rest"}, square(37.4801, 55.6701), "calm")
    llm = FakeLlm(
        [{"items": [{"id": 1, "entity": "Детская площадка", "confidence": 0.95}]}]
    )
    service = service_for(store, llm)
    upload = await service.compliance_uploads.create("alice", design_files([calm_rest]))
    files = await service.compliance_uploads.load([upload["upload_id"]], "alice")

    await run_check(
        service, files, [distance_plan("r3", "Детская площадка", "service")]
    )

    reloaded = await service.compliance_uploads.load([upload["upload_id"]], "alice")
    (cached,) = reloaded.classified.values()
    assert cached["entity"] == "Детская площадка" and cached["confidence"] == 0.95
    # The second run must not call the LLM: FakeLlm has no answer left.
    rerun = [
        event
        async for event in service._run_executable_compliance(
            mcp_client=InProcessMcp(),
            request_id="req2",
            scenario_id=None,
            restrictions=[distance_plan("r3", "Детская площадка", "service")],
            checkpoint={},
            owner="alice",
            user_files=reloaded,
            model="m",
            scope_key="chat-1",
            user_query="Проверь мой проект",
        )
    ]
    mapping = next(e for e in rerun if e["type"] == "user_files")["content"]["mapping"]
    calm = next(item for item in mapping if item["signature"] == "type=calm_rest")
    assert calm["source"] == "llm" and calm["entity"] == "Детская площадка"
