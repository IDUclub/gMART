"""«Какие ограничения есть на территории»: zones of executable norms, territory filter."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from shapely.geometry import Point, Polygon, mapping, shape

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_inventory import (
    RestrictionZoneBuilder,
    describe_zone,
)
from src.agents.services.compilance.compliance_inventory_report import (
    build_inventory_report,
)
from src.agents.services.compilance.compliance_result_harness import (
    ComplianceResultHarness,
)
from src.agents.services.compilance.compliance_scope import (
    ComplianceScope,
    ComplianceScopeResolver,
    ScopeRequest,
    render_choice,
    scope_for_choice,
)
from src.agents.services.compilance.compliance_territory import (
    ComplianceTerritoryFilter,
    TerritoryDocuments,
)
from src.agents.services.pipeline_state import PipelineStep
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.idu_mcp.tools_services.compliance_geometry import ComplianceGeometryTools

PROJECT = Polygon([(29.99, 59.99), (30.02, 59.99), (30.02, 60.01), (29.99, 60.01)])
RESIDENTIAL_ZONE = Polygon([(30.0, 60.0), (30.05, 60.0), (30.05, 60.02), (30.0, 60.02)])


def _fc(*features):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": str(index),
                "geometry": mapping(geometry),
                "properties": properties,
            }
            for index, (geometry, properties) in enumerate(features)
        ],
    }


def _distance_plan(**params):
    return {
        "schema_version": "1.0",
        "template": "distance_from_source",
        "template_version": 1,
        "params": {
            "source_layer": "source",
            "targets": ["targets"],
            "geometry_mode": "buffered",
            "distance_m": 50,
            "predicate": "intersects",
            "violation_when": "matched",
            "result_mode": "both",
            **params,
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
                    "role": "targets",
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                    "geometry_types": ["Polygon"],
                },
            ],
            "attributes": [],
        },
        "source": {
            "restriction_id": "r-school",
            "document_name": "СП 42.13330.2016",
            "clause_number": "10.4",
            "extraction_text": "Расстояние от школы до жилого дома не менее 50 м.",
        },
        "planner_status": "auto",
    }


def _zonal_plan(zones_entity="Жилая зона", threshold=None):
    return {
        "schema_version": "1.0",
        "template": "zonal_attribute_threshold",
        "template_version": 1,
        "params": {
            "objects_layer": "objects",
            "zones_layer": "zones",
            "attribute_role": "floors",
            "operator": "<=",
            "threshold_source": threshold
            or {"kind": "constant", "value": 9, "unit": "floors"},
        },
        "declared_requirements": {
            "layers": [
                {
                    "role": "objects",
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                },
                {
                    "role": "zones",
                    "entity": zones_entity,
                    "entity_type": "functional_zone",
                    "geometry_types": ["Polygon", "MultiPolygon"],
                },
            ],
            "attributes": [
                {
                    "role": "floors",
                    "on": "objects",
                    "accepts": [
                        {"field": "floors", "unit": "floors", "quality": "direct"}
                    ],
                },
                *(
                    [
                        {
                            "role": "zone_floors",
                            "on": "zones",
                            "accepts": [
                                {
                                    "field": "max_floors",
                                    "unit": "floors",
                                    "quality": "direct",
                                }
                            ],
                        }
                    ]
                    if threshold
                    else []
                ),
            ],
        },
        "source": {
            "restriction_id": "r-height",
            "document_name": "ПЗЗ",
            "clause_number": "3",
            "extraction_text": "Высота жилых домов — не более 9 этажей.",
        },
        "planner_status": "reviewed",
    }


class FakeMcp:
    """Scenario with two schools and a residential zone; no residential houses."""

    def __init__(self, *, schools=True):
        self.calls = []
        self.schools = schools

    async def resolve_urban_entity_types(self, *, service_names, physical_object_names):
        return {
            "service": {
                name: {"found": True, "canonical_name": name, "type_id": 1}
                for name in service_names
            },
            "physical_object": {
                name: {"found": True, "canonical_name": name, "type_id": 2}
                for name in physical_object_names
            },
        }

    async def execute_tool(self, name, arguments, meta=None):
        self.calls.append((name, arguments))
        if name == "GetServices":
            features = (
                [
                    (Point(30.0, 60.0), {"service_id": 1, "name": "Школа № 1"}),
                    (Point(30.01, 60.0), {"service_id": 2, "name": "Школа № 2"}),
                ]
                if self.schools
                else []
            )
            return {"Школа": {**_fc(*features), "meta": {"complete": True}}}
        if name == "GetPhysicalObjects":
            raise AssertionError("inventory must not load the targets")
        if name == "GetFunctionalZones":
            wanted = arguments.get("zone_type_names")
            zones = (
                [
                    (
                        RESIDENTIAL_ZONE,
                        {
                            "functional_zone_type": {"name": "Жилая зона"},
                            "max_floors": 12,
                        },
                    )
                ]
                if wanted is None or "Жилая зона" in wanted
                else []
            )
            return {"functional_zones": {**_fc(*zones), "meta": {"complete": True}}}
        if name == "GetProjectTerritory":
            return {"project_territory": _fc((PROJECT, {"name": "Территория проекта"}))}
        if name == "CreateRestrictionZones":
            zones = ComplianceGeometryTools().restriction_zones(
                **{
                    key: value
                    for key, value in arguments.items()
                    if key != "layer_name"
                }
            )
            return {arguments["layer_name"]: zones}
        raise AssertionError(name)


def _builder():
    return RestrictionZoneBuilder(ComplianceTemplateExecutor())


# --- IDU MCP geometry ------------------------------------------------------------


def test_buffer_zones_keep_the_source_and_carry_the_norm():
    layers = {"Школа": _fc((Point(30.0, 60.0), {"service_id": 1, "name": "Школа"}))}

    result = ComplianceGeometryTools().restriction_zones(
        geometry_mode="buffer",
        source_layer="Школа",
        layers=layers,
        distance_m=50,
        properties={"restriction_title": "Зона", "zone_kind": "restriction"},
    )

    [feature] = result["features"]
    assert feature["properties"]["name"] == "Школа"
    assert feature["properties"]["zone_kind"] == "restriction"
    assert feature["properties"]["buffer_size"] == 50
    assert shape(feature["geometry"]).contains(Point(30.0003, 60.0))
    assert not shape(feature["geometry"]).contains(Point(30.002, 60.0))
    assert result["meta"]["zones"] == 1


def test_attribute_buffer_skips_sources_without_a_matching_value():
    layers = {
        "АЗС": _fc(
            (Point(30.0, 60.0), {"capacity": 5}),
            (Point(30.1, 60.0), {"capacity": None}),
        )
    }

    result = ComplianceGeometryTools().restriction_zones(
        geometry_mode="attribute_buffer",
        source_layer="АЗС",
        layers=layers,
        attribute_field="capacity",
        bands=[{"min": 0, "max": 10, "distance_m": 100}],
    )

    assert [f["properties"]["buffer_size"] for f in result["features"]] == [100]
    assert result["meta"]["skipped_objects"] == 1


def test_geometry_zones_are_clipped_and_take_their_own_threshold():
    layers = {
        "functional_zones": _fc(
            (RESIDENTIAL_ZONE, {"max_floors": 12}),
            (Polygon([(29.0, 59.0), (29.1, 59.0), (29.1, 59.1)]), {"max_floors": None}),
        ),
        "project_territory": _fc((PROJECT, {})),
    }

    result = ComplianceGeometryTools().restriction_zones(
        geometry_mode="geometry",
        source_layer="functional_zones",
        layers=layers,
        threshold_field="max_floors",
        clip_layer="project_territory",
        properties={"threshold": 1, "operator": "<="},
    )

    [feature] = result["features"]
    assert feature["properties"]["threshold"] == 12
    assert PROJECT.buffer(1e-9).contains(shape(feature["geometry"]))
    assert result["meta"]["skipped_objects"] == 1


# --- zone builder ----------------------------------------------------------------


async def test_distance_norm_draws_buffers_around_sources_without_targets():
    mcp = FakeMcp()

    zone = await _builder().build(mcp, _distance_plan(), 772)

    assert zone.status == "shown"
    assert zone.zone_kind == "restriction"
    assert zone.zone_count == 2
    assert zone.description == {
        "around": "Школа",
        "distance_m": 50.0,
        "applies_to": ["Жилой дом"],
    }
    assert [call["function"]["name"] for call in zone.tool_calls] == [
        "GetServices",
        "CreateRestrictionZones",
    ]
    stored = zone.tool_calls[-1]["function"]["arguments"]
    assert "layers" not in stored
    assert stored["geometry_mode"] == "buffer" and stored["distance_m"] == 50
    feature = zone.zones["features"][0]["properties"]
    assert (
        feature["restriction_title"] == "Зона ограничения — СП 42.13330.2016, п. 10.4"
    )
    assert feature["applies_to"] == ["Жилой дом"]


async def test_unmatched_distance_norm_is_a_required_zone():
    zone = await _builder().build(
        FakeMcp(), _distance_plan(violation_when="not_matched"), 772
    )

    assert zone.zone_kind == "required"
    assert "должны располагаться «Жилой дом»" in describe_zone(zone.payload())


async def test_norm_without_sources_in_the_scenario_is_not_drawn():
    mcp = FakeMcp(schools=False)

    zone = await _builder().build(mcp, _distance_plan(), 772)

    assert zone.status == "no_objects"
    assert zone.zone_count == 0
    assert "CreateRestrictionZones" not in [name for name, _ in mcp.calls]


async def test_zonal_norm_for_every_zone_covers_the_project_territory():
    mcp = FakeMcp()
    territory = _fc((PROJECT, {"name": "Территория проекта"}))

    zone = await _builder().build(
        mcp, _zonal_plan(zones_entity="functional_zones"), 772, territory
    )

    assert zone.status == "shown"
    assert [name for name, _ in mcp.calls] == ["CreateRestrictionZones"]
    properties = zone.zones["features"][0]["properties"]
    assert properties["threshold"] == 9 and properties["operator"] == "<="
    assert describe_zone(zone.payload()) == (
        "вся территория проекта; для «Жилой дом»: ≤ 9 floors"
    )


async def test_zonal_norm_for_every_zone_needs_the_project_boundary():
    zone = await _builder().build(
        FakeMcp(), _zonal_plan(zones_entity="functional_zones"), 772, None
    )

    assert zone.status == "unverifiable"
    assert zone.missing_requirements == ["layer:project_territory"]


async def test_zonal_norm_draws_its_zones_clipped_to_the_project():
    mcp = FakeMcp()
    territory = _fc((PROJECT, {}))

    zone = await _builder().build(
        mcp,
        _zonal_plan(threshold={"kind": "attribute_role", "role": "zone_floors"}),
        772,
        territory,
    )

    assert zone.status == "shown"
    arguments = zone.tool_calls[-1]["function"]["arguments"]
    # Replay feeds GetFunctionalZones' own key to the zone builder.
    assert arguments["source_layer"] == "functional_zones"
    assert arguments["clip_layer"] == "project_territory"
    assert arguments["threshold_field"] == "max_floors"
    [feature] = zone.zones["features"]
    assert feature["properties"]["threshold"] == 12
    assert PROJECT.buffer(1e-9).contains(shape(feature["geometry"]))


# --- territory filter -------------------------------------------------------------


async def test_territory_keeps_graph_documents_in_force_by_id_or_name():
    dvd = SimpleNamespace(
        list_documents=AsyncMock(
            return_value=[
                {"doc_id": "d1", "name": "Другое имя"},
                {"doc_id": "x", "name": "ПЗЗ города"},
            ]
        )
    )
    graph = SimpleNamespace(
        list_restriction_documents=AsyncMock(
            return_value=[
                {"doc_id": "d1", "name": "СП 42.13330.2016"},
                {"doc_id": "d2", "name": "пзз  города"},
                {"doc_id": "d3", "name": "ПЗЗ другого города"},
            ]
        )
    )

    result = await ComplianceTerritoryFilter().resolve(dvd, graph, 772)

    assert result.status == "ok"
    assert result.allowed == ("СП 42.13330.2016", "пзз  города")
    assert result.excluded == ("ПЗЗ другого города",)
    dvd.list_documents.assert_awaited_once_with(scenario_id=772)


async def test_territory_is_strict_without_idu_dvd():
    graph = SimpleNamespace(list_restriction_documents=AsyncMock())

    missing = await ComplianceTerritoryFilter().resolve(None, graph, 772)
    failing = await ComplianceTerritoryFilter().resolve(
        SimpleNamespace(list_documents=AsyncMock(side_effect=RuntimeError("down"))),
        graph,
        772,
    )

    assert missing.status == failing.status == "unavailable"
    assert "не подключён" in missing.message
    assert "down" in failing.message
    graph.list_restriction_documents.assert_not_awaited()


async def test_territory_without_documents_in_force_stops():
    dvd = SimpleNamespace(list_documents=AsyncMock(return_value=[]))
    graph = SimpleNamespace(
        list_restriction_documents=AsyncMock(return_value=[{"name": "СП 42"}])
    )

    result = await ComplianceTerritoryFilter().resolve(dvd, graph, 772)

    assert result.status == "empty"
    assert TerritoryDocuments.from_dict(result.to_dict()) == result


# --- scope -------------------------------------------------------------------------


def test_scope_filters_documents_by_territory():
    allowed = ("СП 42.13330.2016", "ПЗЗ")

    assert ComplianceScope(allowed_documents=allowed).filters() == {
        "document_names": list(allowed)
    }
    assert ComplianceScope(
        documents=("СП 42.13330.2016",), allowed_documents=allowed
    ).filters() == {"document_names": ["СП 42.13330.2016"]}
    # A chosen document out of force narrows to nothing, not to the whole corpus.
    assert ComplianceScope(
        documents=("СанПиН",), allowed_documents=allowed
    ).filters() == {"document_names": []}
    assert ComplianceScope().filters() == {}


def test_scope_mode_survives_checkpoint_and_choice():
    scope = ComplianceScope(topics=("школа",), mode="inventory")

    assert ComplianceScope.from_dict(scope.to_dict()) == scope
    choice = {"mode": "inventory", "topics": ["школа"], "candidates": []}
    assert scope_for_choice(choice, ("СП 42",)).is_inventory
    assert "ограничения из которого показать" in render_choice(
        {**choice, "matched": True, "references": ["СП"]}
    )


async def test_resolver_reads_the_inventory_mode():
    resolver = ComplianceScopeResolver(llm_client=None)
    resolver._extract = AsyncMock(return_value=ScopeRequest(mode="inventory"))

    outcome = await resolver.resolve(
        SimpleNamespace(), "m", "Какие ограничения есть на территории проекта?"
    )

    assert outcome.kind == "scoped"
    assert outcome.scope.is_inventory
    assert not outcome.scope.is_filtered


async def test_named_document_out_of_force_is_refused():
    resolver = ComplianceScopeResolver(llm_client=None)
    resolver._extract = AsyncMock(return_value=ScopeRequest(mode="inventory"))
    graph = SimpleNamespace(
        list_restriction_documents=AsyncMock(
            return_value=[{"name": "СП 42.13330.2016", "executable_count": 3}]
        )
    )

    outcome = await resolver.resolve(
        graph,
        "m",
        "Какие ограничения по СП 42.13330 есть?",
        allowed_documents=("ПЗЗ",),
    )

    assert outcome.kind == "empty"
    assert outcome.message.startswith("Перечень ограничений не составлен")
    assert "не действует на территории сценария" in outcome.message


def test_inventory_question_is_not_answered_from_the_last_check():
    harness = ComplianceResultHarness()
    assert harness._EXPLICIT_RERUN.search("Какие ограничения есть на территории?")
    assert harness._EXPLICIT_RERUN.search("покажи зоны ограничений")
    assert not harness._EXPLICIT_RERUN.search("Какие ограничения нарушены?")


# --- pipeline ----------------------------------------------------------------------


def _inventory_service(mcp):
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        buffer_event=AsyncMock(), save_checkpoint=AsyncMock(), set_status=AsyncMock()
    )
    service.compliance_executor = ComplianceTemplateExecutor()
    service.zone_builder = RestrictionZoneBuilder(service.compliance_executor)
    return service


async def test_inventory_emits_only_drawn_zones_and_a_summary():
    mcp = FakeMcp()
    service = _inventory_service(mcp)
    second = deepcopy(_zonal_plan(zones_entity="Промышленная зона"))
    second["source"]["restriction_id"] = "r-industrial"
    territory = TerritoryDocuments(
        status="ok", allowed=("СП 42.13330.2016", "ПЗЗ"), excluded=("СанПиН",)
    )

    events = [
        event
        async for event in service._run_restriction_inventory(
            mcp_client=mcp,
            request_id="request-1",
            scenario_id=772,
            restrictions=[
                {"id": "r-school", "check_plan": _distance_plan()},
                {"id": "r-industrial", "check_plan": second},
                {"id": "r-text"},
            ],
            checkpoint={},
            territory=territory,
            scope=ComplianceScope(mode="inventory"),
        )
    ]

    for event in events:
        RestrictionsResponse.model_validate(event)
    layers = [e["content"]["name"] for e in events if e["type"] == "feature_collection"]
    # The industrial zone is absent from the scenario: no empty layer.
    assert layers == ["Зона ограничения — СП 42.13330.2016, п. 10.4"]
    zones = [e["content"] for e in events if e["type"] == "restriction_zone"]
    assert [zone["status"] for zone in zones] == ["shown", "no_objects"]
    summary = next(e["content"] for e in events if e["type"] == "restriction_inventory")
    assert summary["shown_norms"] == 1 and summary["no_objects_norms"] == 1
    assert summary["skipped_without_plan"] == 1
    assert summary["territory"] == {
        "documents_in_force": 2,
        "documents_out_of_force": 1,
    }
    text = next(e["content"]["text"] for e in events if e["type"] == "chunk")
    assert "Показано на карте: 1" in text
    assert "50 м вокруг объектов «Школа»" in text
    assert "не действуют на ней: 1" in text
    replayed = [
        call["function"]["name"]
        for e in events
        if e["type"] == "tool_call"
        for call in e["content"]["tool_calls"]
    ]
    assert replayed == ["GetProjectTerritory", "GetServices", "CreateRestrictionZones"]
    steps = [
        call.args[1] for call in service.state_store.save_checkpoint.await_args_list
    ]
    assert steps == [
        PipelineStep.CHECK_PLAN_VALIDATION,
        PipelineStep.RESTRICTION_INVENTORY,
    ]


async def test_completed_inventory_is_not_rebuilt_on_reconnect():
    service = _inventory_service(FakeMcp())
    service.zone_builder = SimpleNamespace(build=AsyncMock())

    events = [
        event
        async for event in service._run_restriction_inventory(
            mcp_client=object(),
            request_id="request-1",
            scenario_id=772,
            restrictions=[],
            checkpoint={PipelineStep.RESTRICTION_INVENTORY: {}},
        )
    ]

    assert events == []
    service.zone_builder.build.assert_not_awaited()


@pytest.mark.parametrize("status", ["unavailable", "empty"])
async def test_pipeline_stops_when_the_territory_gives_no_documents(status):
    service = object.__new__(RestrictionParserService)
    service.state_store = SimpleNamespace(
        new_request_id=lambda: "request-1",
        create=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        save_checkpoint=AsyncMock(),
        buffer_event=AsyncMock(),
        set_status=AsyncMock(),
    )
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *args: None
    )
    service.compliance_territory = SimpleNamespace(
        resolve=AsyncMock(
            return_value=TerritoryDocuments(status=status, message="Нет документов.")
        )
    )
    service.compliance_scope = SimpleNamespace(resolve=AsyncMock())
    dvd = object()

    events = [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query="Какие ограничения есть на территории проекта?",
            scenario_id=772,
            token_ref=["token"],
            persist_history=False,
            normgraph_mcp_client=object(),
            history_agent="compliance",
            dvd_mcp_client=dvd,
        )
    ]

    assert events[-1]["content"] == {"text": "Нет документов.", "done": True}
    assert service.compliance_territory.resolve.await_args.args[0] is dvd
    service.compliance_scope.resolve.assert_not_awaited()


# --- report ------------------------------------------------------------------------


def test_inventory_report_lists_drawn_and_hidden_zones():
    zone = {
        "status": "shown",
        "zone_kind": "restriction",
        "template": "distance_from_source",
        "zone_count": 2,
        "description": {"around": "Школа", "distance_m": 50, "applies_to": ["Дом"]},
        "source": {
            "document_name": "СП 42.13330.2016",
            "clause_number": "10.4",
            "extraction_text": "Не менее 50 м.",
        },
    }
    hidden = {
        "status": "no_objects",
        "zone_kind": "restriction",
        "source": {"document_name": "ПЗЗ"},
    }
    summary = {
        "total_norms": 2,
        "restriction_zones": 1,
        "zones": [zone, hidden],
        "territory": {"documents_in_force": 2, "documents_out_of_force": 1},
    }

    report = build_inventory_report(summary)

    assert "### 1. СП 42.13330.2016, п. 10.4" in report
    assert "50 м вокруг объектов «Школа»; ограничение для «Дом»" in report
    assert "- ПЗЗ — в сценарии нет объектов" in report
    assert "| Документов, не действующих на ней | 1 |" in report
    assert build_inventory_report({"zones": [hidden]}) is None
