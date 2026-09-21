"""Public layer contracts across agent output, transports and replay."""

from copy import deepcopy

import pytest

from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.provision_executor import ProvisionAgentExecutor
from src.agents.a2a.scenario_data_executor import ScenarioDataAgentExecutor
from src.agents.services.layer_attributes import compact_layer, compact_layer_event
from src.agents.services.orchestrator.orchestrator_service import OrchestratorService
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService
from src.agents.services.service_entities.orchestrator_plan import OrchestratorStep


def collection(properties):
    return {
        "type": "FeatureCollection",
        "bbox": [30, 60, 31, 61],
        "features": [
            {
                "type": "Feature",
                "id": "object-1",
                "geometry": {
                    "type": "Point",
                    "coordinates": [30.123456789, 60.123456789],
                },
                "properties": properties,
            }
        ],
    }


def properties(event):
    return event["content"]["feature_collection"]["features"][0]["properties"]


@pytest.mark.parametrize("profile", ["restrictions", "provision", "effects", "pzz"])
def test_basic_identity_is_compact_and_inputs_are_not_mutated(profile):
    source = collection(
        {
            "physical_object_id": 7,
            "name": "Школа",
            "address": "Улица, 1",
            "physical_object_type": {
                "id": 4,
                "name": "Здание",
                "created_at": "old",
                "children": [1],
            },
            "created_at": "old",
            "updated_at": "new",
            "is_locked": False,
            "_row_id": "internal",
            "building": {"storeys_count": 4},
        }
    )
    original = deepcopy(source)
    result = compact_layer(source, profile)
    assert result["features"][0]["properties"] == {
        "physical_object_id": 7,
        "name": "Школа",
        "address": "Улица, 1",
        "physical_object_type": {"id": 4, "name": "Здание"},
    }
    assert result["features"][0]["geometry"] == original["features"][0]["geometry"]
    assert result["features"][0]["id"] == "object-1"
    assert result["bbox"] == original["bbox"]
    result["features"][0]["properties"]["physical_object_type"]["name"] = "Changed"
    assert source == original


def test_restriction_and_compliance_keep_all_reasons_but_not_internal_evidence():
    source = collection(
        {
            "buffer_size": 100,
            "restriction_name": "Зона школ",
            "compliance_status": "violated",
            "compliance_evidence": [
                {
                    "restriction_id": "r1",
                    "measured_value": 30,
                    "threshold": 50,
                    "unit": "м",
                    "generator_refs": [
                        {"id": "school/1", "name": "Школа", "debug": "raw"}
                    ],
                    "provenance": {
                        "document_name": "СП",
                        "clause_number": "1.2",
                        "extra": {"debug": "raw"},
                    },
                    "input_revision": "internal",
                    "template_version": 1,
                }
            ],
            "restriction_evidence": [{"title": "Первое"}, {"title": "Второе"}],
            "updated_at": "old",
            "population": 100,
        }
    )
    event = next(RestrictionParserService._feature_collections({"objects": source}))
    result = properties(event)
    assert result["restriction_evidence"] == [{"title": "Первое"}, {"title": "Второе"}]
    evidence = result["compliance_evidence"][0]
    assert evidence == {
        "restriction_id": "r1",
        "measured_value": 30,
        "threshold": 50,
        "unit": "м",
        "generator_refs": [{"id": "school/1", "name": "Школа"}],
        "provenance": {"document_name": "СП", "clause_number": "1.2"},
    }
    assert result["buffer_size"] == 100
    assert result["compliance_status"] == "violated"
    assert "updated_at" not in result and "population" not in result
    artifact = RestrictionAgentExecutor._geojson_artifact(
        "objects", event["content"]["feature_collection"]
    )
    assert artifact["parts"][0]["data"]["features"][0]["properties"] == result


@pytest.mark.parametrize(
    "layer,metrics",
    [
        (
            "buildings",
            {"population": 20, "demand": 3, "demand_left": 1, "provision_value": 0.5},
        ),
        ("services", {"capacity": 100, "capacity_left": 20, "service_load": 80}),
        (
            "links",
            {"building_index": 1, "service_index": 2, "distance": 300, "demand": 2},
        ),
    ],
)
def test_provision_layers_keep_calculation_fields(layer, metrics):
    source = collection({**metrics, "created_at": "old", "living_area": 1000})
    data = {"services": {"1": {"name": "Школа", "layers": {layer: source}}}}
    event = next(ProvisionService._provision_feature_collections(data))
    assert properties(event) == metrics
    direct = next(ProvisionService._feature_collections({layer: source}))
    assert properties(direct) == metrics
    artifact = ProvisionAgentExecutor._geojson_artifact(
        layer, event["content"]["feature_collection"]
    )
    assert artifact["parts"][0]["data"]["features"][0]["properties"] == metrics


def test_effects_preserve_russian_and_english_before_after_metrics():
    metrics = {
        "supplied_demands_within_before": 10,
        "supplied_demands_within_after": 20,
        "absolute_total": 10,
        "index_total": 0.5,
        "is_project": True,
        "Абсолютный эффект (чел)": 10,
        "Удовлетворённый спрос в нормативной доступности (после) (чел)": 20,
        "Вместимость (чел)": 100,
    }
    source = collection({**metrics, "debug": "raw", "Количество этажей": 4})
    events = list(
        ProvisionService._effects_feature_collections(
            {
                "before_prove_data": {"buildings": source},
                "after_prove_data": {"services": source},
                "effects": source,
            }
        )
    )
    assert len(events) == 3
    assert all(properties(event) == metrics for event in events)


@pytest.mark.parametrize(
    "agent", ["restriction", "compliance", "provision", "pzz", "scenario_data"]
)
def test_orchestrator_replay_respects_originating_agent(agent):
    source = collection({"id": 1, "name": "Объект", "internal": "raw"})
    event = {
        "type": "feature_collection",
        "content": {"name": "layer", "feature_collection": source},
    }
    step = OrchestratorStep(agent=agent, task="test")
    wrapped = OrchestratorService._step_event(1, step, event)
    replay = compact_layer_event(wrapped, "orchestrator")["content"]["event"]
    if agent == "scenario_data":
        assert replay is event
    else:
        assert properties(replay) == {"id": 1, "name": "Объект"}
    assert properties(event)["internal"] == "raw"


def test_scenario_data_preserves_every_attribute_through_a2a_and_replay():
    source = collection(
        {"id": 1, "created_at": "old", "building": {"any": [1, 2]}, "custom": "value"}
    )
    _, result = next(ScenarioDataService._feature_collections({"result": source}))
    artifact = ScenarioDataAgentExecutor._geojson_artifact("data", result)
    assert artifact["parts"][0]["data"] == source
    event = {"type": "feature_collection", "content": {"feature_collection": result}}
    assert compact_layer_event(event, "scenario_data") is event


def test_empty_layers_null_properties_and_non_layer_events():
    assert (
        compact_layer({"type": "FeatureCollection", "features": []}, "pzz")["features"]
        == []
    )
    assert compact_layer(collection(None), "pzz")["features"][0]["properties"] is None
    event = {"type": "table", "content": {"columns": ["any"]}}
    assert compact_layer_event(event, "provision") is event


def test_pzz_classification_status_and_building_category_are_preserved():
    expected = {
        "Статус_классификации": "Требуется ручная проверка",
        "Категория_объекта": "Здание",
        "ВРИ_ЕГРН": "Жилой дом",
        "Топ1_возможный_ВРИ": "2.1 — ИЖС",
        "Причина": "Неоднозначная классификация",
    }
    result = compact_layer(
        collection({**expected, "Топ5_возможных_ВРИ": "raw", "CHECK_SCOPE": "debug"}),
        "pzz",
    )
    assert result["features"][0]["properties"] == expected
