"""Run-level attribute decisions, two-phase queries, replay and checkpoints."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZipFile

import pytest

from src.agents.services.compilance.compliance_archive import ComplianceArchive
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_inventory import RestrictionZoneBuilder
from src.agents.services.compilance.compliance_retrieval import ScenarioComplianceSource
from src.agents.services.service_entities.compliance import DeclaredRequirements
from tests.unit.test_compliance_phase2 import IDS, Mcp, fc, feature, plan, run, service


def physical_plan(
    index=0, *, source=True, target=True, radius=100, predicate="intersects"
):
    raw = plan(index, radius=radius, predicate=predicate)
    for item, enabled in zip(raw["declared_requirements"]["layers"], (source, target)):
        if enabled:
            item["entity_type"] = "physical_object"
    return raw


class PhysicalMcp(Mcp):
    async def execute_tool(self, name, arguments, meta=None):
        if name == "GetPhysicalObjects":
            self.calls.append((name, deepcopy(arguments)))
            return {
                entity: fc(
                    feature(physical_object_id=7, building={"height": 12}),
                    complete=True,
                )
                for entity in arguments["physical_objects_names"]
            }
        return await super().execute_tool(name, arguments, meta)


async def prepared(raws):
    executor = ComplianceTemplateExecutor(effects_client_factory=None)
    mcp = PhysicalMcp()
    source = ScenarioComplianceSource(executor, mcp, 772)
    await source.load_sources(raws, RestrictionZoneBuilder(executor))
    for index, raw in enumerate(raws):
        source.zones[raw["source"]["restriction_id"]] = SimpleNamespace(
            status="shown",
            zones=fc(feature()),
            zone_count=1,
            tool_calls=[
                {
                    "function": {
                        "name": "CreateRestrictionZones",
                        "arguments": {"layer_name": f"zone-{index}"},
                    }
                }
            ],
        )
    return source, mcp


async def test_physical_sources_max_buffer_targets_union_replay_progress_snapshot():
    source, mcp = await prepared([physical_plan(0), physical_plan(1, radius=300)])
    progress = AsyncMock()
    await source.load_targets(progress)
    queries = [
        args for name, args in mcp.calls if name == "GetPhysicalObjectsByGeometry"
    ]
    assert len(queries) == 2
    src, target = queries
    assert src["physical_objects_names"] == ["Школа"]
    assert src["zones_layers"] == ["project_territory"] and src["buffer_m"] == 300
    assert src["layer_name"] == "Источники — Школа"
    assert target["physical_objects_names"] == ["Театр"]
    assert target["zones_layers"] == ["zone-0", "zone-1"]
    assert target["layer_name"] == "Цели — Театр"
    assert target["buffer_m"] == 0 and target["intersection_type"] == "intersects"
    progress.assert_awaited_once_with(1, 1, "Театр")
    assert all("layers" not in call["function"]["arguments"] for call in source.calls)

    checked, params, requirements = source.canonical[IDS[0]]
    layers, _ = await source.retrieve(checked, params, requirements)
    mapped, replay = source.replay_layers(checked, params, requirements, layers)
    assert [item.entity for item in mapped.layers] == [
        "Источники — Школа",
        "Цели — Театр",
    ]
    assert set(replay) == {"Источники — Школа", "Цели — Театр", "project_territory"}
    restored = ScenarioComplianceSource(source.executor, mcp, 772)
    restored.restore(json.loads(json.dumps(source.snapshot(), ensure_ascii=False)))
    restored_layers, _ = await restored.retrieve(checked, params, requirements)
    assert restored.replay_layers(checked, params, requirements, restored_layers) == (
        mapped,
        replay,
    )


@pytest.mark.parametrize(
    "predicates,expected",
    [(("within", "within"), "within"), (("within", "intersects"), "intersects")],
)
async def test_physical_target_predicate_matches_services(predicates, expected):
    source, mcp = await prepared(
        [physical_plan(i, predicate=p) for i, p in enumerate(predicates)]
    )
    await source.load_targets()
    target = [
        args for name, args in mcp.calls if name == "GetPhysicalObjectsByGeometry"
    ][-1]
    assert target["intersection_type"] == expected


@pytest.mark.parametrize("attribute_plan_first", [True, False])
async def test_any_attribute_keeps_entity_full_as_source_and_target(
    attribute_plan_first,
):
    attributed = physical_plan(0, target=False)
    attributed["declared_requirements"]["attributes"] = [
        {
            "role": "height",
            "on": "source",
            "required": False,
            "accepts": [{"field": "building.height", "unit": "m", "quality": "direct"}],
        }
    ]
    plain = physical_plan(1, source=False)
    plain["declared_requirements"]["layers"][1]["entity"] = "Школа"
    raws = [attributed, plain] if attribute_plan_first else [plain, attributed]
    source, mcp = await prepared(raws)
    await source.load_targets()
    assert source.full_physical_entities == {"Школа"}
    full_calls = [args for name, args in mcp.calls if name == "GetPhysicalObjects"]
    assert full_calls == [
        {"physical_objects_names": ["Школа"], "scenario_id": 772, "centers_only": False}
    ]
    assert not any(name == "GetPhysicalObjectsByGeometry" for name, _ in mcp.calls)
    # load_sources passes only a layer requirement to _retrieve_layers. Its explicit
    # run-level decision must retain the building fields on both roles.
    assert (
        source.sources["physical_object:Школа"]["features"][0]["properties"][
            "building"
        ]["height"]
        == 12
    )
    assert (
        source.targets["physical_object:Школа"]
        == source.sources["physical_object:Школа"]
    )
    restored = ScenarioComplianceSource(source.executor, mcp, 772)
    restored.restore(json.loads(json.dumps(source.snapshot(), ensure_ascii=False)))
    assert restored.full_physical_entities == {"Школа"}
    for raw in raws:
        checked, params, reqs = source.canonical[raw["source"]["restriction_id"]]
        layers, _ = await restored.retrieve(checked, params, reqs)
        _, replay = restored.replay_layers(checked, params, reqs, layers)
        assert replay["Школа"]["features"][0]["properties"]["building"]["height"] == 12


async def test_service_and_physical_entity_same_name_remain_distinct():
    physical = physical_plan(0, target=False)
    services = plan(1)
    source, mcp = await prepared([physical, services])
    await source.load_targets()
    assert (
        "physical_object:Школа" in source.sources and "service:Школа" in source.sources
    )
    names = [
        args["layer_name"] for name, args in mcp.calls if name.endswith("ByGeometry")
    ]
    assert "Источники — Школа (physical_object)" in names
    assert "Источники — Школа (service)" in names
    for rid, expected in [(IDS[0], "physical_object_id"), (IDS[1], "service_id")]:
        checked, params, reqs = source.canonical[rid]
        layers, _ = await source.retrieve(checked, params, reqs)
        mapped, replay = source.replay_layers(checked, params, reqs, layers)
        assert expected in replay[mapped.layers[0].entity]["features"][0]["properties"]


async def test_empty_physical_target_zone_needs_no_query():
    source, mcp = await prepared([physical_plan()])
    source.zones[IDS[0]].zone_count = 0
    source.zones[IDS[0]].zones = fc()
    await source.load_targets()
    assert (
        len([1 for name, _ in mcp.calls if name == "GetPhysicalObjectsByGeometry"]) == 1
    )
    checked, params, reqs = source.canonical[IDS[0]]
    layers, _ = await source.retrieve(checked, params, reqs)
    _, replay = source.replay_layers(checked, params, reqs, layers)
    assert replay[f"Цели — Театр (пустая зона {IDS[0]})"]["features"] == []


async def test_physical_archive_origin_and_recorded_calls(tmp_path):
    svc, _ = service(tmp_path)
    events = await run(svc, PhysicalMcp(), plans=[physical_plan()])
    calls = [
        c
        for e in events
        if e["type"] == "tool_call"
        for c in e["content"]["tool_calls"]
    ]
    assert (
        sum(c["function"]["name"] == "GetPhysicalObjectsByGeometry" for c in calls) == 2
    )
    assert all("layers" not in c["function"]["arguments"] for c in calls)
    with ZipFile(next(ComplianceArchive("run-1", 772).root.glob("*.zip"))) as zipped:
        record = json.loads(zipped.read(f"violated/{IDS[0]}/result.json"))
    inputs = {item["name"]: item["origin"] for item in record["input_layers"]}
    assert (
        inputs["Источники — Школа"]
        == inputs["Цели — Театр"]
        == "GetPhysicalObjectsByGeometry"
    )


@pytest.mark.parametrize(
    "prefix,reason",
    [
        ("Источники", "source_layers:retrieval_failed"),
        ("Цели", "objects_by_geometry:retrieval_failed"),
    ],
)
async def test_physical_failures_have_services_reason_codes(tmp_path, prefix, reason):
    class Broken(PhysicalMcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetPhysicalObjectsByGeometry" and arguments[
                "layer_name"
            ].startswith(prefix):
                raise RuntimeError("private upstream detail")
            return await super().execute_tool(name, arguments, meta)

    svc, _ = service(tmp_path)
    events = await run(svc, Broken(), plans=[physical_plan()])
    result = next(e["content"] for e in events if e["type"] == "compliance_result")
    assert (
        result["status"] == "not_checked" and reason in result["missing_requirements"]
    )
    assert "private upstream detail" not in json.dumps(events)


async def test_fallback_respects_attributes_explicit_override_and_user_overlay():
    executor = ComplianceTemplateExecutor(effects_client_factory=None)
    reqs = DeclaredRequirements.model_validate(
        physical_plan(source=True, target=False)["declared_requirements"]
    )
    for kwargs, expected in [
        ({}, "GetPhysicalObjectsByGeometry"),
        ({"full_physical_entities": {"Школа"}}, "GetPhysicalObjects"),
        ({"preserve_physical_attributes": True}, "GetPhysicalObjects"),
    ]:
        mcp = PhysicalMcp()
        _, calls = await executor._retrieve_layers(
            mcp, reqs, 772, buffer_m=250, **kwargs
        )
        call = next(c["function"] for c in calls if c["function"]["name"] == expected)
        if expected.endswith("ByGeometry"):
            assert call["arguments"]["layer_name"] == "Школа"
            assert call["arguments"]["buffer_m"] == 250
            assert "layers" not in call["arguments"]

    declared = reqs.model_dump(mode="json")
    declared["attributes"] = [
        {
            "role": "height",
            "on": "source",
            "accepts": [{"field": "building.height", "quality": "direct", "unit": "m"}],
        }
    ]
    reqs = DeclaredRequirements.model_validate(declared)
    mcp = PhysicalMcp()
    await executor._retrieve_layers(mcp, reqs, 772)
    assert any(name == "GetPhysicalObjects" for name, _ in mcp.calls)
    assert not any(name == "GetPhysicalObjectsByGeometry" for name, _ in mcp.calls)


async def test_restore_legacy_checkpoint_keeps_full_physical_layers():
    executor = ComplianceTemplateExecutor(effects_client_factory=None)
    source = ScenarioComplianceSource(executor, None, 772)
    source.restore(
        {
            "sources": {"Школа": fc(feature(service_id=1))},
            "targets": {
                "Театр": fc(feature(physical_object_id=2, building={"height": 12}))
            },
            "territory": fc(feature()),
            "errors": {},
            "calls": [],
        }
    )
    raw = physical_plan(source=False)
    checked, params = executor.registry.validate_plan(raw)
    requirements = DeclaredRequirements.model_validate(raw["declared_requirements"])
    layers, _ = await source.retrieve(checked, params, requirements)
    mapped, replay = source.replay_layers(checked, params, requirements, layers)
    assert [item.entity for item in mapped.layers] == ["Источники — Школа", "Театр"]
    assert replay["Театр"]["features"][0]["properties"]["building"]["height"] == 12
