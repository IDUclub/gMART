"""Review regressions: retrieval isolation, bounded persistence and file delivery."""

import asyncio
import json
import os
import threading
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.common.files.layer_cache import LayerCache
from src.agents.common.files.temporary_file_store import FILES_TTL_SECONDS
from src.agents.services.compilance.compliance_archive import (
    ComplianceArchive,
    build_summary_report,
    llm_summary,
)
from src.agents.services.compilance.compliance_entity_mapping import (
    EntityLayers,
    EntityTarget,
)
from src.agents.services.compilance.compliance_retrieval import ScenarioComplianceSource
from src.agents.services.service_entities.compliance import DeclaredRequirements
from tests.unit.test_compliance_new_templates import FakeEffects, _provision_plan
from tests.unit.test_compliance_phase2 import IDS, Mcp, fc, feature, plan, run, service


@pytest.mark.parametrize("companion", [None, "empty", "populated"])
async def test_empty_zone_is_a_formal_pass_independent_of_shared_target(
    tmp_path, companion
):
    class EmptySources(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetServicesByGeometry" and arguments["services_names"] == [
                "Школа"
            ]:
                self.calls.append((name, deepcopy(arguments)))
                return {arguments["layer_name"]: fc(complete=True)}
            return await super().execute_tool(name, arguments, meta)

    svc, _ = service(tmp_path)
    plans = [plan(0)]
    if companion:
        other = plan(1, radius=300)
        if companion == "populated":
            other["declared_requirements"]["layers"][0]["entity"] = "Больница"
        plans.append(other)
    mcp = EmptySources()
    events = await run(svc, mcp, plans=plans)
    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    result = summary["results"][0]
    assert result["status"] == "passed" and result["formal_pass"] is True
    assert result["coverage"]["applicable_objects"] == 0
    target_calls = [
        args
        for name, args in mcp.calls
        if name == "GetServicesByGeometry" and args["services_names"] == ["Театр"]
    ]
    assert len(target_calls) == (1 if companion == "populated" else 0)
    root = ComplianceArchive("run-1", 772).root
    record = json.loads(
        (root / result["archive_path"] / "result.json").read_text("utf-8")
    )
    assert not any(
        c["function"]["arguments"].get("layer_name") == "Цели — Театр"
        for c in record["tool_calls"]
    )


async def test_source_and_target_pools_never_fall_back_to_each_other():
    source = ScenarioComplianceSource(None, None, 772)
    checked, params = service_executor().registry.validate_plan(plan(0))
    requirements = DeclaredRequirements.model_validate(plan(0)["declared_requirements"])
    source.sources = {"Школа": fc(complete=True), "Театр": fc(feature(), complete=True)}
    layers, _ = await source.retrieve(checked, params, requirements)
    assert "Театр" not in layers
    _, replay = source.replay_layers(checked, params, requirements, layers)
    assert "Цели — Театр" not in replay
    source.targets = {"Школа": fc(feature(), complete=True)}
    source.sources = {}
    layers, _ = await source.retrieve(checked, params, requirements)
    assert "Школа" not in layers


async def test_empty_zone_does_not_require_a_failed_physical_target_fetch(tmp_path):
    class EmptySourceAndUnavailableTargets(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetPhysicalObjects":
                raise RuntimeError("physical targets unavailable")
            if name == "GetServicesByGeometry":
                self.calls.append((name, deepcopy(arguments)))
                return {arguments["layer_name"]: fc(complete=True)}
            return await super().execute_tool(name, arguments, meta)

    raw = plan(0)
    raw["declared_requirements"]["layers"][1].update(
        entity="Жилой дом", entity_type="physical_object"
    )
    svc, _ = service(tmp_path)
    mcp = EmptySourceAndUnavailableTargets()
    events = await run(svc, mcp, plans=[raw])
    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    assert summary["results"][0]["status"] == "passed"
    assert summary["results"][0]["formal_pass"] is True


def service_executor():
    from src.agents.services.compilance.compliance_executor import (
        ComplianceTemplateExecutor,
    )

    return ComplianceTemplateExecutor(effects_client_factory=None)


async def test_provision_uses_object_effects_without_fetching_service_targets(tmp_path):
    class RejectGeometry(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetServicesByGeometry":
                raise AssertionError("Provision must not fetch services")
            return await super().execute_tool(name, arguments, meta)

    svc, _ = service(tmp_path)
    effects = FakeEffects()

    async def effects_factory(owner):
        return effects

    svc.compliance_executor.effects_client_factory = effects_factory
    events = await run(svc, RejectGeometry(), plans=[_provision_plan()])
    result = next(e["content"] for e in events if e["type"] == "compliance_result")
    assert result["verification_status"] == "complete"
    assert result["status"] == "violated" and len(effects.calls) == 1


@pytest.mark.parametrize("inventory", [False, True])
async def test_file_store_oserror_keeps_layers_and_completes_run(tmp_path, inventory):
    svc, checkpoints = service(tmp_path)
    loop_thread = threading.get_ident()
    threads = []

    def fail(*args, **kwargs):
        threads.append(threading.get_ident())
        raise OSError("disk unavailable")

    svc.file_store = SimpleNamespace(save=Mock(side_effect=fail))
    events = await run(svc, Mcp(), inventory=inventory)
    assert checkpoints["compliance_completed"] is True
    assert any(e["type"] == "feature_collection" for e in events)
    assert not any(e["type"] == "file" for e in events)
    assert threads and all(t != loop_thread for t in threads)
    assert str(svc.state_store.set_status.await_args.args[-1]) == "done"


@pytest.mark.parametrize("inventory", [False, True])
async def test_input_json_roundtrip_stays_off_loop_without_repeated_zone_inputs(
    tmp_path, monkeypatch, inventory
):
    import src.agents.services.compilance.compliance_archive as module

    loop_thread = threading.get_ident()
    serialized, parsed = [], []
    original_json, original_read = module._json, ComplianceArchive.read_inputs

    def serialize(value):
        serialized.append(threading.get_ident())
        return original_json(value)

    def read(path):
        parsed.append(threading.get_ident())
        return original_read(path)

    monkeypatch.setattr(module, "_json", serialize)
    monkeypatch.setattr(ComplianceArchive, "read_inputs", staticmethod(read))
    svc, checkpoints = service(tmp_path)
    mcp = Mcp()
    await run(svc, mcp, inventory=inventory)
    root = ComplianceArchive("run-1", 772).root
    saved = json.loads(
        (root / ("inventory.json" if inventory else "inputs.json")).read_text("utf-8")
    )
    records = saved["zones"] if inventory else saved["zones"].values()
    assert all("input_layers" not in z for z in records)
    if not inventory:
        assert saved["extra_layers"] == {}
        assert all(z["input_layer_refs"] for z in saved["zones"].values())
    checkpoints.pop("compliance_completed")
    calls = len(mcp.calls)
    await run(svc, mcp, checkpoint=checkpoints, inventory=inventory)
    assert len(mcp.calls) == calls
    assert parsed and serialized
    assert all(t != loop_thread for t in [*parsed, *serialized])


def test_compliance_cache_purge_uses_mtime_and_file_ttl_margin(tmp_path, monkeypatch):
    monkeypatch.delenv("GMART_COMPLIANCE_CACHE_TTL_SECONDS", raising=False)
    cache = LayerCache(tmp_path)
    assert cache.compliance_ttl_seconds > FILES_TTL_SECONDS
    now = time.time()
    stale = cache.compliance_root / "expired"
    recent = cache.compliance_root / "recent"
    for path in (stale, recent):
        path.mkdir(parents=True)
        (path / "result.json").write_text("{}", encoding="utf-8")
    os.utime(stale, (now - cache.compliance_ttl_seconds - 10,) * 2)
    os.utime(recent, (now - FILES_TTL_SECONDS,) * 2)
    assert cache.purge_expired() == 1
    assert not stale.exists() and recent.exists()
    assert cache.purge_expired() == 0
    monkeypatch.setenv("GMART_COMPLIANCE_CACHE_TTL_SECONDS", "10")
    assert LayerCache(tmp_path).purge_expired() == 1


async def test_archive_calls_and_provenance_match_exact_input_names(tmp_path):
    svc, _ = service(tmp_path)
    other = plan(1, radius=300)
    other["declared_requirements"]["layers"][0]["entity"] = "Больница"
    events = await run(svc, Mcp(), plans=[plan(0), other])
    root = ComplianceArchive("run-1", 772).root
    summary = next(e["content"] for e in events if e["type"] == "compliance_summary")
    for item, own, unrelated in zip(
        summary["results"], ("Школа", "Больница"), ("Больница", "Школа")
    ):
        record = json.loads(
            (root / item["archive_path"] / "result.json").read_text("utf-8")
        )
        calls = record["tool_calls"]
        assert sum(c["function"]["name"] == "GetProjectTerritory" for c in calls) == 1
        assert not any(
            c["function"]["arguments"].get("layer_name") == f"Источники — {unrelated}"
            for c in calls
        )
        assert any(
            c["function"]["arguments"].get("layer_name") == f"Источники — {own}"
            for c in calls
        )
        assert all(
            c["function"]["arguments"].get("restriction_id", item["restriction_id"])
            == item["restriction_id"]
            for c in calls
        )
        for layer in record["input_layers"]:
            if layer["name"].startswith(("Источники — ", "Цели — ")):
                assert layer["origin"] == "GetServicesByGeometry"
        assert (
            item["check_type_en"] == "distance"
            and item["check_type_ru"] == "расстояние"
        )
    report = (root / "summary.md").read_text("utf-8")
    assert "| расстояние |" in report
    assert "| distance_from_source |" not in report


async def test_archive_does_not_guess_provenance_from_the_layer_role(tmp_path):
    from src.agents.services.compilance.compliance_layers import norm_layer

    execution = await service_executor().execute(Mcp(), plan(0), 772)
    archive = ComplianceArchive("exact-origin", 772, root=tmp_path)
    record = archive.write_execution(
        execution,
        norm_layer(execution),
        shared_calls=[
            {
                "function": {
                    "name": "GetFunctionalZones",
                    "arguments": {"layer_name": "Театр-extra"},
                }
            },
            {
                "function": {
                    "name": "GetServicesByGeometry",
                    "arguments": {"layer_name": "Театр-extra"},
                }
            },
        ],
    )
    assert not any(
        call["function"]["arguments"].get("layer_name") == "Театр-extra"
        for call in record["tool_calls"]
    )
    target = next(item for item in record["input_layers"] if item["name"] == "Театр")
    assert target["origin"] == "GetServicesByGeometry"


def test_not_checked_report_groups_categories_and_llm_summary_is_bounded():
    results = [
        {
            "status": "not_checked",
            "status_reason": f"objects_by_geometry:Клиника {i}:ConnectionError secret {i}",
        }
        for i in range(100)
    ]
    results += [
        {"status": "not_checked", "status_reason": f"layer:Жилой дом {i}"}
        for i in range(10)
    ]
    results += [
        {"status": "violated", "source": {"clause_number": str(i)}} for i in range(70)
    ]
    summary = {"results": results}
    report = (
        build_summary_report(summary)
        .split("## Не запущено\n\n")[1]
        .split("## Где смотреть")[0]
    )
    assert "ошибка получения данных: 100" in report
    assert "нет необходимых слоёв или атрибутов: 10" in report
    assert "secret" not in report and "Клиника" not in report
    compact = llm_summary(summary)
    assert compact["not_checked"] == {
        "count": 110,
        "categories": {"retrieval_failed": 100, "missing_data": 10},
    }
    assert len(compact["norms"]) == 50
    assert compact["omitted_norms"] == 20 and compact["note"] == "и ещё 20"
    assert "secret" not in json.dumps(compact)


async def test_target_progress_arrives_before_the_last_fetch_finishes(tmp_path):
    first_progress = asyncio.Event()

    class ControlledMcp(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if (
                name == "GetServicesByGeometry"
                and arguments["layer_name"] == "Цели — Клиника"
            ):
                await first_progress.wait()
            return await super().execute_tool(name, arguments, meta)

    svc, _ = service(tmp_path)
    other = plan(1, radius=300)
    other["declared_requirements"]["layers"][1]["entity"] = "Клиника"
    events = []

    async def consume():
        async for event in svc._run_executable_compliance(
            mcp_client=ControlledMcp(),
            model="mock",
            request_id="run-1",
            scenario_id=772,
            restrictions=[{"check_plan": p} for p in (plan(0), other)],
            checkpoint={},
            owner="author",
        ):
            events.append(event)
            if (
                event["type"] == "progress"
                and event["content"]["stage"] == "objects_by_geometry"
                and event["content"]["current"] == 1
            ):
                assert event["content"]["total"] == 2
                first_progress.set()

    await asyncio.wait_for(consume(), timeout=5)
    assert first_progress.is_set()
    stages = [e["content"]["stage"] for e in events if e["type"] == "progress"]
    assert (
        stages.index("violation_geometry")
        < stages.index("cache")
        < stages.index("verdict_aggregation")
    )


@pytest.mark.parametrize("stage", ["territory", "source", "target", "physical"])
async def test_retrieval_errors_are_sanitised(tmp_path, stage):
    class BrokenMcp(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            fails = (
                (stage == "territory" and name == "GetProjectTerritory")
                or (stage == "physical" and name == "GetPhysicalObjects")
                or (
                    stage in {"source", "target"}
                    and name == "GetServicesByGeometry"
                    and arguments["layer_name"].startswith(
                        "Источники" if stage == "source" else "Цели"
                    )
                )
            )
            if fails:
                raise RuntimeError(
                    "PRIVATE payload https://secret.example?token=private"
                )
            return await super().execute_tool(name, arguments, meta)

    svc, _ = service(tmp_path)
    raw = plan(0)
    if stage == "physical":
        raw["declared_requirements"]["layers"][0].update(
            entity="Жилой дом", entity_type="physical_object"
        )
    events = await run(svc, BrokenMcp(), plans=[raw])
    result = next(e["content"] for e in events if e["type"] == "compliance_result")
    assert result["status"] == "not_checked"
    assert "PRIVATE" not in json.dumps(events) and "secret.example" not in json.dumps(
        events
    )
    assert any(
        "retrieval_failed" in reason for reason in result["missing_requirements"]
    )


@pytest.mark.parametrize("stage", ["territory", "source", "target", "physical"])
async def test_retrieval_token_expiry_propagates(tmp_path, stage):
    class ExpiredMcp(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if (
                (stage == "territory" and name == "GetProjectTerritory")
                or (stage == "physical" and name == "GetPhysicalObjects")
                or (
                    stage in {"source", "target"}
                    and name == "GetServicesByGeometry"
                    and arguments["layer_name"].startswith(
                        "Источники" if stage == "source" else "Цели"
                    )
                )
            ):
                raise TokenExpiredError("expired")
            return await super().execute_tool(name, arguments, meta)

    svc, checkpoints = service(tmp_path)
    raw = plan(0)
    if stage == "physical":
        raw["declared_requirements"]["layers"][0].update(
            entity="Жилой дом", entity_type="physical_object"
        )
    with pytest.raises(TokenExpiredError):
        await run(svc, ExpiredMcp(), plans=[raw])
    assert not checkpoints.get("compliance_completed")


async def test_result_checkpoints_are_batched_and_cache_files_are_private(tmp_path):
    svc, _ = service(tmp_path)
    plans = []
    for index in range(30):
        raw = plan(0, radius=index + 1)
        raw["source"]["restriction_id"] = f"00000000-0000-4000-8000-{index + 1:012d}"
        plans.append(raw)
    await run(svc, Mcp(), plans=plans)
    writes = [
        call
        for call in svc.state_store.save_checkpoint.await_args_list
        if call.args[1] == "compliance_executions"
    ]
    assert len(writes) == 2
    root = ComplianceArchive("run-1", 772).root
    assert len(list((root / ".executions").glob("*.json"))) == 30
    with ZipFile(next(root.glob("*.zip"))) as zipped:
        assert not any(name.startswith(".executions/") for name in zipped.namelist())


async def test_reconnect_before_a_redis_batch_reuses_completed_norm(tmp_path):
    svc, checkpoints = service(tmp_path)
    mcp = Mcp()
    plans = [plan(0), plan(1, radius=300)]
    stream = svc._run_executable_compliance(
        mcp_client=mcp,
        model="mock",
        request_id="run-1",
        scenario_id=772,
        restrictions=[{"check_plan": p} for p in plans],
        checkpoint={},
        owner="author",
    )
    async for event in stream:
        if event["type"] == "compliance_result":
            break
    await stream.aclose()
    assert "compliance_executions" not in checkpoints
    events = await run(svc, mcp, checkpoint=checkpoints, plans=plans)
    checks = [
        args["restriction_id"] for name, args in mcp.calls if name.startswith("Check")
    ]
    assert checks == IDS
    assert len([e for e in events if e["type"] == "compliance_result"]) == 2
    assert checkpoints["compliance_completed"] is True


async def test_fallback_by_geometry_receives_only_referenced_zone_layers():
    class WithPhysical(Mcp):
        async def execute_tool(self, name, arguments, meta=None):
            if name == "GetPhysicalObjects":
                return {"Дом": fc(feature(), complete=True)}
            return await super().execute_tool(name, arguments, meta)

    requirements = DeclaredRequirements(
        layers=[
            {"role": "house", "entity": "Дом", "entity_type": "physical_object"},
            {"role": "school", "entity": "Школа", "entity_type": "service"},
            {"role": "clinic", "entity": "Клиника", "entity_type": "service"},
        ]
    )
    mcp = WithPhysical()
    layers, _ = await service_executor()._retrieve_layers(mcp, requirements, 772)
    assert "Дом" in layers and "Школа" in layers
    queries = [args for name, args in mcp.calls if name == "GetServicesByGeometry"]
    assert len(queries) == 2
    assert all(
        set(args["layers"]) == set(args["zones_layers"]) == {"project_territory"}
        for args in queries
    )


async def test_user_files_overlay_fetches_territory_once_for_distinct_norms(
    tmp_path, monkeypatch
):
    import src.agents.services.restriction.restriction_parser_service as module

    svc, _ = service(tmp_path)
    mcp = Mcp()
    other = plan(1, radius=300)
    other["declared_requirements"]["layers"][0]["entity"] = "Больница"
    plans = [plan(0), other]
    mapping = SimpleNamespace(assignment=lambda: {})
    monkeypatch.setattr(module.EntityMapping, "from_dict", lambda data: mapping)
    monkeypatch.setattr(
        module,
        "build_entity_layers",
        lambda *args: EntityLayers(
            {
                EntityTarget("Театр", "service"): [
                    feature({"type": "Point", "coordinates": [30, 60]})
                ]
            },
            {},
            None,
        ),
    )
    monkeypatch.setattr(
        module, "mapping_event", lambda *args: {"type": "mapping", "content": {}}
    )
    monkeypatch.setattr(module, "user_files_summary", lambda **kwargs: {})
    out = []
    async for _ in svc._prepare_user_files(
        request_id="run-1",
        mcp_client=mcp,
        plans=plans,
        sources_by_id={},
        user_files=SimpleNamespace(upload_ids=["upload"]),
        model="mock",
        scope_key=None,
        user_query="Проверь",
        scope=None,
        scenario_id=772,
        checkpoint={
            module.PipelineStep.COMPLIANCE_MAPPING: {"mapping": {"saved": True}}
        },
        out=out,
    ):
        pass
    source = out[0][0]
    for raw in plans:
        canonical = await svc.compliance_executor.canonical_requirements(mcp, raw)
        await source.retrieve(*canonical)
    assert sum(name == "GetProjectTerritory" for name, _ in mcp.calls) == 1
