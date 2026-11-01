"""Routing and planning stages: the structured decisions that pick agents and tools.

Where a builder post-processes the model's plan with code (canonical catalog names,
deterministic complements), the output's top level is the model's own plan and the
pipeline result is under ``final``: checks on the top level measure the model,
checks on ``final.*`` measure what the agent would actually execute.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from benchmarks.harness.stage import StageContext, stage
from src.agents.dto.pzz_request_dto import PzzInputs
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.normgraph.normgraph_reasoning import NormGraphRetrievalPlanner
from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
from src.agents.services.orchestrator.orchestrator_plan_builder import (
    OrchestratorPlanBuilder,
)
from src.agents.services.project_files.project_params import extract_intent
from src.agents.services.provision.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.pzz.pzz_service import PzzService
from src.agents.services.restriction.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.service_entities.orchestrator_plan import OrchestratorAgent


def _capture_first(builder: Any, method: str) -> dict[str, Any]:
    """Keep the first value ``builder.<method>`` returns: the model's raw plan."""
    captured: dict[str, Any] = {}
    original = getattr(builder, method)

    async def wrapper(*args, **kwargs):
        result = await original(*args, **kwargs)
        captured.setdefault("raw", result)
        return result

    setattr(builder, method, wrapper)
    return captured


def _capture_first_arg(owner: Any, method: str) -> dict[str, Any]:
    """Keep the first argument of the first ``owner.<method>`` call."""
    captured: dict[str, Any] = {}
    original = getattr(owner, method)

    def wrapper(first, *args, **kwargs):
        captured.setdefault("raw", first)
        return original(first, *args, **kwargs)

    setattr(owner, method, wrapper)
    return captured


def _with_final(raw: Any, final: Any) -> dict[str, Any]:
    return {**raw.model_dump(mode="json"), "final": final.model_dump(mode="json")}


def _orchestrator_agents(inp: dict[str, Any]) -> list:
    """The catalogue as ``available_agents`` would filter it with every MCP configured."""
    keys = inp.get("agents")
    if keys:
        return [AGENT_CATALOG[OrchestratorAgent(key)] for key in keys]
    scenario_id = inp.get("scenario_id")
    uploads = int(inp.get("compliance_uploads") or 0)
    return [
        entry
        for entry in AGENT_CATALOG.values()
        if not entry.requires_scenario_id
        or scenario_id is not None
        or (uploads and entry.key == OrchestratorAgent.COMPLIANCE)
    ]


@stage(
    "orchestrator.plan",
    description="Маршрутизация запроса между агентами (план шагов)",
    scope="OrchestratorPlanBuilder.build_plan; верх — сырой план модели, final — после "
    "детерминированных дополнений (сравнение документов, нормы+документы).",
)
async def orchestrator_plan(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    builder = OrchestratorPlanBuilder(ctx.llm)
    captured = _capture_first(builder, "_request_plan")
    final = await builder.build_plan(
        ctx.model,
        inp["user_query"],
        _orchestrator_agents(inp),
        history=inp.get("history"),
        scenario_id=inp.get("scenario_id"),
        pzz_inputs=inp.get("pzz_inputs"),
        compliance_uploads=int(inp.get("compliance_uploads") or 0),
    )
    return _with_final(captured["raw"], final)


@stage(
    "restriction.plan",
    description="План построения буферов/ограничений по каталогам сценария",
    scope="RestrictionPlanBuilder.build_plan с repair-проходами; верх — первый план "
    "модели, final — канонизированный план после всех проходов.",
)
async def restriction_plan(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    builder = RestrictionPlanBuilder(ctx.llm)
    captured = _capture_first(builder, "_request_plan")
    final = await builder.build_plan(
        ctx.model,
        inp["user_query"],
        int(inp.get("scenario_id") or 0),
        list(inp.get("services_catalog") or []),
        list(inp.get("physical_objects_catalog") or []),
        history=inp.get("history"),
        normgraph_restrictions=inp.get("normgraph_restrictions"),
    )
    return _with_final(captured["raw"], final)


@stage(
    "provision.plan",
    description="Режим и параметры расчёта обеспеченности",
    scope="ProvisionPlanBuilder.build_plan; верх — план модели, final — после "
    "приведения названий к каталогу.",
)
async def provision_plan(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    builder = ProvisionPlanBuilder(ctx.llm)
    captured = _capture_first(builder, "_request_plan")
    final = await builder.build_plan(
        ctx.model,
        inp["user_query"],
        list(inp["services_catalog"]),
        history=inp.get("history"),
    )
    if "raw" not in captured:  # empty catalogue: decided without the model
        return {**final.model_dump(mode="json"), "final": final.model_dump(mode="json")}
    return _with_final(captured["raw"], final)


@stage(
    "norms.planner",
    description="Параметры запроса к графу норм NormGraph",
    scope="NormGraphRetrievalPlanner.build_plan (вкл. детерминированный clamp).",
)
async def norms_planner(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await NormGraphRetrievalPlanner(ctx.llm).build_plan(
        ctx.model,
        inp["user_query"],
        history=inp.get("history"),
        prev_critique=inp.get("prev_critique"),
        prev_query=inp.get("prev_query"),
        prev_object=inp.get("prev_object"),
    )


@stage(
    "dvd.planner",
    description="План поиска по документам IDU_DVD",
    scope="RetrievalPlanner.build_plan; верх — план модели, final — после clamp и "
    "apply_scope (адрес и документ из запроса код подставляет сам).",
)
async def dvd_planner(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    planner = RetrievalPlanner(ctx.llm)
    # The model's plan is the first argument of the deterministic clamp.
    captured = _capture_first_arg(planner, "_clamp")
    final = await planner.build_plan(
        ctx.model,
        inp["user_query"],
        history=inp.get("history"),
        prev_critique=inp.get("prev_critique"),
        prev_query=inp.get("prev_query"),
        available_tags=inp.get("available_tags"),
    )
    return _with_final(captured["raw"], final)


@stage(
    "pzz.inputs",
    description="Режим проверки ПЗЗ, год и источник зон из запроса",
    scope="PzzService._resolve_inputs без загрузок (иначе режим выводится кодом).",
)
async def pzz_inputs(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    resolved = await PzzService._resolve_inputs(
        SimpleNamespace(llm_client=ctx.llm),
        ctx.model,
        inp["user_query"],
        inp.get("scenario_id"),
        PzzInputs.model_validate(inp.get("inputs") or {}),
        inp.get("history") or [],
    )
    return {
        "mode": resolved.mode,
        "year": resolved.year,
        "source": resolved.source,
        "llm_called": bool(ctx.llm.calls),
    }


@stage(
    "project.inputs",
    description="Название проекта/сценария, профиль и политика объектов",
    scope="project_params.extract_intent.",
)
async def project_inputs(inp: dict[str, Any], ctx: StageContext) -> Any:
    profiles = [
        SimpleNamespace(name=item["name"], label=item["label"])
        for item in inp["profiles"]
    ]
    return await extract_intent(
        ctx.llm, ctx.model, inp["user_query"], profiles, history=inp.get("history")
    )
