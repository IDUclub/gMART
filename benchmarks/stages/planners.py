"""Routing and planning stages: the structured decisions that pick agents and tools.

The export keeps the first model request of each builder: the model's own plan, before
the code canonicalises names, adds deterministic steps or runs repair passes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from benchmarks.harness.stage import StageContext, stage
from src.agents.dto.pzz_request_dto import PzzInputs
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.normgraph.normgraph_reasoning import NormGraphRetrievalPlanner
from src.agents.services.orchestrator.orchestrator_adk import AdkOrchestration
from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
from src.agents.services.orchestrator.orchestrator_plan_builder import (
    OrchestratorPlanBuilder,
)
from src.agents.services.project_files.project_params import extract_intent
from src.agents.services.provision.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.pzz.pzz_service import PzzService
from src.agents.services.restriction.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.service_entities.orchestrator_plan import OrchestratorAgent


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
    scope="OrchestratorPlanBuilder.build_plan: план модели до детерминированных "
    "дополнений (сравнение документов, нормы+документы).",
)
async def orchestrator_plan(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await OrchestratorPlanBuilder(ctx.llm).build_plan(
        ctx.model,
        inp["user_query"],
        _orchestrator_agents(inp),
        history=inp.get("history"),
        scenario_id=inp.get("scenario_id"),
        pzz_inputs=inp.get("pzz_inputs"),
        compliance_uploads=int(inp.get("compliance_uploads") or 0),
    )


@stage(
    "orchestrator.route",
    description="Автономный оркестратор (Google ADK): первый ход — вызовы агентов",
    scope="AdkOrchestration: первый запрос ADK-агента с инструментами run_<агент> и "
    "ask_clarification, до детерминированных поправок плана.",
)
async def orchestrator_route(inp: dict[str, Any], ctx: StageContext) -> Any:
    host = SimpleNamespace(
        llm_client=ctx.llm, plan_builder=OrchestratorPlanBuilder(ctx.llm)
    )
    orchestration = AdkOrchestration(
        host,
        model=ctx.model,
        user_query=inp["user_query"],
        history=list(inp.get("history") or []),
        agents=_orchestrator_agents(inp),
        step_kwargs={},
        scenario_id=inp.get("scenario_id"),
        pzz_inputs=inp.get("pzz_inputs"),
        compliance_uploads=int(inp.get("compliance_uploads") or 0),
    )
    async for _ in orchestration.events():
        pass
    return orchestration.state.plan


@stage(
    "restriction.plan",
    description="План построения буферов/ограничений по каталогам сценария",
    scope="RestrictionPlanBuilder.build_plan: первый план модели, без канонизации "
    "названий и repair-проходов.",
)
async def restriction_plan(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await RestrictionPlanBuilder(ctx.llm).build_plan(
        ctx.model,
        inp["user_query"],
        int(inp.get("scenario_id") or 0),
        list(inp.get("services_catalog") or []),
        list(inp.get("physical_objects_catalog") or []),
        history=inp.get("history"),
        normgraph_restrictions=inp.get("normgraph_restrictions"),
    )


@stage(
    "provision.plan",
    description="Режим и параметры расчёта обеспеченности",
    scope="ProvisionPlanBuilder.build_plan: план модели до приведения названий к "
    "каталогу; пустой каталог решается кодом.",
)
async def provision_plan(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await ProvisionPlanBuilder(ctx.llm).build_plan(
        ctx.model,
        inp["user_query"],
        list(inp["services_catalog"]),
        history=inp.get("history"),
    )


@stage(
    "norms.planner",
    description="Параметры запроса к графу норм NormGraph",
    scope="NormGraphRetrievalPlanner.build_plan: план модели до детерминированного clamp.",
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
    scope="RetrievalPlanner.build_plan: план модели до clamp и apply_scope (адрес и "
    "документ из запроса код подставляет сам).",
)
async def dvd_planner(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await RetrievalPlanner(ctx.llm).build_plan(
        ctx.model,
        inp["user_query"],
        history=inp.get("history"),
        prev_critique=inp.get("prev_critique"),
        prev_query=inp.get("prev_query"),
        available_tags=inp.get("available_tags"),
    )


@stage(
    "pzz.inputs",
    description="Режим проверки ПЗЗ, год и источник зон из запроса",
    scope="PzzService._resolve_inputs без загрузок (иначе режим выводится кодом).",
)
async def pzz_inputs(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await PzzService._resolve_inputs(
        SimpleNamespace(llm_client=ctx.llm),
        ctx.model,
        inp["user_query"],
        inp.get("scenario_id"),
        PzzInputs.model_validate(inp.get("inputs") or {}),
        inp.get("history") or [],
    )


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
