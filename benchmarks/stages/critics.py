"""Critic and judge stages: a verdict on a drafted answer, compared with the known one."""

from __future__ import annotations

from typing import Any

from benchmarks.harness.stage import StageContext, stage
from benchmarks.stages.answers import norms_context
from src.agents.services.normgraph.normgraph_reasoning import (
    NormGraphAnswerCritic,
    _request_json,
)
from src.agents.services.scenario_data.scenario_data_evaluator import (
    ScenarioDataEvaluator,
)
from src.agents.services.service_entities.normgraph_plan import NormGraphCriticVerdict


@stage(
    "norms.review",
    description="Критик ответа по нормам: обоснован ли ответ контекстом",
    scope="Вызов модели NormGraphAnswerCritic без кодовой предпроверки чисел "
    "(ungrounded_quantities), чтобы измерить саму модель.",
)
async def norms_review(inp: dict[str, Any], ctx: StageContext) -> Any:
    messages = [
        {"role": "system", "content": NormGraphAnswerCritic._prompt()},
        {
            "role": "user",
            "content": NormGraphAnswerCritic._payload(
                inp["user_query"], norms_context(inp), inp["answer"]
            ),
        },
    ]
    return await _request_json(
        ctx.llm,
        ctx.model,
        messages,
        NormGraphCriticVerdict,
        request_kind="norms.review",
    )


@stage(
    "scenario.judge",
    description="Судья ответа по данным сценария: достаточен ли ответ",
    scope="ScenarioDataEvaluator._judge без детерминированных проверок перед ним.",
)
async def scenario_judge(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await ScenarioDataEvaluator(ctx.llm)._judge(
        ctx.model,
        inp["user_query"],
        inp["observations"],
        inp["answer"],
        required_output=inp.get("required_output"),
    )
