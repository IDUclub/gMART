"""Free-text answer stages: scored by a reviewer, with formal checks where possible.

The answer generators are methods of the agents' services. They are called on an
instance built without ``__init__`` (no Redis, MCP or Urban API clients) that has only
``llm_client`` set — the generators use nothing else — so the prompt is exactly the
one the agent sends. ``output.text`` is the raw model text; ``review.md`` shows it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from benchmarks.harness.stage import StageContext, stage
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.normgraph.normgraph_context import NormGraphContextBuilder
from src.agents.services.normgraph.normgraph_rag_service import NormGraphRagService
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.scenario_data.scenario_data_summary import (
    summarize_comparison,
)
from src.agents.services.service_entities.restriction_plan import RestrictionPlan


def norms_context(inp: dict[str, Any]) -> str:
    """The case's ``context`` string, or one built from NormGraph-shaped ``hits``."""
    if inp.get("context") is not None:
        return str(inp["context"])
    return NormGraphContextBuilder().build_context(
        inp.get("hits") or [],
        neighbors=inp.get("neighbors"),
        conflicts=inp.get("conflicts"),
    )


def _service(cls: type, ctx: StageContext):
    service = object.__new__(cls)
    service.llm_client = ctx.llm
    return service


def _answer_text(ctx: StageContext) -> str:
    return ctx.llm.calls[-1].content if ctx.llm.calls else ""


@stage(
    "norms.answer",
    kind="manual",
    description="Ответ по найденным нормативным ограничениям (с цитатами и источниками)",
    scope="NormGraphRagService._generate_answer, контекст — NormGraphContextBuilder.",
)
async def norms_answer(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    context = norms_context(inp)
    service = _service(NormGraphRagService, ctx)
    async for _ in service._generate_answer(
        ctx.model,
        inp["user_query"],
        context,
        ctx.temperature,
        inp.get("history") or [],
        1,
        revision_note=inp.get("revision_note"),
    ):
        pass
    return {"text": _answer_text(ctx), "context": context}


@stage(
    "restriction.explain",
    kind="manual",
    description="Объяснение пользователю выбранных параметров построения зон",
    scope="RestrictionParserService.generate_plan_explanation.",
)
async def restriction_explain(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    plan = RestrictionPlan.model_validate(
        {"original": inp["user_query"], **inp["plan"]}
    )
    service = _service(RestrictionParserService, ctx)
    async for _ in service.generate_plan_explanation(
        ctx.model, inp["user_query"], plan, ctx.temperature, inp.get("history")
    ):
        pass
    return {"text": _answer_text(ctx)}


_FALLBACK = "__computed_summary__"


@stage(
    "scenario.summary",
    kind="manual",
    description="Сводка сравнения показателей сценариев по посчитанным фактам",
    scope="scenario_data_summary.summarize_comparison; fallback_used=true — текст "
    "модели отброшен проверкой (числа не из фактов, обрыв, пусто).",
)
async def scenario_summary(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    final = await summarize_comparison(
        ctx.llm,
        ctx.model,
        rows=inp["rows"],
        column_labels=inp["column_labels"],
        counts=inp["counts"],
        missing=list(inp.get("missing") or []),
        fallback=_FALLBACK,
    )
    return {
        "text": _answer_text(ctx),
        "fallback_used": final == _FALLBACK,
        "final": None if final == _FALLBACK else final,
    }


@stage(
    "chat_title",
    kind="manual",
    description="Название чата по первому запросу",
    scope="BaseLlmService.generate_chat_title (одна попытка).",
)
async def chat_title(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    fake = SimpleNamespace(llm_client=ctx.llm)

    async def get_models():
        return []

    fake.get_models = get_models
    fake.generate_chat_title = (
        lambda *args, **kwargs: BaseLlmService.generate_chat_title(
            fake, *args, **kwargs
        )
    )
    title = await BaseLlmService.generate_chat_title(
        fake,
        ctx.model,
        inp["user_query"],
        inp.get("additional_instructions") or "",
        list(inp.get("existing_names") or []),
        max_retries=1,
    )
    return {"text": title}


@stage(
    "chat",
    kind="manual",
    description="Произвольный запрос с готовыми сообщениями (снимок промпта)",
    scope="Прямой вызов модели с messages/format/options из кейса.",
)
async def raw_chat(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    response = await ctx.llm.chat(
        ctx.model,
        inp["messages"],
        request_kind=inp.get("request_kind") or "benchmark.chat",
        think=inp.get("think", False),
        format=inp.get("format"),
        options={"temperature": ctx.temperature, **(inp.get("options") or {})},
    )
    text = response["message"]["content"] or ""
    output: dict[str, Any] = {"text": text}
    if inp.get("format") is not None:
        try:
            output["json"] = json.loads(text)
        except json.JSONDecodeError:
            output["json"] = None
    return output
