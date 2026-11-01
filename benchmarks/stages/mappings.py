"""Mapping stages: user words, file tags and columns onto catalogue entries."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from benchmarks.harness.stage import StageContext, stage
from src.agents.services.compilance.compliance_entity_mapping import (
    EntityMapper,
    EntityTarget,
    MappingProposal,
    TagGroup,
    signature_of,
)
from src.agents.services.compilance.compliance_scope import ComplianceScopeResolver
from src.agents.services.pzz.pzz_columns import detect_columns
from src.agents.services.scenario_data.scenario_data_type_mapper import (
    TypeMappingCandidate,
    TypeMappingRequest,
    UrbanTypeMapper,
)


def _targets(items: list[dict[str, Any]]) -> list[EntityTarget]:
    return [EntityTarget(item["name"], item["entity_type"]) for item in items]


@stage(
    "scenario.request",
    description="Намерение запроса к данным сценария: операция и тип сущности",
    scope="UrbanTypeMapper.classify_scenario_entity_request (без кавычек в запросе — "
    "иначе тип берётся кодом).",
)
async def scenario_request(inp: dict[str, Any], ctx: StageContext) -> Any:
    request = await UrbanTypeMapper(ctx.llm).classify_scenario_entity_request(
        ctx.model, inp["user_query"]
    )
    return {**request.model_dump(mode="json"), "types": request.types()}


@stage(
    "scenario.select",
    description="Выбор типа сущности из каталога сценария",
    scope="UrbanTypeMapper.select_scenario_entities; точное совпадение названия "
    "решается кодом без модели (поле llm_called).",
)
async def scenario_select(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    candidates = {
        key: {"domain": value["domain"], "name": value["name"]}
        for key, value in inp["candidates"].items()
    }
    selection = await UrbanTypeMapper(ctx.llm).select_scenario_entities(
        ctx.model,
        inp["user_query"],
        candidates,
        requested_type=inp.get("requested_type"),
    )
    chosen = selection.candidate
    return {
        "candidate": chosen,
        "name": candidates[chosen]["name"] if chosen in candidates else None,
        "llm_called": bool(ctx.llm.calls),
    }


@stage(
    "scenario.candidates",
    description="Отбор и независимая проверка справочных типов (select + assess)",
    scope="UrbanTypeMapper.resolve_candidates: два вызова модели.",
)
async def scenario_candidates(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    requests = [TypeMappingRequest(**item) for item in inp["requests"]]
    candidates = [TypeMappingCandidate(**item) for item in inp["candidates"]]
    resolution = await UrbanTypeMapper(ctx.llm).resolve_candidates(
        ctx.model,
        inp["user_query"],
        SimpleNamespace(objective=inp.get("objective") or inp["user_query"]),
        requests,
        candidates,
    )
    return {
        "accepted_ids": [item.candidate_id for item in resolution.accepted],
        "accepted_names": [item.name for item in resolution.accepted],
        "missing_values": resolution.missing_values,
        "complete": resolution.complete,
    }


@stage(
    "compliance.mapping",
    description="Сопоставление групп объектов файла (теги) с типами из норм",
    scope="EntityMapper._classify: только вызов модели, без табличных правил.",
)
async def compliance_mapping(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    groups = [
        TagGroup(
            signature=signature_of(item["tags"]),
            tags=item["tags"],
            count=int(item.get("count") or 1),
            layers=list(item.get("layers") or ["project"]),
            example=item.get("example") or {},
        )
        for item in inp["groups"]
    ]
    classified = await EntityMapper(ctx.llm)._classify(
        ctx.model, groups, _targets(inp["candidates"])
    )
    result = {}
    for item, group in zip(inp["groups"], groups):
        target, confidence, reason = classified.get(group.signature, (None, 0.0, None))
        result[str(item["id"])] = {
            "entity": target.name if target else None,
            "entity_type": target.entity_type if target else None,
            "confidence": confidence,
            "reason": reason,
        }
    return {"groups": result}


@stage(
    "compliance.mapping_reply",
    description="Разбор ответа пользователя на вопрос о сопоставлениях",
    scope="EntityMapper.resolve_reply; короткие «да/нет/2 — школа» разбираются кодом "
    "(поле llm_called).",
)
async def compliance_mapping_reply(
    inp: dict[str, Any], ctx: StageContext
) -> dict[str, Any]:
    candidates = _targets(inp["candidates"])
    by_name = {target.name: target for target in candidates}
    pending = [
        MappingProposal(
            signature=f"group-{number}",
            label=item.get("label") or item["proposed"],
            count=1,
            layers=["project"],
            roles={"project": 1},
            target=by_name.get(item["proposed"]),
            confidence=0.5,
            source="llm",
        )
        for number, item in enumerate(inp["pending"], start=1)
    ]
    reply = await EntityMapper(ctx.llm).resolve_reply(
        ctx.model, inp["reply"], pending, candidates
    )
    number_of = {item.signature: str(n) for n, item in enumerate(pending, start=1)}
    return {
        "kind": reply.kind,
        "overrides": {
            number_of[signature]: target.name if target else None
            for signature, target in reply.overrides.items()
        },
        "llm_called": bool(ctx.llm.calls),
    }


@stage(
    "compliance.scope",
    description="Темы и документы, которыми пользователь сузил проверку",
    scope="ComplianceScopeResolver._extract.",
)
async def compliance_scope(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await ComplianceScopeResolver(ctx.llm)._extract(
        ctx.model, inp["user_query"], inp.get("history")
    )


class _ResolutionsStub:
    """NormGraph ``resolve_entities`` answered from the case."""

    def __init__(self, resolutions: dict[str, list[dict[str, Any]]]) -> None:
        self.resolutions = resolutions

    async def resolve_entities(self, topics, limit=10):
        return [
            {"term": topic, "candidates": self.resolutions.get(topic, [])[:limit]}
            for topic in topics
        ]


@stage(
    "compliance.scope_entities",
    description="Выбор сущностей графа норм для темы проверки",
    scope="ComplianceScopeResolver._topic_entities с кандидатами NormGraph из кейса; "
    "если модель ничего не выбрала, код берёт кандидатов с совпадением имени.",
)
async def compliance_scope_entities(
    inp: dict[str, Any], ctx: StageContext
) -> dict[str, Any]:
    by_topic = await ComplianceScopeResolver(ctx.llm)._topic_entities(
        _ResolutionsStub(inp["candidates"]),
        ctx.model,
        inp["user_query"],
        list(inp["topics"]),
    )
    return {"by_topic": by_topic}


@stage(
    "pzz.columns",
    description="Определение колонок ВРИ/индекса/названия зоны по атрибутам слоя",
    scope="pzz_columns.detect_columns; точные алиасы колонок решаются кодом.",
)
async def pzz_columns(inp: dict[str, Any], ctx: StageContext) -> dict[str, Any]:
    features = [
        {"type": "Feature", "geometry": None, "properties": properties}
        for properties in inp["rows"]
    ]
    resolved = await detect_columns(
        ctx.llm,
        ctx.model,
        {"type": "FeatureCollection", "features": features},
        list(inp["targets"]),
        dict(inp.get("explicit") or {}),
    )
    return {"columns": resolved, "llm_called": bool(ctx.llm.calls)}
