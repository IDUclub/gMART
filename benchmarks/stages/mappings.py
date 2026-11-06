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
    return await UrbanTypeMapper(ctx.llm).classify_scenario_entity_request(
        ctx.model, inp["user_query"]
    )


@stage(
    "scenario.select",
    description="Выбор типа сущности из каталога сценария",
    scope="UrbanTypeMapper.select_scenario_entities; точное совпадение названия "
    "решается кодом без модели (поле llm_called).",
)
async def scenario_select(inp: dict[str, Any], ctx: StageContext) -> Any:
    candidates = {
        key: {"domain": value["domain"], "name": value["name"]}
        for key, value in inp["candidates"].items()
    }
    return await UrbanTypeMapper(ctx.llm).select_scenario_entities(
        ctx.model,
        inp["user_query"],
        candidates,
        requested_type=inp.get("requested_type"),
    )


def _type_candidates(inp: dict[str, Any]) -> list[TypeMappingCandidate]:
    return [TypeMappingCandidate(**item) for item in inp["candidates"]]


def _acquisition(inp: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(objective=inp.get("objective") or inp["user_query"])


@stage(
    "scenario.candidates.select",
    description="Отбор справочных типов из кандидатов под запрос",
    scope="UrbanTypeMapper._select_candidates — первый из двух вызовов resolve_candidates.",
)
async def scenario_candidates_select(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await UrbanTypeMapper(ctx.llm)._select_candidates(
        ctx.model, inp["user_query"], _acquisition(inp), _type_candidates(inp)
    )


@stage(
    "scenario.candidates.assess",
    description="Независимая проверка выбранных справочных типов",
    scope="UrbanTypeMapper._assess_candidates — второй вызов resolve_candidates; "
    "на проверку подаются все кандидаты кейса, включая неверные.",
)
async def scenario_candidates_assess(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await UrbanTypeMapper(ctx.llm)._assess_candidates(
        ctx.model, inp["user_query"], _acquisition(inp), _type_candidates(inp)
    )


@stage(
    "compliance.mapping",
    description="Сопоставление групп объектов файла (теги) с типами из норм",
    scope="EntityMapper._classify: только вызов модели, без табличных правил.",
)
async def compliance_mapping(inp: dict[str, Any], ctx: StageContext) -> Any:
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
    return await EntityMapper(ctx.llm)._classify(
        ctx.model, groups, _targets(inp["candidates"])
    )


@stage(
    "compliance.mapping_reply",
    description="Разбор ответа пользователя на вопрос о сопоставлениях",
    scope="EntityMapper.resolve_reply; короткие «да/нет/2 — школа» разбираются кодом "
    "(поле llm_called).",
)
async def compliance_mapping_reply(inp: dict[str, Any], ctx: StageContext) -> Any:
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
    return await EntityMapper(ctx.llm).resolve_reply(
        ctx.model, inp["reply"], pending, candidates
    )


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
    scope="ComplianceScopeResolver._topic_entities с кандидатами NormGraph из кейса.",
)
async def compliance_scope_entities(inp: dict[str, Any], ctx: StageContext) -> Any:
    return await ComplianceScopeResolver(ctx.llm)._topic_entities(
        _ResolutionsStub(inp["candidates"]),
        ctx.model,
        inp["user_query"],
        list(inp["topics"]),
    )


@stage(
    "pzz.columns",
    description="Определение колонок ВРИ/индекса/названия зоны по атрибутам слоя",
    scope="pzz_columns.detect_columns; точные алиасы колонок решаются кодом.",
)
async def pzz_columns(inp: dict[str, Any], ctx: StageContext) -> Any:
    features = [
        {"type": "Feature", "geometry": None, "properties": properties}
        for properties in inp["rows"]
    ]
    return await detect_columns(
        ctx.llm,
        ctx.model,
        {"type": "FeatureCollection", "features": features},
        list(inp["targets"]),
        dict(inp.get("explicit") or {}),
    )
