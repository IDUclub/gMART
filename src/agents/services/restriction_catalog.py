from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from src.agents.services.service_entities.restriction_plan import (
    BufferRule,
    EntityRef,
    RestrictionPlan,
    RestrictionRule,
    RestrictionTaskMode,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


def normalize_name(name: str) -> str:
    return " ".join(name.casefold().strip().split())


def strip_json_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```")
        content = content.removesuffix("```")
    return content.strip()


def parse_catalog_prompt(prompt: str) -> list[str]:
    if ":" in prompt:
        prompt = prompt.split(":", 1)[1]
    return [item.strip() for item in prompt.split(",") if item.strip()]


class RestrictionPlanBuilder:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self._plan_cache: dict[str, RestrictionPlan] = {}

    @staticmethod
    async def get_entity_catalogs(
        mcp_client: IduMcpClient,
        scenario_id: int,
    ) -> tuple[list[str], list[str]]:
        services_prompt = await mcp_client.get_available_services_prompt(scenario_id)
        physical_objects_prompt = (
            await mcp_client.get_available_physical_objects_prompt(scenario_id)
        )
        return parse_catalog_prompt(services_prompt), parse_catalog_prompt(
            physical_objects_prompt
        )

    async def build_plan(
        self,
        model: str,
        user_query: str,
        scenario_id: int,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> RestrictionPlan:
        cache_key = self._plan_cache_key(
            model,
            scenario_id,
            user_query,
            services_catalog,
            physical_objects_catalog,
        )
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        raw_plan = await self._request_plan(
            model,
            self._build_prompt(user_query, services_catalog, physical_objects_catalog),
        )
        plan = self.validate_and_canonicalize_plan(
            raw_plan,
            user_query,
            services_catalog,
            physical_objects_catalog,
        )

        unresolved_names = self._find_unresolved_names(
            raw_plan,
            services_catalog,
            physical_objects_catalog,
        )
        if unresolved_names:
            raw_plan = await self._request_plan(
                model,
                self._build_repair_prompt(
                    user_query,
                    raw_plan,
                    unresolved_names,
                    services_catalog,
                    physical_objects_catalog,
                ),
            )
            plan = self.validate_and_canonicalize_plan(
                raw_plan,
                user_query,
                services_catalog,
                physical_objects_catalog,
            )

        self._plan_cache[cache_key] = plan
        logger.info(
            f"Built restriction plan: {plan.model_dump_json(ensure_ascii=False)}"
        )
        return plan

    def validate_and_canonicalize_plan(
        self,
        plan: RestrictionPlan,
        user_query: str,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> RestrictionPlan:
        catalogs = {
            "service": services_catalog,
            "physical_object": physical_objects_catalog,
        }
        source_candidates, target_candidates = self._collect_entity_candidates(
            plan, catalogs
        )
        source_entities, source_aliases = self._canonicalize_entities(
            source_candidates, catalogs
        )
        target_entities, target_aliases = self._canonicalize_entities(
            target_candidates, catalogs
        )
        aliases = self._build_alias_map(
            plan,
            catalogs,
            source_entities + target_entities,
            source_aliases | target_aliases,
        )

        buffer_rules = self._canonicalize_buffer_rules(plan.buffer_rules, aliases)
        restriction_rules = self._canonicalize_restriction_rules(
            plan.restriction_rules, aliases
        )
        mode, clarification = self._validate_mode(
            plan,
            source_entities,
            target_entities,
            buffer_rules,
            restriction_rules,
        )

        return RestrictionPlan(
            mode=mode,
            source_entities=source_entities,
            target_entities=(
                target_entities if mode == RestrictionTaskMode.RESTRICTIONS else []
            ),
            buffer_rules=buffer_rules,
            restriction_rules=(
                restriction_rules if mode == RestrictionTaskMode.RESTRICTIONS else []
            ),
            selection_reasons=plan.selection_reasons,
            confidence=plan.confidence,
            clarification_question=clarification,
            original=user_query,
        )

    async def _request_plan(self, model: str, prompt: str) -> RestrictionPlan:
        response = await self.llm_client.chat(
            model=model,
            options={"temperature": 0},
            messages=[{"role": "system", "content": prompt}],
        )
        try:
            return RestrictionPlan.model_validate_json(
                strip_json_fence(response["message"]["content"])
            )
        except (ValidationError, json.JSONDecodeError) as e:
            logger.exception(e)
            raise ValueError("Model returned invalid restriction plan") from e

    @staticmethod
    def _plan_cache_key(
        model: str,
        scenario_id: int,
        user_query: str,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> str:
        payload = {
            "model": model,
            "scenario_id": scenario_id,
            "user_query": normalize_name(user_query),
            "services_catalog": sorted(services_catalog),
            "physical_objects_catalog": sorted(physical_objects_catalog),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _build_prompt(
        user_query: str,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> str:
        response_structure = {
            "mode": "buffers_only | restrictions | needs_clarification",
            "source_entities": [
                {"name": "string", "entity_type": "service | physical_object"}
            ],
            "target_entities": [
                {"name": "string", "entity_type": "service | physical_object"}
            ],
            "buffer_rules": [
                {
                    "source_name": "string",
                    "buffer_size": 300,
                    "buffer_type": "round | flat | square",
                    "title": "string",
                }
            ],
            "restriction_rules": [
                {
                    "source_name": "string",
                    "target_names": ["string"],
                    "title": "string",
                    "description": "string",
                }
            ],
            "selection_reasons": [
                {
                    "step": "mode | source_entities | target_entities | buffer_rules | restriction_rules",
                    "reason": "string",
                }
            ],
            "confidence": 0.0,
            "clarification_question": None,
            "original": user_query,
        }
        return f"""
        Сформируй детерминированный план выполнения GIS-запроса.
        Верни только валидный JSON без markdown и без пояснений.

        Доступные сервисы:
        {services_catalog}

        Доступные физические объекты:
        {physical_objects_catalog}

        Формат ответа:
        {json.dumps(response_structure, ensure_ascii=False)}

        Правила:
        - Используй только имена из доступных списков, не придумывай новые.
        - Если пользователь использует обобщающую категорию, разверни её в конкретные имена из доступных списков.
        - Если обобщающая категория соответствует нескольким доступным именам, включи все такие имена.
        - Не возвращай обобщающую категорию, если в доступных списках есть более конкретные слои.
        - mode = "buffers_only", если пользователь просит построить/показать/получить только буферные зоны.
        - mode = "restrictions", если пользователь просит определить запрет, ограничение, затронутые объекты или применить буферы к другим объектам.
        - mode = "needs_clarification", если нет радиуса буфера или непонятно, от каких объектов строить буфер.
        - source_entities: объекты, от которых строятся буферы.
        - target_entities: объекты, на которые накладываются ограничения; для buffers_only оставь пустым списком.
        - buffer_rules должны быть для каждого source_entities.
        - restriction_rules нужны только для mode = "restrictions".
        - selection_reasons: коротко объясни, почему выбран режим, источники, цели, радиусы и правила.
        - Пиши selection_reasons простым языком, без технических терминов.
        - Если пользователь не указал тип буфера, используй "round".
        - Если пользователь не указал title, сформируй короткое название из запроса.
        - confidence укажи от 0 до 1.

        Запрос пользователя:
        {user_query}
        """

    @staticmethod
    def _build_repair_prompt(
        user_query: str,
        plan: RestrictionPlan,
        unresolved_names: list[str],
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> str:
        return f"""
        Исправь JSON-план GIS-запроса.
        Верни только валидный JSON той же структуры, без markdown и без пояснений.

        Запрос пользователя:
        {user_query}

        Текущий план:
        {plan.model_dump_json(ensure_ascii=False)}

        В текущем плане есть имена, которых нет в доступных списках:
        {unresolved_names}

        Доступные сервисы:
        {services_catalog}

        Доступные физические объекты:
        {physical_objects_catalog}

        Правила исправления:
        - Используй только точные имена из доступных списков.
        - Не оставляй в плане обобщающие категории, если им соответствуют конкретные доступные имена.
        - Если одно обобщение соответствует нескольким доступным именам, добавь все подходящие имена.
        - Для каждого source entity должна быть отдельная buffer_rule с тем же радиусом, типом буфера и названием.
        - Для restriction_rules замени обобщающие source_name и target_names на конкретные доступные имена.
        - Обнови selection_reasons так, чтобы они объясняли уже исправленный выбор простым языком.
        - Если подходящих имён в доступных списках нет, верни mode = "needs_clarification".
        """

    def _collect_entity_candidates(
        self,
        plan: RestrictionPlan,
        catalogs: dict[str, list[str]],
    ) -> tuple[list[EntityRef], list[EntityRef]]:
        source_candidates = list(plan.source_entities)
        target_candidates = list(plan.target_entities)
        for rule in plan.buffer_rules:
            source_candidates.extend(
                self._infer_entity_refs(rule.source_name, catalogs)
            )
        for rule in plan.restriction_rules:
            source_candidates.extend(
                self._infer_entity_refs(rule.source_name, catalogs)
            )
            for target_name in rule.target_names:
                target_candidates.extend(self._infer_entity_refs(target_name, catalogs))
        return source_candidates, target_candidates

    def _find_unresolved_names(
        self,
        plan: RestrictionPlan,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> list[str]:
        catalogs = {
            "service": services_catalog,
            "physical_object": physical_objects_catalog,
        }
        unresolved = []
        for entity in [*plan.source_entities, *plan.target_entities]:
            if not self._canonical_name(entity.name, catalogs[entity.entity_type]):
                unresolved.append(entity.name)
        for rule_name in self._iter_rule_names(plan):
            if not self._exists_in_any_catalog(rule_name, catalogs):
                unresolved.append(rule_name)
        return list(dict.fromkeys(unresolved))

    def _exists_in_any_catalog(
        self,
        name: str,
        catalogs: dict[str, list[str]],
    ) -> bool:
        return any(self._canonical_name(name, catalog) for catalog in catalogs.values())

    def _canonicalize_entities(
        self,
        entities: list[EntityRef],
        catalogs: dict[str, list[str]],
    ) -> tuple[list[EntityRef], dict[str, list[str]]]:
        result = []
        aliases: dict[str, list[str]] = {}
        seen = set()
        for entity in entities:
            matches = self._resolve_catalog_names(
                entity.name, catalogs[entity.entity_type]
            )
            if not matches:
                continue
            aliases[normalize_name(entity.name)] = matches
            for canonical in matches:
                key = (entity.entity_type, normalize_name(canonical))
                if key in seen:
                    continue
                seen.add(key)
                result.append(EntityRef(name=canonical, entity_type=entity.entity_type))
        return result, aliases

    def _build_alias_map(
        self,
        plan: RestrictionPlan,
        catalogs: dict[str, list[str]],
        entities: list[EntityRef],
        aliases: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        for alias_name in self._iter_rule_names(plan):
            inferred_names = [
                entity.name for entity in self._infer_entity_refs(alias_name, catalogs)
            ]
            if inferred_names:
                normalized_alias = normalize_name(alias_name)
                aliases[normalized_alias] = list(
                    dict.fromkeys([*aliases.get(normalized_alias, []), *inferred_names])
                )
        for entity in entities:
            aliases[normalize_name(entity.name)] = [entity.name]
        return aliases

    @staticmethod
    def _iter_rule_names(plan: RestrictionPlan):
        yield from (rule.source_name for rule in plan.buffer_rules)
        for rule in plan.restriction_rules:
            yield rule.source_name
            yield from rule.target_names

    def _canonicalize_buffer_rules(
        self,
        rules: list[BufferRule],
        aliases: dict[str, list[str]],
    ) -> list[BufferRule]:
        result = []
        seen = set()
        for rule in rules:
            for source_name in aliases.get(normalize_name(rule.source_name), []):
                if source_name in seen:
                    continue
                seen.add(source_name)
                result.append(
                    BufferRule(
                        source_name=source_name,
                        buffer_size=rule.buffer_size,
                        buffer_type=rule.buffer_type,
                        title=rule.title,
                    )
                )
        return result

    def _canonicalize_restriction_rules(
        self,
        rules: list[RestrictionRule],
        aliases: dict[str, list[str]],
    ) -> list[RestrictionRule]:
        result = []
        for rule in rules:
            source_names = aliases.get(normalize_name(rule.source_name), [])
            target_names = list(
                dict.fromkeys(
                    target_name
                    for target in rule.target_names
                    for target_name in aliases.get(normalize_name(target), [])
                )
            )
            if not source_names or not target_names:
                continue
            result.extend(
                RestrictionRule(
                    source_name=source_name,
                    target_names=target_names,
                    title=rule.title,
                    description=rule.description,
                )
                for source_name in source_names
            )
        return result

    @staticmethod
    def _validate_mode(
        plan: RestrictionPlan,
        source_entities: list[EntityRef],
        target_entities: list[EntityRef],
        buffer_rules: list[BufferRule],
        restriction_rules: list[RestrictionRule],
    ) -> tuple[RestrictionTaskMode, str | None]:
        if not source_entities or not buffer_rules:
            return (
                RestrictionTaskMode.NEEDS_CLARIFICATION,
                plan.clarification_question
                or "Уточните, от каких объектов и на каком расстоянии нужно построить буферы.",
            )
        if plan.mode == RestrictionTaskMode.RESTRICTIONS and (
            not target_entities or not restriction_rules
        ):
            return (
                RestrictionTaskMode.NEEDS_CLARIFICATION,
                plan.clarification_question
                or "Уточните, на какие объекты должны накладываться ограничения.",
            )
        return plan.mode, plan.clarification_question

    @staticmethod
    def _canonical_name(name: str, catalog: list[str]) -> str | None:
        normalized_catalog = {normalize_name(item): item for item in catalog}
        return normalized_catalog.get(normalize_name(name))

    def _resolve_catalog_names(self, name: str, catalog: list[str]) -> list[str]:
        canonical = self._canonical_name(name, catalog)
        if canonical:
            return [canonical]
        return []

    def _infer_entity_refs(
        self,
        name: str,
        catalogs: dict[str, list[str]],
    ) -> list[EntityRef]:
        return [
            EntityRef(name=matched_name, entity_type=entity_type)
            for entity_type, catalog in catalogs.items()
            for matched_name in self._resolve_catalog_names(name, catalog)
        ]
