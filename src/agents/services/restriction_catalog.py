from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from src.agents.services.service_entities.restriction_plan import (
    BufferRule,
    EntityRef,
    RestrictionPlan,
    RestrictionProvenance,
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
        history: list[dict] | None = None,
        normgraph_restrictions: list[dict[str, Any]] | None = None,
    ) -> RestrictionPlan:
        normgraph_restrictions = normgraph_restrictions or []
        cache_key = self._plan_cache_key(
            model,
            scenario_id,
            user_query,
            services_catalog,
            physical_objects_catalog,
            normgraph_restrictions,
        )
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        raw_plan = await self._request_plan(
            model,
            self._build_prompt(
                services_catalog,
                physical_objects_catalog,
                normgraph_restrictions,
            ),
            user_query=user_query,
            history=history,
        )
        plan = self.validate_and_canonicalize_plan(
            raw_plan,
            user_query,
            services_catalog,
            physical_objects_catalog,
        )
        plan = self._ground_normgraph_rules(plan, normgraph_restrictions)

        semantic_issues = self._find_semantic_issues(
            plan, user_query, normgraph_restrictions
        )
        if semantic_issues:
            raw_plan = await self._request_plan(
                model,
                self._build_semantic_repair_prompt(
                    user_query,
                    raw_plan,
                    semantic_issues,
                    services_catalog,
                    physical_objects_catalog,
                    normgraph_restrictions,
                ),
            )
            plan = self.validate_and_canonicalize_plan(
                raw_plan,
                user_query,
                services_catalog,
                physical_objects_catalog,
            )
            plan = self._ground_normgraph_rules(plan, normgraph_restrictions)

        unresolved_names = self._find_unresolved_names(
            raw_plan,
            services_catalog,
            physical_objects_catalog,
        )
        if unresolved_names:
            # Repair pass: full context is embedded in the system prompt,
            # history is not needed here.
            raw_plan = await self._request_plan(
                model,
                self._build_repair_prompt(
                    user_query,
                    raw_plan,
                    unresolved_names,
                    services_catalog,
                    physical_objects_catalog,
                    normgraph_restrictions,
                ),
            )
            plan = self.validate_and_canonicalize_plan(
                raw_plan,
                user_query,
                services_catalog,
                physical_objects_catalog,
            )
            plan = self._ground_normgraph_rules(plan, normgraph_restrictions)

        remaining_semantic_issues = self._find_semantic_issues(
            plan, user_query, normgraph_restrictions
        )
        if remaining_semantic_issues:
            plan = plan.model_copy(
                update={
                    "mode": RestrictionTaskMode.NEEDS_CLARIFICATION,
                    "target_entities": [],
                    "restriction_rules": [],
                    "clarification_question": (
                        "Не удалось однозначно определить целевые объекты проверки. "
                        "Уточните, какие объекты нужно проверить на пересечение с буферной зоной."
                    ),
                }
            )

        if plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION:
            plan = self._enrich_clarification(
                plan, services_catalog, physical_objects_catalog
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

    async def _request_plan(
        self,
        model: str,
        prompt: str,
        user_query: str | None = None,
        history: list[dict] | None = None,
        _retries: int = 2,
        _messages: list[dict] | None = None,
    ) -> RestrictionPlan:
        # On a repair retry ``_messages`` carries the full conversation so far
        # (system prompt + history + user query + the model's invalid answer +
        # the fix instruction). Rebuilding it from ``prompt`` alone — as the old
        # code did — discarded the invalid answer, the fix instruction AND the
        # user query, turning the "repair" into a blind re-roll.
        if _messages is not None:
            messages = _messages
        else:
            messages = [{"role": "system", "content": prompt}]
            if history:
                messages.extend(history)
            if user_query:
                messages.append({"role": "user", "content": user_query})
        response = await self.llm_client.chat(
            model=model,
            think=False,
            format=RestrictionPlan.model_json_schema(),
            options={
                "temperature": 0,
                "num_predict": 4096,
                "num_ctx": 16384,
            },
            messages=messages,
        )
        content = response["message"]["content"]
        logger.debug(f"LLM plan response [{model}]: {content}")
        try:
            return RestrictionPlan.model_validate_json(strip_json_fence(content))
        except (ValidationError, json.JSONDecodeError) as e:
            if _retries > 0:
                logger.warning(
                    f"LLM returned invalid plan JSON (retries left: {_retries}), asking model to fix it. Error: {e}"
                )
                messages.append({"role": "assistant", "content": content})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий ответ содержит невалидный или неполный JSON. "
                            "Верни тот же план целиком в виде валидного JSON без markdown и без пояснений. "
                            "Убедись, что JSON полный — все скобки и кавычки закрыты."
                        ),
                    }
                )
                return await self._request_plan(
                    model=model,
                    prompt=prompt,
                    _retries=_retries - 1,
                    _messages=messages,
                )
            logger.exception(e)
            raise ValueError("Model returned invalid restriction plan") from e

    @staticmethod
    def _plan_cache_key(
        model: str,
        scenario_id: int,
        user_query: str,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
        normgraph_restrictions: list[dict[str, Any]] | None = None,
    ) -> str:
        payload = {
            "model": model,
            "scenario_id": scenario_id,
            "user_query": normalize_name(user_query),
            "services_catalog": sorted(services_catalog),
            "physical_objects_catalog": sorted(physical_objects_catalog),
            "normgraph_restrictions": normgraph_restrictions or [],
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _build_prompt(
        services_catalog: list[str],
        physical_objects_catalog: list[str],
        normgraph_restrictions: list[dict[str, Any]] | None = None,
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
                    "origin": "normgraph | user",
                    "restriction_id": "string | null",
                    "provenance": "object | null",
                }
            ],
            "restriction_rules": [
                {
                    "source_name": "string",
                    "target_names": ["string"],
                    "title": "string",
                    "description": "string",
                    "origin": "normgraph | user",
                    "restriction_id": "string | null",
                    "provenance": "object | null",
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
            "original": "<запрос пользователя>",
        }
        return f"""
        Сформируй детерминированный план выполнения GIS-запроса.
        Верни только валидный JSON без markdown и без пояснений.

        Доступные сервисы:
        {services_catalog}

        Доступные физические объекты:
        {physical_objects_catalog}

        Релевантные канонические ограничения NormGraph:
        {json.dumps(normgraph_restrictions or [], ensure_ascii=False)}

        Формат ответа:
        {json.dumps(response_structure, ensure_ascii=False)}

        Правила:
        - Используй только имена из доступных списков, не придумывай новые.
        - Если пользователь использует обобщающую категорию, разверни её в конкретные имена из доступных списков.
        - Если обобщающая категория соответствует нескольким доступным именам, включи все такие имена.
        - Не возвращай обобщающую категорию, если в доступных списках есть более конкретные слои.
        - mode = "buffers_only", если пользователь просит построить/показать/получить только буферные зоны.
        - mode = "restrictions", если пользователь просит определить запрет, ограничение, затронутые объекты или применить буферы к другим объектам.
        - Формулировки «какие объекты попадают», «пересекают зону», «затронуты» или
          «выведи объекты» всегда означают mode = "restrictions": явно заполни
          target_entities и restriction_rules. Нельзя выбирать buffers_only и обещать
          отфильтровать целевые объекты позже — в этом режиме такая проверка не выполняется.
        - mode = "needs_clarification", если нет радиуса буфера или непонятно, от каких объектов строить буфер.
        - source_entities: объекты, от которых строятся буферы.
        - target_entities: объекты, на которые накладываются ограничения; для buffers_only оставь пустым списком.
        - Объекты, которые пользователь просит проверить, найти или вывести, всегда являются
          target_entities. Объект, от границы которого отсчитывается расстояние, является
          source_entities. Например, для «какие жилые дома ближе 50 м к лесу» источник —
          «лес», цель — «жилой дом», независимо от грамматического порядка слов в норме.
        - buffer_rules должны быть для каждого source_entities.
        - restriction_rules нужны только для mode = "restrictions".
        - selection_reasons: коротко объясни, почему выбран режим, источники, цели, радиусы и правила.
        - Пиши selection_reasons простым языком, без технических терминов.
        - Если пользователь не указал тип буфера, используй "round".
        - Если пользователь не указал title, сформируй короткое название из запроса.
        - confidence укажи от 0 до 1.
        - Если релевантное ограничение NormGraph однозначно соответствует слоям каталогов,
          используй subject как источник буфера, object как целевой слой, value.number как
          радиус в метрах. Скопируй id в restriction_id, укажи origin = "normgraph" и
          скопируй provenance. Текст extraction_text также включи в provenance.
        - Не изменяй расстояние из NormGraph и не приписывай ему другие источники или цели.
        - Если пользователь явно задал собственное расстояние в текущем запросе, можно создать
          временное правило с origin = "user", restriction_id = null и provenance = null.
        - Если пользователь сослался на СП, СНиП, ГОСТ, СанПиН, пункт документа или прямо
          попросил каноническое правило и оно присутствует в списке NormGraph, обязательно
          используй origin = "normgraph" и точный restriction_id. Повторённое пользователем
          нормативное расстояние не превращает норму во временное пользовательское правило.
        - Не показывай и не генерируй программный код.

        Строгие правила соответствия объектов:
        - Объект из каталога обязан семантически напрямую соответствовать тому, что запросил пользователь.
          Критерий: пользователь, прочитав название из каталога, должен согласиться, что именно это он и имел в виду.
        - Запрещено подставлять широкую категорию-родитель вместо конкретного запрошенного типа.
          Пример нарушения: пользователь просит «школы» или «продуктовые магазины» — выбирать «нежилые здания» нельзя,
          даже если школы или магазины формально являются нежилыми зданиями.
        - Если в каталоге нет объекта, явно и непосредственно соответствующего запросу, — не подбирай замену
          и не используй широкие или косвенно подходящие категории. Используй mode = "needs_clarification".
        - В clarification_question при mode = "needs_clarification" обязательно:
            1. Укажи, какие именно запрошенные объекты не найдены в каталоге.
            2. Перечисли все доступные объекты из обоих каталогов.
            3. Попроси пользователя переформулировать запрос целиком, включая все параметры
               (объекты, расстояние, тип анализа).
        """

    @staticmethod
    def _build_semantic_repair_prompt(
        user_query: str,
        plan: RestrictionPlan,
        issues: list[str],
        services_catalog: list[str],
        physical_objects_catalog: list[str],
        normgraph_restrictions: list[dict[str, Any]] | None = None,
    ) -> str:
        return f"""
        Исправь семантически противоречивый JSON-план GIS-запроса.
        Верни только валидный JSON той же структуры, без markdown и пояснений.

        Запрос пользователя:
        {user_query}

        Текущий план:
        {plan.model_dump_json(ensure_ascii=False)}

        Обнаруженные противоречия:
        {issues}

        Доступные сервисы:
        {services_catalog}

        Доступные физические объекты:
        {physical_objects_catalog}

        Релевантные канонические ограничения NormGraph:
        {json.dumps(normgraph_restrictions or [], ensure_ascii=False)}

        Если пользователь просит найти, какие объекты попадают в буфер, пересекают его
        или затронуты им, используй mode = "restrictions". Заполни target_entities и
        restriction_rules точными именами из каталогов. mode = "buffers_only" допустим
        только когда результатом должны быть сами буферные геометрии без проверки объектов.
        Объекты, которые пользователь просит проверить или вывести, должны быть целями;
        буфер строй вокруг объекта, от границы которого измеряется расстояние. Для запроса
        «проверь жилые дома у леса» source = «лес», target = «жилой дом».
        Для пользовательского временного правила используй origin = "user",
        restriction_id = null и provenance = null.
        """

    @staticmethod
    def _find_semantic_issues(
        plan: RestrictionPlan,
        user_query: str,
        normgraph_restrictions: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Detect plans that silently drop an explicitly requested intersection check."""

        query = normalize_name(user_query)
        intersection_intent = any(
            marker in query
            for marker in (
                "попада",
                "пересеч",
                "затронут",
                "провер",
                "какие объекты",
                "выведи объекты",
            )
        )
        if plan.mode == RestrictionTaskMode.BUFFERS_ONLY and intersection_intent:
            return [
                "Пользователь запросил проверку и вывод затронутых объектов, но план "
                "выбрал режим только построения буферов и потерял целевые объекты."
            ]
        if (
            plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION
            and intersection_intent
            and plan.source_entities
            and plan.buffer_rules
        ):
            return [
                "План нашёл источник и расстояние, но потерял целевые объекты или "
                "перепутал направление проверки. Объекты, которые пользователь просит "
                "проверить и вывести, должны быть target_entities; объект, от границы "
                "которого измеряется расстояние, должен быть source_entities."
            ]
        normative_intent = bool(
            re.search(
                r"\b(?:normgraph|сп\s*\d|снип|гост|санпин|санпин|пункт(?:а|е|у)?\s*\d)",
                query,
            )
        )
        if (
            normative_intent
            and normgraph_restrictions
            and not any(
                rule.origin == "normgraph" and rule.restriction_id
                for rule in [*plan.buffer_rules, *plan.restriction_rules]
            )
        ):
            return [
                "Пользователь запросил каноническое ограничение и NormGraph вернул "
                "подходящее правило, но план ошибочно создал пользовательское правило "
                "или потерял restriction_id."
            ]
        return []

    @staticmethod
    def _build_repair_prompt(
        user_query: str,
        plan: RestrictionPlan,
        unresolved_names: list[str],
        services_catalog: list[str],
        physical_objects_catalog: list[str],
        normgraph_restrictions: list[dict[str, Any]] | None = None,
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

        Релевантные канонические ограничения NormGraph:
        {json.dumps(normgraph_restrictions or [], ensure_ascii=False)}

        Правила исправления:
        - Используй только точные имена из доступных списков.
        - Не оставляй в плане обобщающие категории, если им соответствуют конкретные доступные имена.
        - Если одно обобщение соответствует нескольким доступным именам, добавь все подходящие имена.
        - Для каждого source entity должна быть отдельная buffer_rule с тем же радиусом, типом буфера и названием.
        - Для restriction_rules замени обобщающие source_name и target_names на конкретные доступные имена.
        - Обнови selection_reasons так, чтобы они объясняли уже исправленный выбор простым языком.
        - Для правил NormGraph сохрани точные origin, restriction_id, provenance и value.number
          из приведённого ограничения; не подменяй их значениями из памяти.

        Строгие правила соответствия объектов:
        - Объект из каталога обязан семантически напрямую соответствовать тому, что запросил пользователь.
          Критерий: пользователь, прочитав название из каталога, должен согласиться, что именно это он и имел в виду.
        - Запрещено заменять конкретный запрошенный тип широкой категорией-родителем.
          Пример нарушения: «нежилые здания» вместо «школы» — даже если школы формально являются нежилыми зданиями.
        - Если подходящего объекта нет в каталоге, используй mode = "needs_clarification".
          В clarification_question обязательно: укажи, каких именно объектов нет; перечисли все доступные объекты
          из обоих каталогов; попроси пользователя переформулировать запрос полностью, включая все параметры
          (объекты, расстояние, тип анализа).
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
                        origin=rule.origin,
                        restriction_id=rule.restriction_id,
                        provenance=rule.provenance,
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
                    origin=rule.origin,
                    restriction_id=rule.restriction_id,
                    provenance=rule.provenance,
                )
                for source_name in source_names
            )
        return result

    @classmethod
    def _ground_normgraph_rules(
        cls,
        plan: RestrictionPlan,
        restrictions: list[dict[str, Any]],
    ) -> RestrictionPlan:
        """Replace LLM-copied normative metadata with the exact retrieved values."""

        hits_by_id = {
            str(hit.get("id")): hit
            for hit in restrictions
            if isinstance(hit, dict) and hit.get("id")
        }

        def ground(rule: BufferRule | RestrictionRule):
            hit = hits_by_id.get(str(rule.restriction_id))
            if not hit:
                if rule.origin == "normgraph":
                    return None
                return rule
            provenance = cls._restriction_provenance(hit)
            updates: dict[str, Any] = {
                "origin": "normgraph",
                "restriction_id": str(hit["id"]),
                "provenance": provenance,
            }
            if isinstance(rule, BufferRule):
                updates["buffer_size"] = float(hit["value"]["number"])
            return rule.model_copy(update=updates)

        buffer_rules = [
            grounded
            for rule in plan.buffer_rules
            if (grounded := ground(rule)) is not None
        ]
        restriction_rules = [
            grounded
            for rule in plan.restriction_rules
            if (grounded := ground(rule)) is not None
        ]
        updates: dict[str, Any] = {
            "buffer_rules": buffer_rules,
            "restriction_rules": restriction_rules,
        }
        if not buffer_rules or (
            plan.mode == RestrictionTaskMode.RESTRICTIONS and not restriction_rules
        ):
            updates.update(
                mode=RestrictionTaskMode.NEEDS_CLARIFICATION,
                target_entities=[],
                clarification_question=(
                    "Не удалось однозначно связать найденное нормативное ограничение "
                    "с объектами сценария. Уточните источники, целевые объекты и, если "
                    "это временное правило, расстояние."
                ),
            )
        return plan.model_copy(
            update={
                **updates,
            }
        )

    @staticmethod
    def _restriction_provenance(hit: dict[str, Any]) -> RestrictionProvenance:
        raw = hit.get("provenance") or {}
        known = {
            "document_id": raw.get("doc_id"),
            "document_name": raw.get("name"),
            "document_version": raw.get("version"),
            "clause_id": raw.get("clause_node_id"),
            "clause_number": raw.get("numbering"),
            "breadcrumb": raw.get("breadcrumb"),
            "extraction_text": hit.get("extraction_text"),
        }
        extra = {
            key: value
            for key, value in raw.items()
            if key
            not in {
                "doc_id",
                "name",
                "version",
                "clause_node_id",
                "numbering",
                "breadcrumb",
            }
        }
        return RestrictionProvenance(**known, extra=extra)

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
    def _enrich_clarification(
        plan: RestrictionPlan,
        services_catalog: list[str],
        physical_objects_catalog: list[str],
    ) -> RestrictionPlan:
        base_question = (plan.clarification_question or "").strip()

        catalog_lines: list[str] = []
        if services_catalog:
            catalog_lines.append(
                "Доступные сервисы: " + ", ".join(services_catalog) + "."
            )
        if physical_objects_catalog:
            catalog_lines.append(
                "Доступные физические объекты: "
                + ", ".join(physical_objects_catalog)
                + "."
            )

        suffix_parts: list[str] = []
        if catalog_lines:
            suffix_parts.append("\n".join(catalog_lines))
        suffix_parts.append(
            "Пожалуйста, переформулируйте запрос полностью, включая все параметры"
            " (объекты, расстояние, тип анализа)."
        )
        suffix = "\n\n".join(suffix_parts)

        full_question = (
            f"{base_question}\n\n{suffix}".strip() if base_question else suffix
        )
        return plan.model_copy(update={"clarification_question": full_question})

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
