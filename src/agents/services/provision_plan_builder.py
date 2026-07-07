from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import ValidationError

from src.agents.services.restriction_catalog import (
    parse_catalog_prompt,
    strip_json_fence,
)
from src.agents.services.service_entities.provision_plan import (
    ProvisionPlan,
    ProvisionPlanMode,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


class ProvisionPlanBuilder:
    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    @staticmethod
    async def get_services_catalog(
        mcp_client: IduMcpClient,
        scenario_id: int,
    ) -> list[str]:
        prompt = await mcp_client.get_available_services_prompt(scenario_id)
        return parse_catalog_prompt(prompt)

    async def build_plan(
        self,
        model: str,
        user_query: str,
        services_catalog: list[str],
        history: list[dict] | None = None,
    ) -> ProvisionPlan:
        raw = await self._request_plan(model, user_query, services_catalog, history)
        raw = self._canonicalize_plan(raw, services_catalog)
        if (
            raw.mode == ProvisionPlanMode.NEEDS_CLARIFICATION
            and not raw.clarification_question
        ):
            raw = raw.model_copy(
                update={
                    "clarification_question": self._clarification_text(
                        None, services_catalog
                    )
                }
            )
        logger.info(f"Built provision plan: {raw.model_dump_json(ensure_ascii=False)}")
        return raw

    def _canonicalize_plan(
        self, plan: ProvisionPlan, services_catalog: list[str]
    ) -> ProvisionPlan:
        """Resolve every LLM-provided service name to its canonical catalog form."""
        if plan.mode in (ProvisionPlanMode.EFFECTS, ProvisionPlanMode.PROVISION):
            canonical = (
                self._find_canonical(plan.service_name, services_catalog)
                if plan.service_name
                else None
            )
            if canonical is None:
                return ProvisionPlan(
                    mode=ProvisionPlanMode.NEEDS_CLARIFICATION,
                    clarification_question=self._clarification_text(
                        plan.service_name, services_catalog
                    ),
                )
            return plan.model_copy(update={"service_name": canonical})

        if plan.mode == ProvisionPlanMode.SUMMARY:
            requested = self._canonicalize_names(plan.service_names, services_catalog)
            if plan.service_names and not requested:
                return ProvisionPlan(
                    mode=ProvisionPlanMode.NEEDS_CLARIFICATION,
                    clarification_question=self._clarification_text(
                        ", ".join(plan.service_names), services_catalog
                    ),
                )
            layer_names = self._canonicalize_names(
                plan.layer_service_names, services_catalog
            )
            return plan.model_copy(
                update={"service_names": requested, "layer_service_names": layer_names}
            )

        return plan

    def _canonicalize_names(self, names: list[str], catalog: list[str]) -> list[str]:
        result = []
        for name in names:
            canonical = self._find_canonical(name, catalog)
            if canonical and canonical not in result:
                result.append(canonical)
        return result

    async def _request_plan(
        self,
        model: str,
        user_query: str,
        services_catalog: list[str],
        history: list[dict] | None = None,
        _retries: int = 2,
    ) -> ProvisionPlan:
        messages: list[dict] = [
            {"role": "system", "content": self._build_prompt(services_catalog)},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        for attempt in range(_retries + 1):
            response = await self.llm_client.chat(
                model=model,
                options={"temperature": 0, "num_predict": 512},
                messages=messages,
            )
            content = response["message"]["content"]
            logger.debug(f"LLM provision plan response [{model}]: {content}")
            try:
                return ProvisionPlan.model_validate_json(strip_json_fence(content))
            except (ValidationError, json.JSONDecodeError) as exc:
                if attempt < _retries:
                    logger.warning(
                        f"LLM returned invalid provision plan JSON "
                        f"(retries left: {_retries - attempt - 1}): {exc}"
                    )
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Твой предыдущий ответ содержит невалидный JSON. "
                                "Верни только валидный JSON нужной структуры без markdown и пояснений."
                            ),
                        }
                    )
                else:
                    raise ValueError("Model returned invalid provision plan") from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _build_prompt(services_catalog: list[str]) -> str:
        response_structure = {
            "mode": "effects | provision | summary | list_services | needs_clarification",
            "service_name": "точное название сервиса из доступных или null",
            "service_names": ["названия сервисов для сводки или пустой список"],
            "layer_service_names": [
                "сервисы, для которых пользователь явно попросил слои/карту"
            ],
            "target_population": "целое число жителей или null",
            "clarification_question": "вопрос пользователю или null",
        }
        return f"""Ты классифицируешь запрос пользователя к сервису анализа обеспеченности \
городскими сервисами и извлекаешь параметры расчёта.

Доступные сервисы: {services_catalog}

Верни только валидный JSON без markdown и пояснений:
{json.dumps(response_structure, ensure_ascii=False)}

Режимы (mode):
- "list_services" — пользователь спрашивает, какие сервисы есть/доступны в проекте, \
сценарии или контексте. Примеры: «какие сервисы есть в проекте?», «что доступно для анализа?».
- "summary" — пользователь просит сводку/обзор обеспеченности по всем или нескольким сервисам, \
либо спрашивает, какими сервисами территория/проект/население обеспечены хуже или лучше всего. \
Примеры: «дай сводку по обеспеченности сервисами», «какими сервисами меньше всего обеспечен проект?».
- "provision" — вопрос о ТЕКУЩЕЙ обеспеченности одним конкретным сервисом, без упоминания \
эффектов, изменений или влияния проекта. Пример: «какая обеспеченность школами?».
- "effects" — вопрос об эффектах, изменениях или влиянии проекта на обеспеченность одним \
конкретным сервисом (сравнение до/после). Примеры: «как проект повлияет на обеспеченность школами?», \
«рассчитай эффекты обеспеченности школами». Если из запроса про один сервис непонятно, \
нужны текущая обеспеченность или эффекты — выбирай "effects".
- "needs_clarification" — запрос не подходит ни под один режим, упомянутый сервис отсутствует \
в доступных или запрос неоднозначен.

Правила заполнения полей:
- service_name — только для "effects" и "provision": ТОЧНОЕ название, скопированное дословно \
из списка доступных сервисов. Не используй форму слова из запроса пользователя. \
Пример: пользователь написал «школами» или «школ», в каталоге есть «Школы» — верни "Школы".
- service_names — только для "summary": список точных названий из доступных, если пользователь \
ограничил сводку конкретными сервисами; иначе пустой список (значит — по всем доступным).
- layer_service_names — только для "summary": точные названия сервисов, для которых пользователь \
ЯВНО попросил показать слои/карту/геометрию; иначе пустой список.
- Слова «проект», «сценарий», «территория», «население» обозначают контекст анализа, \
а не названия сервисов. Не сопоставляй их с каталогом сервисов.
- target_population — целевая численность населения для расчёта (для "effects", "provision" \
и "summary"): заполняй, только если пользователь явно указал число жителей в текущем запросе \
или ранее в диалоге; иначе null.
- clarification_question обязателен при mode = "needs_clarification": укажи, чего именно не хватает, \
перечисли все доступные сервисы и попроси уточнить запрос.
- Не придумывай сервисы, которых нет в доступных.

Правила работы с историей диалога:
- Сообщения переданы в хронологическом порядке, текущий запрос пользователя — последний.
- Если текущий запрос неполный (например «а теперь по школам», «пересчитай с населением 30000»), \
восстанавливай недостающие параметры — режим, сервис, население — из предыдущих сообщений.
- При противоречиях между сообщениями приоритет всегда у более поздних. \
Если население указывалось несколько раз — используй значение из самого последнего сообщения."""

    @staticmethod
    def _find_canonical(name: str, catalog: list[str]) -> str | None:
        normalized = name.casefold().strip()
        for item in catalog:
            if item.casefold().strip() == normalized:
                return item
        return None

    @staticmethod
    def _clarification_text(requested: str | None, services_catalog: list[str]) -> str:
        catalog_str = (
            ", ".join(services_catalog)
            if services_catalog
            else "нет доступных сервисов"
        )
        if requested:
            return (
                f"Сервис «{requested}» не найден в каталоге сценария. "
                f"Доступные сервисы: {catalog_str}. "
                "Пожалуйста, уточните запрос, указав один из доступных сервисов."
            )
        return (
            f"Не удалось определить, какой анализ обеспеченности нужен. "
            f"Доступные сервисы: {catalog_str}. "
            "Уточните: нужен список сервисов, сводка по всем сервисам "
            "или расчёт по конкретному сервису."
        )
