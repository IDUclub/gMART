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
        if raw.mode == ProvisionPlanMode.FOUND and raw.service_name:
            canonical = self._find_canonical(raw.service_name, services_catalog)
            if canonical is None:
                raw = ProvisionPlan(
                    mode=ProvisionPlanMode.NEEDS_CLARIFICATION,
                    clarification_question=self._clarification_text(
                        raw.service_name, services_catalog
                    ),
                )
            else:
                raw = raw.model_copy(update={"service_name": canonical})
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
            "mode": "found | needs_clarification",
            "service_name": "точное название сервиса из доступных или null",
            "target_population": "целое число жителей или null",
            "clarification_question": "вопрос пользователю или null",
        }
        return f"""Ты анализируешь запрос пользователя для расчёта эффектов обеспеченности городским сервисом.

Доступные сервисы: {services_catalog}

Верни только валидный JSON без markdown и пояснений:
{json.dumps(response_structure, ensure_ascii=False)}

Правила:
- mode = "found" если в запросе есть сервис, совпадающий с одним из доступных.
- mode = "needs_clarification" если подходящего сервиса нет в доступных или запрос неоднозначен.
- service_name должен точно совпадать с названием из доступных сервисов; null если не найдено.
- target_population — только если пользователь явно указал число жителей/население; иначе null.
- clarification_question обязателен при mode = "needs_clarification": укажи, чего именно не хватает, \
перечисли все доступные сервисы и попроси уточнить запрос.
- Не придумывай сервисы, которых нет в доступных."""

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
            f"В запросе не указан сервис для расчёта эффектов обеспеченности. "
            f"Доступные сервисы: {catalog_str}. "
            "Укажите название сервиса в запросе."
        )
