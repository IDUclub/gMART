from __future__ import annotations

import json
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

from src.agents.services.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.dvd_plan import (
    CriticVerdict,
    RetrievalPlan,
)

T = TypeVar("T", bound=BaseModel)

_LIMIT_MIN, _LIMIT_MAX = 1, 20
_CONTEXT_HEIGHT_MIN, _CONTEXT_HEIGHT_MAX = 0, 5


async def _request_json(
    llm_client,
    model: str,
    messages: list[dict],
    model_cls: type[T],
    retries: int = 2,
) -> T:
    """
    Ask the LLM for a JSON object and parse it into ``model_cls``.

    Mirrors the structured-output convention used by ProvisionPlanBuilder: temperature 0,
    strip markdown fences, retry by feeding the invalid response back to the model.
    """
    for attempt in range(retries + 1):
        response = await llm_client.chat(
            model=model,
            options={"temperature": 0, "num_predict": 512},
            messages=messages,
        )
        content = response["message"]["content"]
        logger.debug(f"LLM {model_cls.__name__} response [{model}]: {content}")
        try:
            return model_cls.model_validate_json(strip_json_fence(content))
        except (ValidationError, json.JSONDecodeError) as exc:
            if attempt < retries:
                logger.warning(
                    f"LLM returned invalid {model_cls.__name__} JSON "
                    f"(retries left: {retries - attempt - 1}): {exc}"
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий ответ содержит невалидный JSON. "
                            "Верни только валидный JSON нужной структуры без markdown и пояснений."
                        ),
                    },
                ]
            else:
                raise ValueError(
                    f"Model returned invalid {model_cls.__name__} JSON"
                ) from exc
    raise AssertionError("unreachable")


class RetrievalPlanner:
    """Builds a :class:`RetrievalPlan` for a RAG round via structured LLM output."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def build_plan(
        self,
        model: str,
        user_query: str,
        history: list[dict] | None = None,
        prev_critique: str | None = None,
        prev_query: str | None = None,
    ) -> RetrievalPlan:
        messages: list[dict] = [
            {"role": "system", "content": self._prompt(prev_critique, prev_query)},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        plan = await _request_json(self.llm_client, model, messages, RetrievalPlan)
        plan = self._clamp(plan, user_query)
        logger.info(f"DVD retrieval plan: {plan.model_dump_json(ensure_ascii=False)}")
        return plan

    @staticmethod
    def _clamp(plan: RetrievalPlan, user_query: str) -> RetrievalPlan:
        return plan.model_copy(
            update={
                "search_query": (plan.search_query or "").strip() or user_query,
                "limit": min(max(plan.limit, _LIMIT_MIN), _LIMIT_MAX),
                "context_height": min(
                    max(plan.context_height, _CONTEXT_HEIGHT_MIN), _CONTEXT_HEIGHT_MAX
                ),
            }
        )

    @staticmethod
    def _prompt(prev_critique: str | None, prev_query: str | None) -> str:
        structure = {
            "search_query": "строка для векторного поиска",
            "kind": "text | table | all",
            "limit": 10,
            "context_height": 1,
        }
        prompt = f"""Ты планируешь поиск по векторной базе нормативных документов \
(градостроительство и городское планирование).
По вопросу пользователя сформируй параметры поиска. \
Верни только валидный JSON без markdown и пояснений:
{json.dumps(structure, ensure_ascii=False)}

Правила:
- search_query — краткий поисковый запрос на русском, отражающий суть вопроса \
(ключевые термины, нормативная лексика). Не копируй вопрос дословно — выдели суть.
- kind = "table" если вопрос про числовые нормативы, показатели или таблицы; \
"text" для текстовых формулировок, определений и требований; "all" если неясно.
- limit — сколько фрагментов извлечь (целое 1–20). Больше для широких/обзорных \
вопросов, меньше для точечных.
- context_height — сколько соседних фрагментов прикреплять к каждому найденному \
(целое 0–5). Больше (2–3), когда важен контекст вокруг (определения, процедуры, \
перечни, ссылки на смежные пункты); 0–1 для точечных фактов."""
        if prev_critique:
            prompt += f"""

Предыдущая попытка ответа не прошла самопроверку.
Замечание критика: {prev_critique}
Предыдущий поисковый запрос: «{prev_query}».
Сформируй ИНОЙ, улучшенный поисковый запрос (синонимы, иные формулировки, \
официальная терминология); при необходимости измени kind, limit или context_height."""
        return prompt


class AnswerCritic:
    """Reviews a drafted answer against the retrieved context via structured LLM output."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def review(
        self,
        model: str,
        user_query: str,
        context: str,
        answer: str,
    ) -> CriticVerdict:
        messages: list[dict] = [
            {"role": "system", "content": self._prompt()},
            {"role": "user", "content": self._payload(user_query, context, answer)},
        ]
        try:
            verdict = await _request_json(
                self.llm_client, model, messages, CriticVerdict
            )
        except ValueError:
            # If the critic itself fails to produce valid JSON, accept the draft
            # rather than loop forever.
            logger.warning("Critic produced invalid JSON, accepting draft by default")
            return CriticVerdict(satisfied=True)
        logger.info(
            f"DVD critic verdict: {verdict.model_dump_json(ensure_ascii=False)}"
        )
        return verdict

    @staticmethod
    def _prompt() -> str:
        structure = {
            "satisfied": "true | false",
            "critique": "кратко: что не так с ответом (пусто, если всё хорошо)",
            "refined_search_query": "улучшенный поисковый запрос или null",
        }
        return f"""Ты — строгий рецензент ответов ассистента по нормативной документации.
Оцени, полностью ли ответ обоснован приведёнными фрагментами и отвечает ли на вопрос. \
Верни только валидный JSON без markdown:
{json.dumps(structure, ensure_ascii=False)}

Критерии отказа (satisfied = false):
- В ответе есть утверждения, не подтверждённые фрагментами (галлюцинации).
- Ответ неполный или не отвечает на вопрос пользователя.
- Не указаны источники (документ, редакция, номер пункта), хотя они есть во фрагментах.
- Во фрагментах недостаточно данных — тогда обязательно предложи refined_search_query \
для нового поиска.

Если ответ корректен, полон и обоснован — satisfied = true, critique пустой, \
refined_search_query = null. Будь требователен по сути, но не придирайся к стилю."""

    @staticmethod
    def _payload(user_query: str, context: str, answer: str) -> str:
        ctx = context or "(релевантные фрагменты не найдены)"
        return (
            f"Вопрос пользователя:\n{user_query}\n\n"
            f"Доступные фрагменты:\n{ctx}\n\n"
            f"Ответ ассистента для проверки:\n{answer}"
        )
