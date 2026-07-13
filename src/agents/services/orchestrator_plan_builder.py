from __future__ import annotations

import json

from loguru import logger
from pydantic import ValidationError

from src.agents.services.orchestrator_catalog import AgentCatalogEntry
from src.agents.services.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.orchestrator_plan import (
    MAX_PLAN_STEPS,
    OrchestratorPlan,
    OrchestratorPlanMode,
)


class OrchestratorPlanBuilder:
    """
    LLM planner that maps a user request onto a sequential plan of agent steps.

    Mirrors ``ProvisionPlanBuilder``: a deterministic (temperature 0) chat call
    with a Russian system prompt embedding the agent catalogue and a JSON
    skeleton, a self-repair retry loop on invalid JSON, and a canonicalization
    pass that downgrades plans referencing unavailable agents to a clarification.
    """

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def build_plan(
        self,
        model: str,
        user_query: str,
        agents: list[AgentCatalogEntry],
        history: list[dict] | None = None,
    ) -> OrchestratorPlan:
        plan = await self._request_plan(model, user_query, agents, history)
        plan = self._canonicalize_plan(plan, agents)
        if (
            plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION
            and not plan.clarification_question
        ):
            plan = plan.model_copy(
                update={"clarification_question": self._clarification_text(agents)}
            )
        logger.info(
            f"Built orchestration plan: {plan.model_dump_json(ensure_ascii=False)}"
        )
        return plan

    def _canonicalize_plan(
        self, plan: OrchestratorPlan, agents: list[AgentCatalogEntry]
    ) -> OrchestratorPlan:
        """Downgrade plans referencing unavailable agents; cap the step count."""
        if plan.mode != OrchestratorPlanMode.EXECUTE:
            return plan
        available_keys = {entry.key for entry in agents}
        if any(step.agent not in available_keys for step in plan.steps):
            return OrchestratorPlan(
                mode=OrchestratorPlanMode.NEEDS_CLARIFICATION,
                clarification_question=self._clarification_text(agents),
            )
        if len(plan.steps) > MAX_PLAN_STEPS:
            return plan.model_copy(update={"steps": plan.steps[:MAX_PLAN_STEPS]})
        return plan

    async def _request_plan(
        self,
        model: str,
        user_query: str,
        agents: list[AgentCatalogEntry],
        history: list[dict] | None = None,
        _retries: int = 2,
    ) -> OrchestratorPlan:
        messages: list[dict] = [
            {"role": "system", "content": self._build_prompt(agents)},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        for attempt in range(_retries + 1):
            response = await self.llm_client.chat(
                model=model,
                options={"temperature": 0, "num_predict": 768},
                messages=messages,
            )
            content = response["message"]["content"]
            logger.debug(f"LLM orchestration plan response [{model}]: {content}")
            try:
                return OrchestratorPlan.model_validate_json(strip_json_fence(content))
            except (ValidationError, json.JSONDecodeError) as exc:
                if attempt < _retries:
                    logger.warning(
                        f"LLM returned invalid orchestration plan JSON "
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
                    raise ValueError(
                        "Model returned invalid orchestration plan"
                    ) from exc
        raise AssertionError("unreachable")

    @staticmethod
    def _build_prompt(agents: list[AgentCatalogEntry]) -> str:
        agents_block = "\n".join(
            f'- "{entry.key}" — {entry.title}. {entry.description} '
            f"Примеры запросов: {'; '.join(f'«{example}»' for example in entry.examples)}."
            for entry in agents
        )
        response_structure = {
            "mode": "execute | needs_clarification",
            "steps": [
                {
                    "agent": "ключ агента из списка доступных",
                    "task": "самодостаточная формулировка подзадачи на русском",
                }
            ],
            "clarification_question": "вопрос пользователю или null",
        }
        return f"""Ты — маршрутизатор запросов пользователя между специализированными агентами \
платформы градостроительного анализа. Твоя задача — разобрать запрос и составить план \
из шагов, каждый из которых выполняет один агент.

Доступные агенты:
{agents_block or '- (нет доступных агентов)'}

Верни только валидный JSON без markdown и пояснений:
{json.dumps(response_structure, ensure_ascii=False)}

Режимы (mode):
- "execute" — запрос (или его части) подходит хотя бы одному доступному агенту. \
Поле steps обязательно и содержит от 1 до {MAX_PLAN_STEPS} шагов.
- "needs_clarification" — запрос не подходит ни одному доступному агенту, неоднозначен \
или требует данных, которых нет (например, нужен расчёт по сценарию, а агенты расчёта \
недоступны без выбранного сценария). Поле steps должно быть пустым, а \
clarification_question обязателен.

Правила составления шагов:
- Используй ТОЛЬКО ключи агентов из списка доступных. Не придумывай агентов.
- Разбивай запрос на несколько шагов только когда для его частей действительно нужны \
РАЗНЫЕ агенты; иначе делай один шаг. Не дублируй один и тот же агент без необходимости.
- Каждый task — самодостаточная формулировка подзадачи на русском: агент видит только \
свой task и не видит исходный запрос и диалог. Переноси в task все нужные детали \
(названия сервисов, объекты, расстояния, условия).
- Упорядочивай шаги так, чтобы более поздние могли опираться на результаты более ранних: \
агенту будет автоматически передана краткая выжимка результатов предыдущих шагов.
- clarification_question при mode = "needs_clarification": укажи, чего не хватает или \
что непонятно, перечисли доступных агентов и их возможности и попроси уточнить запрос.

Правила работы с историей диалога:
- Сообщения переданы в хронологическом порядке, текущий запрос пользователя — последний.
- Если текущий запрос неполный (например «а теперь для школ», «повтори с буфером 200 метров»), \
восстанавливай недостающие детали из предыдущих сообщений.
- При противоречиях между сообщениями приоритет всегда у более поздних."""

    @staticmethod
    def _clarification_text(agents: list[AgentCatalogEntry]) -> str:
        if not agents:
            return (
                "Сейчас нет доступных агентов для обработки запроса. "
                "Если нужен расчёт ограничений или обеспеченности, выберите сценарий "
                "и повторите запрос."
            )
        agents_str = "; ".join(f"{entry.title} ({entry.key})" for entry in agents)
        return (
            "Не удалось определить, какой агент должен обработать запрос. "
            f"Доступные агенты: {agents_str}. "
            "Пожалуйста, уточните, что именно нужно сделать."
        )
