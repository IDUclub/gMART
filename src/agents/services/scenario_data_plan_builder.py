from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.agents.mcp_clients.urban_mcp_client import (
    URBAN_MCP_GROUP_DESCRIPTIONS,
    URBAN_MCP_GROUPS,
    UrbanMcpTool,
)
from src.agents.services.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.scenario_data_action import (
    ScenarioDataAction,
    ScenarioDataActionKind,
)

MAX_SCENARIO_TOOL_CALLS = 6
MAX_PLANNER_RETRIES = 2
#: Answer budget for the planner. The retry budget is deliberately much larger: an empty
#: reply usually means the reasoning trace consumed everything before any content was emitted.
PLANNER_NUM_PREDICT = 900
PLANNER_NUM_PREDICT_RETRY = 3000
#: Effort used on a retry after an empty reply. "low" is exactly the value that produces no
#: content on a Harmony-served gpt-oss, so escalating to "medium" is the fix, not a guess.
PLANNER_RETRY_REASONING_EFFORT = "medium"

#: Subjects a tool can be *about* that a general data question is not asking for. These tools
#: stay in the catalogue — restriction zones are legitimate Urban API data and must remain
#: answerable — but they must not outrank an on-topic tool merely by sharing generic words.
#: "Получить зоны ограничений объектов на территории" matches "объекты" and "территория" in its
#: title, so on a question about which objects exist it scored level with the plain objects
#: tool, and the model was free to pick either.
_TOPIC_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ограничен", ("ограничен", "буфер", "зон", "buffer", "restrict")),
    ("буфер", ("ограничен", "буфер", "буферн", "buffer")),
    ("показател", ("показател", "индикатор", "indicator", "значени")),
    ("социальн", ("социальн", "soc_group", "ценност")),
    ("норматив", ("норматив", "норм")),
)
#: Enough to drop an off-subject tool below an on-subject one without hiding it outright.
_OFF_TOPIC_PENALTY = 4


def _off_topic_penalty(tool: UrbanMcpTool, context: str) -> int:
    """Penalise a tool whose declared subject the question never mentions."""

    title = tool.title.lower()
    for marker, query_words in _TOPIC_MARKERS:
        if marker in title and not any(word in context for word in query_words):
            return _OFF_TOPIC_PENALTY
    return 0


class ScenarioDataPlanBuilder:
    """Select one grounded Urban MCP action at a time."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def choose_action(
        self,
        model: str,
        user_query: str,
        tools: list[UrbanMcpTool],
        observations: list[dict[str, Any]],
        history: list[dict] | None = None,
        scenario_id: int | None = None,
    ) -> ScenarioDataAction:
        shortlist = self._shortlist(tools, user_query, observations)
        prompt = self._build_prompt(shortlist, observations, scenario_id)
        messages = [
            {"role": "system", "content": prompt},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        error = ""
        for attempt in range(MAX_PLANNER_RETRIES + 1):
            if error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Предыдущий JSON не прошёл проверку: "
                            f"{error}. Верни исправленный JSON строго по схеме."
                        ),
                    }
                )
            # Escalate on each retry. Repeating an identical call is pointless when the
            # server answered with an empty string, and on a Harmony-served gpt-oss the
            # lever that actually matters is the reasoning effort, not the budget:
            # measured on the same prompt, reasoning_effort="low" returns *no content at
            # all* — the model finishes its analysis channel and stops without emitting a
            # final message — while "medium", "high" and omitting the field all answer.
            # Prompt size is not the factor: six tools (2k tokens) fail on "low" just as
            # forty-one (11.5k) do. So a retry raises the effort explicitly, which wins
            # over the configured default because _apply_think uses setdefault.
            budget = PLANNER_NUM_PREDICT if attempt == 0 else PLANNER_NUM_PREDICT_RETRY
            call: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "think": False,
                "options": {"temperature": 0, "num_predict": budget},
            }
            if attempt > 0:
                call["reasoning_effort"] = PLANNER_RETRY_REASONING_EFFORT
            if attempt < MAX_PLANNER_RETRIES:
                call["format"] = ScenarioDataAction.model_json_schema()
            else:
                # Last chance: no structured-output constraint at all, JSON asked for in
                # words. A model that returns nothing under the schema usually answers here.
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Ответь ТОЛЬКО JSON-объектом по описанной схеме, без "
                            "пояснений и без markdown-ограждения."
                        ),
                    }
                ]
                call["messages"] = messages
            response = await self.llm_client.chat(**call)
            raw = response["message"]["content"]
            if not (raw or "").strip():
                # Name the real cause. "Empty answer" on its own sent a reader looking for a
                # vague prompt, when the model had in fact reasoned to a conclusion and
                # simply never emitted it.
                trace = (response["message"].get("thinking") or "").strip()
                error = (
                    "модель не выдала финальный ответ"
                    + (
                        f" (сгенерирован только след рассуждений: {trace[:160]}…)"
                        if trace
                        else " (пустой ответ без следа рассуждений)"
                    )
                    + "; на gpt-oss через Harmony это даёт reasoning_effort=low — "
                    "поднимите OPENAI_THINK_EFFORT до medium"
                )
                logger.warning(
                    f"Empty scenario-data action, attempt {attempt + 1} "
                    f"(num_predict={budget}, format={'format' in call}, "
                    f"reasoning_effort={call.get('reasoning_effort', 'configured')}, "
                    f"reasoning_trace_len={len(trace)})"
                )
                continue
            try:
                action = ScenarioDataAction.model_validate_json(strip_json_fence(raw))
                return self._canonicalize(action, shortlist)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
                logger.warning(
                    f"Invalid scenario-data action, attempt {attempt + 1}: {error}"
                )
        raise ValueError(f"invalid scenario-data action after retries: {error}")

    @staticmethod
    def _canonicalize(
        action: ScenarioDataAction, tools: list[UrbanMcpTool]
    ) -> ScenarioDataAction:
        if action.action == ScenarioDataActionKind.FINAL_ANSWER:
            return action.model_copy(
                update={
                    "group": None,
                    "tool_name": None,
                    "arguments": {},
                    "layer_name": None,
                }
            )
        if action.group not in URBAN_MCP_GROUPS:
            raise ValueError(f"unknown Urban MCP group: {action.group}")
        selected = next(
            (
                tool
                for tool in tools
                if tool.group == action.group and tool.name == action.tool_name
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"tool {action.group}.{action.tool_name} is not in the supplied catalog"
            )
        properties = selected.input_schema.get("properties") or {}
        unknown = set(action.arguments) - set(properties)
        if unknown:
            raise ValueError(
                f"unknown arguments for {selected.name}: {sorted(unknown)}"
            )
        return action

    @classmethod
    def _shortlist(
        cls,
        tools: list[UrbanMcpTool],
        user_query: str,
        observations: list[dict[str, Any]],
    ) -> list[UrbanMcpTool]:
        context = (
            user_query
            + " "
            + " ".join(str(item.get("summary") or "") for item in observations[-3:])
        )
        query_tokens = cls._tokens(context)

        def score(tool: UrbanMcpTool) -> int:
            haystack = " ".join(
                (
                    tool.name,
                    tool.title,
                    tool.description[:600],
                    URBAN_MCP_GROUP_DESCRIPTIONS[tool.group],
                    " ".join(tool.tags),
                )
            ).lower()
            base = sum(
                3 if token in tool.title.lower() else 1
                for token in query_tokens
                if token in haystack
            )
            return base - _off_topic_penalty(tool, context.lower())

        chosen: dict[tuple[str, str], UrbanMcpTool] = {}
        for group in URBAN_MCP_GROUPS:
            group_tools = sorted(
                (tool for tool in tools if tool.group == group),
                key=lambda tool: (-score(tool), tool.name),
            )
            for tool in group_tools[:5]:
                chosen[(tool.group, tool.name)] = tool
        for tool in sorted(tools, key=lambda item: (-score(item), item.name))[:24]:
            chosen[(tool.group, tool.name)] = tool
        return list(chosen.values())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zа-яё0-9_]+", text.lower())
            if len(token) >= 3
        }

    @staticmethod
    def _build_prompt(
        tools: list[UrbanMcpTool],
        observations: list[dict[str, Any]],
        scenario_id: int | None,
    ) -> str:
        catalog = [tool.compact_prompt_entry() for tool in tools]
        response_shape = {
            "action": "call_tool | final_answer",
            "group": "имя MCP-группы или null",
            "tool_name": "точное имя инструмента или null",
            "arguments": {"имя параметра": "значение"},
            "layer_name": "понятное название слоя или null",
            "reason": "краткая причина выбора",
        }
        return f"""Ты — управляющий агент данных городского сценария. На каждом шаге выбери
ровно одно действие: вызвать read-only Urban MCP инструмент или завершить сбор данных.

Доступные инструменты этого шага:
{json.dumps(catalog, ensure_ascii=False)}

Результаты уже выполненных шагов (геометрия сокращена, полные слои уже отправлены клиенту):
{json.dumps(observations, ensure_ascii=False)}

Верни только JSON:
{json.dumps(response_shape, ensure_ascii=False)}

Правила:
- Используй только точные group и tool_name из каталога.
- Не повторяй уже выполненный вызов с теми же аргументами.
- Контекст сценария: {scenario_id if scenario_id is not None else "не выбран"}.
- Если scenario_id выбран, он подставляется системой: не угадывай и не меняй его.
- Если scenario_id не выбран и вопрос требует данных конкретного сценария, заверши сбор
  данных через final_answer: в итоговом ответе нужно попросить пользователя выбрать сценарий.
- Сначала используй справочники, если для основного запроса нужно узнать ID по названию.
- ОБРАТНОЕ НАПРАВЛЕНИЕ ВАЖНЕЕ: если в наблюдении есть unresolved_references, названия по
  этим идентификаторам ещё не получены. Вызови справочник, который вернёт их названия
  (например, типы сервисов или типы объектов), и не выбирай final_answer, пока это не
  сделано, — иначе ответ будет про номера, а не про то, что человек спросил.
- Один вызов почти никогда не отвечает на вопрос целиком. Типичная последовательность:
  получить записи -> получить справочник типов -> сопоставить -> завершить.
- Для запроса слоя выбирай инструмент, возвращающий GeoJSON/геометрию.
- Выбирай инструмент, ПРЕДМЕТ которого совпадает с вопросом. Инструменты про зоны
  ограничений, буферы, показатели и социальные группы доступны и уместны, только если
  спросили именно о них; на вопрос «какие объекты/сервисы есть и сколько» бери инструменты
  по объектам и сервисам, а не по зонам ограничений.
- Выбирай final_answer, только если по наблюдениям можно назвать конкретные сущности и
  числа, а не только их идентификаторы и общее количество.
- Не вызывай инструменты для создания, изменения или удаления данных.
- layer_name заполняй только если ожидается географический слой."""
