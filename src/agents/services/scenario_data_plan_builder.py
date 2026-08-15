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
    ) -> ScenarioDataAction:
        shortlist = self._shortlist(tools, user_query, observations)
        prompt = self._build_prompt(shortlist, observations)
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
            response = await self.llm_client.chat(
                model=model,
                messages=messages,
                think=False,
                format=ScenarioDataAction.model_json_schema(),
                options={"temperature": 0, "num_predict": 900},
            )
            raw = response["message"]["content"]
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
            return sum(
                3 if token in tool.title.lower() else 1
                for token in query_tokens
                if token in haystack
            )

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
        tools: list[UrbanMcpTool], observations: list[dict[str, Any]]
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
- scenario_id подставляется системой. Не угадывай и не меняй его.
- Сначала используй справочники, если для основного запроса нужно узнать ID по названию.
- Для запроса слоя выбирай инструмент, возвращающий GeoJSON/геометрию.
- Если собранных наблюдений достаточно для полного ответа, выбери final_answer.
- Не вызывай инструменты для создания, изменения или удаления данных.
- layer_name заполняй только если ожидается географический слой."""
