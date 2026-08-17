from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncGenerator, Callable
from typing import Any

from loguru import logger

from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    TableColumn,
    TablePartRequest,
    TablePayload,
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.common.exceptions.token_exceptions import (
    PipelineSuspendedError,
    TokenExpiredError,
)
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient, UrbanMcpTool
from src.agents.model_clients.llm_base import LlmChatResponse
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.pipeline_state import (
    TOKEN_REFRESH_TIMEOUT,
    PipelineStateStore,
    PipelineStatus,
)
from src.agents.services.scenario_data_plan_builder import (
    MAX_SCENARIO_TOOL_CALLS,
    ScenarioDataPlanBuilder,
)
from src.agents.services.service_entities.scenario_data_action import (
    ScenarioDataActionKind,
)


class ScenarioDataService(BaseLlmService):
    """Answer grounded questions using every read-only Urban MCP group."""

    def __init__(
        self,
        llm_host: str,
        chat_storage_client,
        urban_api_client,
        state_store: PipelineStateStore,
    ) -> None:
        super().__init__(llm_host, chat_storage_client, urban_api_client)
        self.state_store = state_store
        self.plan_builder = ScenarioDataPlanBuilder(self.llm_client)

    async def run_scenario_data_pipeline(
        self,
        urban_mcp_client: UrbanMcpClient,
        token: str,
        model: str | None,
        temperature: float,
        user_query: str,
        scenario_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Fill in the provider's model when the caller named none; keeps REST and A2A
        # on one behaviour and out of backend-specific literals.
        model = await self.resolve_model(model)
        if request_id is not None and await self.state_store.exists(request_id):
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            return

        request_id = request_id or self.state_store.new_request_id()
        original_chat_id = chat_id
        yield self._buf(request_id, self._pipeline_started(request_id))

        if not chat_id and persist_history:
            try:
                chat_id, title = await self.create_chat(
                    token,
                    model,
                    user_query,
                    additional_instructions=(
                        "Вопрос по данным и географическим слоям городского сценария."
                    ),
                    scenario_id=scenario_id,
                )
                yield self._buf(request_id, self._chat_created(chat_id, title))
            except Exception as exc:  # ChatStorage must not break analysis
                logger.warning(f"Scenario data: failed to create chat: {exc}")
                chat_id = None

        await self.state_store.create(
            request_id,
            chat_id=chat_id,
            user_query=user_query,
            scenario_id=scenario_id,
            model=model,
            temperature=temperature,
        )

        history: list[dict] = []
        if original_chat_id:
            try:
                chat_info = await self.get_chat_messages(token, original_chat_id)
                history = self.build_llm_history(
                    chat_info.messages, current_user_query=user_query
                )
                if persist_history:
                    await self.add_single_message(
                        token,
                        original_chat_id,
                        RoleEnum.USER,
                        user_query,
                        scenario_id=scenario_id,
                    )
            except Exception as exc:
                logger.warning(f"Scenario data: failed to load/persist history: {exc}")

        token_ref = [token]
        parts: list[TextPartRequest | TablePartRequest | ToolCallPartRequest] = []
        observations: list[dict[str, Any]] = []
        if scenario_id is None:
            observations.append(
                {
                    "context": "Сценарий не выбран.",
                    "summary": (
                        "Доступны только общие инструменты без обязательного "
                        "scenario_id. Если вопрос относится к конкретному сценарию, "
                        "нужно попросить пользователя выбрать сценарий."
                    ),
                }
            )
        executed: set[str] = set()

        yield self._buf(
            request_id,
            self._status("tool_discovery", "Загружаю инструменты Urban MCP…"),
        )
        tools_box: list[Any] = []
        async for event in self._retryable_operation(
            request_id,
            urban_mcp_client,
            token_ref,
            urban_mcp_client.load_tools,
            tools_box,
        ):
            yield self._buf(request_id, event)
        loaded_tools: list[UrbanMcpTool] = tools_box[0]
        if not loaded_tools:
            raise ValueError("Urban MCP returned no read-only tools")
        tools = self._tools_for_context(loaded_tools, scenario_id)
        if not tools:
            observations.append(
                {
                    "context": "Нет инструментов без контекста сценария.",
                    "summary": "Попроси пользователя выбрать сценарий.",
                }
            )

        successful_calls = 0
        for _ in range(MAX_SCENARIO_TOOL_CALLS + 3 if tools else 0):
            yield self._buf(
                request_id,
                self._status("planning", "Выбираю следующий источник данных…"),
            )
            action = await self.plan_builder.choose_action(
                model,
                user_query,
                tools,
                observations,
                history,
                scenario_id=scenario_id,
            )
            if action.action == ScenarioDataActionKind.FINAL_ANSWER:
                break
            if successful_calls >= MAX_SCENARIO_TOOL_CALLS:
                break

            tool = urban_mcp_client.get_tool(action.group, action.tool_name)
            try:
                arguments = self._prepare_arguments(tool, action.arguments, scenario_id)
            except ValueError as exc:
                observations.append(
                    {
                        "tool": f"{action.group}.{action.tool_name}",
                        "summary": (
                            "Вызов не выполнен: неверно заполнены обязательные "
                            f"аргументы ({exc}). Исправь аргументы или выбери "
                            "другой инструмент."
                        ),
                    }
                )
                continue
            call_key = json.dumps(
                [action.group, action.tool_name, arguments],
                ensure_ascii=False,
                sort_keys=True,
            )
            if call_key in executed:
                observations.append(
                    {
                        "tool": f"{action.group}.{action.tool_name}",
                        "summary": "Повторный вызов отклонён; выбери другой инструмент или заверши ответ.",
                    }
                )
                continue
            executed.add(call_key)

            tool_call = {
                "tool_name": action.tool_name,
                "arguments": arguments,
                "group": action.group,
            }
            source = f"URBAN_MCP/{action.group}"
            yield self._buf(
                request_id,
                self._tool_call_event(tool_call, source),
            )
            parts.append(
                ToolCallPartRequest(
                    kind="tool_call",
                    payload=ToolCallPayload(
                        execution_mode="sequential",
                        calls=[
                            ToolCall(
                                step=successful_calls + 1,
                                tool_name=action.tool_name,
                                arguments=arguments,
                            )
                        ],
                    ),
                    mcp_source=source,
                )
            )
            yield self._buf(
                request_id,
                self._status("tool_execution", f"Получаю данные: {tool.title}…"),
            )

            result_box: list[Any] = []
            async for event in self._retryable_operation(
                request_id,
                urban_mcp_client,
                token_ref,
                lambda: urban_mcp_client.execute_tool(
                    action.group,
                    action.tool_name,
                    arguments,
                    meta=(
                        {"scenario_id": scenario_id} if scenario_id is not None else {}
                    ),
                ),
                result_box,
            ):
                yield self._buf(request_id, event)
            result = self._unwrap_result(result_box[0])
            successful_calls += 1

            layer_count = 0
            for path, feature_collection in self._feature_collections(result):
                layer_count += 1
                name = action.layer_name or tool.title
                if path:
                    name = f"{name} · {path}"
                yield self._buf(
                    request_id,
                    {
                        "type": "feature_collection",
                        "content": {
                            "name": name,
                            "feature_collection": feature_collection,
                        },
                    },
                )

            table = self._table_from_result(
                result,
                name=f"urban_{action.group}_{action.tool_name}",
                title=tool.title,
            )
            if table is not None:
                yield self._buf(request_id, {"type": "table", "content": table})
                parts.append(self._table_part(table))

            observations.append(
                {
                    "tool": f"{action.group}.{action.tool_name}",
                    "arguments": arguments,
                    "layer_count": layer_count,
                    "summary": self._result_summary(result),
                }
            )

        yield self._buf(
            request_id,
            self._status("response_analysis", "Формирую ответ по полученным данным…"),
        )
        answer_parts: list[str] = []
        async for event in self._stream_answer(
            model, user_query, observations, temperature, history
        ):
            answer_parts.append(event["content"]["text"])
            yield self._buf(request_id, event)

        answer = "".join(answer_parts).strip()
        if answer:
            parts.append(TextPartRequest(kind="text", payload=TextPayload(text=answer)))
        await self.state_store.set_status(request_id, PipelineStatus.DONE)
        if persist_history and chat_id and parts:
            self._schedule_persist(
                token_ref[0], chat_id, parts, scenario_id=scenario_id
            )

    @staticmethod
    def _prepare_arguments(
        tool: UrbanMcpTool,
        arguments: dict[str, Any],
        scenario_id: int | None,
    ) -> dict[str, Any]:
        properties = tool.input_schema.get("properties") or {}
        prepared = {
            key: value
            for key, value in arguments.items()
            if key in properties and value is not None
        }
        if "scenario_id" in properties and scenario_id is not None:
            prepared["scenario_id"] = scenario_id
        required = set(tool.input_schema.get("required") or [])
        missing = required - set(prepared)
        if missing:
            raise ValueError(
                f"Tool {tool.group}.{tool.name} requires arguments: {sorted(missing)}"
            )
        return prepared

    @staticmethod
    def _tools_for_context(
        tools: list[UrbanMcpTool], scenario_id: int | None
    ) -> list[UrbanMcpTool]:
        """Hide tools that require scenario context when no scenario is selected."""

        if scenario_id is not None:
            return tools
        return [
            tool
            for tool in tools
            if "scenario_id" not in set(tool.input_schema.get("required") or [])
        ]

    async def _retryable_operation(
        self,
        request_id: str,
        client: UrbanMcpClient,
        token_ref: list[str],
        operation: Callable,
        result: list[Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        while True:
            try:
                result.append(await operation())
                return
            except TokenExpiredError:
                yield self._token_expired(request_id)
                await self.state_store.set_status(
                    request_id, PipelineStatus.WAITING_TOKEN
                )
                try:
                    token = await asyncio.wait_for(
                        self.state_store.wait_for_token(request_id),
                        timeout=TOKEN_REFRESH_TIMEOUT,
                    )
                except asyncio.TimeoutError as exc:
                    await self.state_store.set_status(
                        request_id, PipelineStatus.SUSPENDED
                    )
                    yield self._pipeline_suspended(request_id)
                    raise PipelineSuspendedError(request_id) from exc
                client.update_token(token)
                token_ref[0] = token
                await self.state_store.set_status(request_id, PipelineStatus.RUNNING)

    async def _stream_answer(
        self,
        model: str,
        user_query: str,
        observations: list[dict[str, Any]],
        temperature: float,
        history: list[dict],
    ) -> AsyncGenerator[dict[str, Any], None]:
        context = json.dumps(observations, ensure_ascii=False)
        if len(context) > 18000:
            context = context[:18000] + "…"
        messages = [
            {
                "role": "system",
                "content": f"""Ответь на вопрос по фактическим результатам Urban MCP.
Не выдумывай отсутствующие данные и явно отмечай пустые результаты. Если были
возвращены географические слои, скажи, какие именно слои отправлены на карту.
Не показывай внутренние JSON, имена MCP-инструментов и технический процесс.

Наблюдения:
{context}""",
            },
            *history,
            {"role": "user", "content": user_query},
        ]
        emitted = False
        async for part in await self.llm_client.chat(
            model,
            messages,
            think=False,
            stream=True,
            options={"temperature": temperature, "num_predict": 1400},
        ):
            part: LlmChatResponse
            text = part.message.content or ""
            if text:
                emitted = True
                yield self._chunk(text, done=False)
        if not emitted:
            yield self._chunk(
                "Данные получены, но модель не сформировала текстовый комментарий.",
                done=False,
            )
        yield self._chunk("", done=True)

    @classmethod
    def _feature_collections(cls, value: Any, path: str = ""):
        if isinstance(value, dict) and value.get("type") == "FeatureCollection":
            yield path, value
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}".strip(".")
                yield from cls._feature_collections(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                child_path = f"{path}[{index}]"
                yield from cls._feature_collections(child, child_path)

    @staticmethod
    def _unwrap_result(value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"result"}:
            return value["result"]
        return value

    @classmethod
    def _result_summary(cls, value: Any) -> str:
        compact = cls._compact(value)
        text = json.dumps(compact, ensure_ascii=False, default=str)
        return text[:5000] + ("…" if len(text) > 5000 else "")

    @classmethod
    def _compact(cls, value: Any, depth: int = 0) -> Any:
        if isinstance(value, dict) and value.get("type") == "FeatureCollection":
            features = value.get("features") or []
            return {
                "type": "FeatureCollection",
                "feature_count": len(features),
                "sample_properties": [
                    cls._compact(feature.get("properties") or {}, depth + 1)
                    for feature in features[:3]
                    if isinstance(feature, dict)
                ],
            }
        if depth >= 3:
            return "…"
        if isinstance(value, dict):
            return {
                str(key): cls._compact(child, depth + 1)
                for key, child in list(value.items())[:20]
                if key not in {"geometry", "coordinates"}
            }
        if isinstance(value, list):
            return [cls._compact(child, depth + 1) for child in value[:8]] + (
                [f"… ещё {len(value) - 8}"] if len(value) > 8 else []
            )
        if isinstance(value, str) and len(value) > 500:
            return value[:500] + "…"
        return value

    @classmethod
    def _table_from_result(
        cls, result: Any, *, name: str, title: str
    ) -> dict[str, Any] | None:
        rows = result
        if isinstance(rows, dict) and isinstance(rows.get("result"), list):
            rows = rows["result"]
        if (
            not isinstance(rows, list)
            or not rows
            or not all(isinstance(row, dict) for row in rows)
        ):
            return None
        keys: list[str] = []
        for row in rows[:20]:
            for key in row:
                if key not in keys:
                    keys.append(str(key))
                if len(keys) >= 12:
                    break
        normalized_rows = []
        for row in rows[:100]:
            normalized_rows.append(
                {key: cls._table_value(row.get(key)) for key in keys}
            )
        return {
            "name": re.sub(r"[^a-zA-Z0-9_]+", "_", name),
            "title": title,
            "columns": [{"key": key, "label": key} for key in keys],
            "rows": normalized_rows,
        }

    @staticmethod
    def _table_value(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)[:1000]
        return value

    @staticmethod
    def _table_part(table: dict[str, Any]) -> TablePartRequest:
        return TablePartRequest(
            kind="table",
            payload=TablePayload(
                name=table["name"],
                title=table["title"],
                columns=[TableColumn(**column) for column in table["columns"]],
                rows=table["rows"],
            ),
        )

    def _schedule_persist(
        self,
        token: str,
        chat_id: str,
        parts: list[TextPartRequest | TablePartRequest | ToolCallPartRequest],
        **metadata,
    ) -> None:
        task = asyncio.create_task(
            self.add_complex_message(
                token, chat_id, RoleEnum.ASSISTANT, parts, **metadata
            )
        )
        task.add_done_callback(self._log_persist_result)

    @staticmethod
    def _log_persist_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.exception(f"Scenario data: failed to persist response: {exc}")

    def _buf(self, request_id: str, event: dict[str, Any]) -> dict[str, Any]:
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    @staticmethod
    def _pipeline_started(request_id: str) -> dict[str, Any]:
        return {"type": "pipeline_started", "content": {"request_id": request_id}}

    @staticmethod
    def _chat_created(chat_id: str, title: str) -> dict[str, Any]:
        return {
            "type": "service_event",
            "content": {
                "event_type": "storage_event",
                "event": {
                    "storage_event_type": "chat_created",
                    "chat_id": chat_id,
                    "chat_title": title,
                },
            },
        }

    @staticmethod
    def _status(status: str, text: str) -> dict[str, Any]:
        return {"type": "status", "content": {"status": status, "text": text}}

    @staticmethod
    def _chunk(text: str, done: bool) -> dict[str, Any]:
        return {"type": "chunk", "content": {"text": text, "done": done}}

    @staticmethod
    def _tool_call_event(call: dict[str, Any], source: str) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "content": {
                "execution_mode": "sequential",
                "tool_calls": [call],
                "mcp_source": source,
            },
        }

    @staticmethod
    def _token_expired(request_id: str) -> dict[str, Any]:
        return {
            "type": "token_expired",
            "content": {
                "request_id": request_id,
                "message": "Token expired. Update token to continue request procedure.",
            },
        }

    @staticmethod
    def _pipeline_suspended(request_id: str) -> dict[str, Any]:
        return {
            "type": "pipeline_suspended",
            "content": {
                "request_id": request_id,
                "message": "Выполнение приостановлено: токен не был обновлён вовремя.",
            },
        }
