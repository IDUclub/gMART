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
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.pipeline_state import (
    TOKEN_REFRESH_TIMEOUT,
    PipelineStateStore,
    PipelineStatus,
)
from src.agents.services.scenario_data_aggregate import (
    aggregate_result,
    unresolved_references,
)
from src.agents.services.scenario_data_evaluator import (
    MAX_ANSWER_ATTEMPTS,
    ScenarioDataEvaluator,
    wants_layers,
)
from src.agents.services.scenario_data_plan_builder import (
    MAX_SCENARIO_TOOL_CALLS,
    ScenarioDataPlanBuilder,
)
from src.agents.services.scenario_data_types import (
    ScenarioEntityKind,
    ScenarioTypeIntent,
    build_type_distribution,
    classify_type_query,
    distribution_answer,
    distribution_table,
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
        self.evaluator = ScenarioDataEvaluator(self.llm_client)

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
                    agent_id="scenario_data",
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
        type_intent = classify_type_query(user_query, history)
        clarification = None
        if type_intent is not None and scenario_id is None:
            clarification = (
                "Сначала выберите сценарий, для которого нужно посчитать объекты "
                "или сервисы по типам."
            )
        elif type_intent is not None:
            clarification = type_intent.clarification
        if clarification:
            yield self._buf(
                request_id,
                self._status("planning", "Уточняю параметры запроса…"),
            )
            for event in self._answer_events(clarification):
                yield self._buf(request_id, event)
            parts.append(
                TextPartRequest(kind="text", payload=TextPayload(text=clarification))
            )
            await self._complete_pipeline(
                request_id,
                token_ref[0],
                chat_id,
                parts,
                scenario_id=scenario_id,
                persist_history=persist_history,
            )
            return

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

        if type_intent is not None and type_intent.kinds:
            type_tools = self._resolve_type_tools(tools, type_intent)
            if type_tools is not None:
                async for event in self._run_type_distribution_pipeline(
                    request_id=request_id,
                    urban_mcp_client=urban_mcp_client,
                    token_ref=token_ref,
                    type_intent=type_intent,
                    type_tools=type_tools,
                    scenario_id=scenario_id,
                    chat_id=chat_id,
                    parts=parts,
                    persist_history=persist_history,
                ):
                    yield event
                return
            observations.append(
                {
                    "context": "Детерминированный подсчёт по типам недоступен.",
                    "summary": (
                        "В текущем каталоге Urban MCP не найдена полная пара "
                        "инструментов записей и проектного справочника; используй "
                        "обычное планирование и не угадывай данные."
                    ),
                }
            )

        # One pass = plan/execute tools, draft an answer, judge it. A rejected answer buys
        # another pass with the evaluator's hint in the observations, so the retry is
        # steered rather than repeated. `executed` is deliberately NOT cleared between
        # passes: an identical call returns identical data and could only produce the
        # same answer.
        answer = ""
        layers_expected = wants_layers(user_query)
        for attempt in range(MAX_ANSWER_ATTEMPTS):
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
                    if attempt > 0 and successful_calls == 0:
                        # A retry that finishes without fetching anything can only reproduce
                        # the answer that was just rejected. Push back once and re-plan.
                        observations.append(
                            {
                                "context": "Повторный проход",
                                "summary": (
                                    "Нельзя завершать сбор, не получив новых данных: "
                                    "предыдущий ответ уже отклонён по этим наблюдениям. "
                                    "Выбери инструмент, закрывающий указанный пробел."
                                ),
                            }
                        )
                        continue
                    break
                if successful_calls >= MAX_SCENARIO_TOOL_CALLS:
                    break

                tool = urban_mcp_client.get_tool(action.group, action.tool_name)
                try:
                    arguments = self._prepare_arguments(
                        tool, action.arguments, scenario_id
                    )
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
                            {"scenario_id": scenario_id}
                            if scenario_id is not None
                            else {}
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

                observation: dict[str, Any] = {
                    "tool": f"{action.group}.{action.tool_name}",
                    "arguments": arguments,
                    "layer_count": layer_count,
                    "summary": self._result_summary(result),
                }
                # Exact counts, computed here rather than left to the model: the summary above
                # is a *sample* of the rows, so counting from it is impossible by construction.
                aggregate = aggregate_result(result)
                if aggregate is not None:
                    observation["aggregate"] = aggregate
                    # "12 services of type 3" is not an answer. Naming an unresolved
                    # reference is a follow-up call, and the planner is told so explicitly
                    # rather than being left to notice on its own — it does not.
                    pending = unresolved_references(aggregate)
                    if pending:
                        observation["unresolved_references"] = pending
                        observation["next_step"] = (
                            "Эти поля — ссылки на справочник: "
                            f"{', '.join(pending)}. Названия по ним ещё не получены. "
                            "Вызови справочный инструмент, возвращающий названия для "
                            "этих идентификаторов, и только потом завершай сбор."
                        )
                observations.append(observation)

            yield self._buf(
                request_id,
                self._status(
                    "response_analysis", "Формирую ответ по полученным данным…"
                ),
            )
            # Drafted without streaming on purpose: the answer must be judged before the user
            # sees it, otherwise a rejected draft would already be on screen.
            answer = await self._draft_answer(
                model, user_query, observations, temperature, history
            )

            yield self._buf(
                request_id,
                self._status("answer_review", "Проверяю полноту ответа…"),
            )
            verdict = await self.evaluator.evaluate(
                model, user_query, observations, answer
            )
            if verdict.sufficient:
                break

            logger.info(
                "Scenario data: answer rejected on attempt "
                f"{attempt + 1}: {'; '.join(verdict.reasons)}"
            )
            if attempt == MAX_ANSWER_ATTEMPTS - 1:
                # Budget spent. The last draft still goes out — a partial answer beats none —
                # but it must not pass for a checked one.
                answer = self._append_shortfall_note(answer, verdict.reasons)
                break

            yield self._buf(
                request_id,
                self._status(
                    "answer_retry", "Ответ неполный, собираю недостающие данные…"
                ),
            )
            observations.append(
                {
                    "context": "Проверка предыдущего ответа",
                    "summary": (
                        f"Предыдущий ответ отклонён. Что исправить: {verdict.hint}"
                    ),
                }
            )
            if layers_expected:
                observations.append(
                    {
                        "context": "Требуются слои",
                        "summary": (
                            "Пользователь просил показать объекты на карте: выбери "
                            "инструмент, возвращающий геометрию (GeoJSON/WithGeometry)."
                        ),
                    }
                )

        for event in self._answer_events(answer):
            yield self._buf(request_id, event)
        answer = answer.strip()
        if answer:
            parts.append(TextPartRequest(kind="text", payload=TextPayload(text=answer)))

        await self._complete_pipeline(
            request_id,
            token_ref[0],
            chat_id,
            parts,
            scenario_id=scenario_id,
            persist_history=persist_history,
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

    @classmethod
    def _resolve_type_tools(
        cls,
        tools: list[UrbanMcpTool],
        intent: ScenarioTypeIntent,
    ) -> (
        dict[
            ScenarioEntityKind,
            tuple[UrbanMcpTool, UrbanMcpTool, UrbanMcpTool | None],
        ]
        | None
    ):
        names = {
            ScenarioEntityKind.PHYSICAL_OBJECT: (
                "GetScenarioPhysicalObjects",
                "GetScenarioPhysicalObjectTypes",
                "GetPhysicalObjectTypes",
            ),
            ScenarioEntityKind.SERVICE: (
                "GetScenarioServices",
                "GetScenarioServiceTypes",
                "GetServiceTypes",
            ),
        }
        resolved = {}
        for kind in intent.kinds:
            entity_name, project_types_name, dictionary_name = names[kind]
            entity_tool = cls._find_type_tool(tools, entity_name, kind, role="entities")
            project_types_tool = cls._find_type_tool(
                tools, project_types_name, kind, role="project_types"
            )
            dictionary_tool = cls._find_type_tool(
                tools, dictionary_name, kind, role="dictionary"
            )
            # Older Urban MCP catalogues may expose only the global dictionary. It is
            # still authoritative, but current catalogues let us fetch the smaller
            # scenario-specific type set first and consult the global dictionary only
            # for an unresolved ID.
            catalog_tool = project_types_tool or dictionary_tool
            if entity_tool is None or catalog_tool is None:
                return None
            fallback_tool = (
                dictionary_tool
                if project_types_tool is not None
                and dictionary_tool is not project_types_tool
                else None
            )
            resolved[kind] = (entity_tool, catalog_tool, fallback_tool)
        return resolved

    @staticmethod
    def _find_type_tool(
        tools: list[UrbanMcpTool],
        preferred_name: str,
        kind: ScenarioEntityKind,
        *,
        role: str,
    ) -> UrbanMcpTool | None:
        exact = next((tool for tool in tools if tool.name == preferred_name), None)
        if exact is not None:
            return exact

        subject = "физическ" if kind == ScenarioEntityKind.PHYSICAL_OBJECT else "сервис"
        candidates = []
        for tool in tools:
            title = tool.title.lower().replace("ё", "е")
            properties = tool.input_schema.get("properties") or {}
            if subject not in title:
                continue
            if role == "entities":
                matches = (
                    tool.group == "projects"
                    and "scenario_id" in properties
                    and "сценар" in title
                    and "тип" not in title
                    and "геометр" not in title
                    and "контекст" not in title
                )
            elif role == "project_types":
                matches = (
                    tool.group == "projects"
                    and "scenario_id" in properties
                    and "сценар" in title
                    and "тип" in title
                )
            else:
                matches = (
                    tool.group == "dictionaries"
                    and "тип" in title
                    and " по " not in title
                )
            if matches:
                candidates.append(tool)
        return (
            sorted(candidates, key=lambda tool: (len(tool.title), tool.name))[0]
            if candidates
            else None
        )

    async def _run_type_distribution_pipeline(
        self,
        *,
        request_id: str,
        urban_mcp_client: UrbanMcpClient,
        token_ref: list[str],
        type_intent: ScenarioTypeIntent,
        type_tools: dict[
            ScenarioEntityKind,
            tuple[UrbanMcpTool, UrbanMcpTool, UrbanMcpTool | None],
        ],
        scenario_id: int | None,
        chat_id: str | None,
        parts: list[TextPartRequest | TablePartRequest | ToolCallPartRequest],
        persist_history: bool,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if scenario_id is None:
            raise ValueError("scenario_id is required for a type distribution")

        distributions = []
        step = 0
        for kind in type_intent.kinds:
            entity_tool, catalog_tool, fallback_tool = type_tools[kind]
            noun = (
                "физические объекты"
                if kind == ScenarioEntityKind.PHYSICAL_OBJECT
                else "сервисы"
            )
            results = []
            calls = (
                (entity_tool, f"Получаю {noun} сценария…"),
                (
                    catalog_tool,
                    f"Получаю типы сценария: {noun}…",
                ),
            )
            for tool, status_text in calls:
                step += 1
                arguments = self._prepare_arguments(tool, {}, scenario_id)
                source = f"URBAN_MCP/{tool.group}"
                tool_call = {
                    "tool_name": tool.name,
                    "arguments": arguments,
                    "group": tool.group,
                }
                yield self._buf(request_id, self._tool_call_event(tool_call, source))
                parts.append(
                    ToolCallPartRequest(
                        kind="tool_call",
                        payload=ToolCallPayload(
                            execution_mode="sequential",
                            calls=[
                                ToolCall(
                                    step=step,
                                    tool_name=tool.name,
                                    arguments=arguments,
                                )
                            ],
                        ),
                        mcp_source=source,
                    )
                )
                yield self._buf(
                    request_id,
                    self._status("tool_execution", status_text),
                )
                result_box: list[Any] = []
                async for event in self._retryable_operation(
                    request_id,
                    urban_mcp_client,
                    token_ref,
                    lambda tool=tool, arguments=arguments: urban_mcp_client.execute_tool(
                        tool.group,
                        tool.name,
                        arguments,
                        meta={"scenario_id": scenario_id},
                    ),
                    result_box,
                ):
                    yield self._buf(request_id, event)
                results.append(self._unwrap_result(result_box[0]))

            yield self._buf(
                request_id,
                self._status(
                    "response_analysis", f"Считаю уникальные сущности: {noun}…"
                ),
            )
            distribution = build_type_distribution(results[0], results[1], kind)
            needs_fallback = any(
                row["status"] != "точное соответствие"
                for row in distribution.rows
                if int(row["count"]) > 0
            )
            if needs_fallback and fallback_tool is not None:
                step += 1
                arguments = self._prepare_arguments(fallback_tool, {}, scenario_id)
                source = f"URBAN_MCP/{fallback_tool.group}"
                tool_call = {
                    "tool_name": fallback_tool.name,
                    "arguments": arguments,
                    "group": fallback_tool.group,
                }
                yield self._buf(request_id, self._tool_call_event(tool_call, source))
                parts.append(
                    ToolCallPartRequest(
                        kind="tool_call",
                        payload=ToolCallPayload(
                            execution_mode="sequential",
                            calls=[
                                ToolCall(
                                    step=step,
                                    tool_name=fallback_tool.name,
                                    arguments=arguments,
                                )
                            ],
                        ),
                        mcp_source=source,
                    )
                )
                yield self._buf(
                    request_id,
                    self._status(
                        "tool_execution",
                        f"Проверяю неопределённые ID в общем справочнике: {noun}…",
                    ),
                )
                fallback_box: list[Any] = []
                async for event in self._retryable_operation(
                    request_id,
                    urban_mcp_client,
                    token_ref,
                    lambda: urban_mcp_client.execute_tool(
                        fallback_tool.group,
                        fallback_tool.name,
                        arguments,
                        meta={"scenario_id": scenario_id},
                    ),
                    fallback_box,
                ):
                    yield self._buf(request_id, event)
                distribution = build_type_distribution(
                    results[0],
                    results[1],
                    kind,
                    fallback_catalog_result=self._unwrap_result(fallback_box[0]),
                )
            distributions.append(distribution)

            table = distribution_table(distribution)
            yield self._buf(request_id, {"type": "table", "content": table})
            parts.append(self._table_part(table))

        yield self._buf(
            request_id,
            self._status("answer_review", "Проверяю итоговые количества…"),
        )
        answer = distribution_answer(scenario_id, distributions)
        for event in self._answer_events(answer):
            yield self._buf(request_id, event)
        parts.append(TextPartRequest(kind="text", payload=TextPayload(text=answer)))
        await self._complete_pipeline(
            request_id,
            token_ref[0],
            chat_id,
            parts,
            scenario_id=scenario_id,
            persist_history=persist_history,
        )

    async def _complete_pipeline(
        self,
        request_id: str,
        token: str,
        chat_id: str | None,
        parts: list[TextPartRequest | TablePartRequest | ToolCallPartRequest],
        *,
        scenario_id: int | None,
        persist_history: bool,
    ) -> None:
        await self.state_store.set_status(request_id, PipelineStatus.DONE)
        if persist_history and chat_id and parts:
            try:
                await self.add_complex_message(
                    token,
                    chat_id,
                    RoleEnum.ASSISTANT,
                    parts,
                    scenario_id=scenario_id,
                )
            except Exception as exc:
                # The accepted SSE answer has already been emitted and remains authoritative
                # for the active browser window; persistence failure must not fail the stream.
                logger.exception(f"Scenario data: failed to persist response: {exc}")

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

    @staticmethod
    def _append_shortfall_note(answer: str, reasons: list[str]) -> str:
        """Say plainly what the last attempt still failed to cover.

        Shipping the rejected draft silently would hand the user the very thing the evaluator
        exists to catch — a fluent answer that quietly omits what was asked for.
        """

        note = "Не удалось получить полностью: " + " ".join(reasons)
        return f"{answer.rstrip()}\n\n{note}" if answer.strip() else note

    def _answer_events(self, answer: str) -> list[dict[str, Any]]:
        """Emit an accepted answer as chunk events, matching the streaming contract.

        The draft is produced in one call so it can be judged first, so there is nothing left
        to stream; the text is still delivered in pieces to keep the client's rendering path
        (and the A2A consumers) unchanged.
        """

        text = answer.strip()
        if not text:
            return [
                self._chunk(
                    "Данные получены, но модель не сформировала текстовый комментарий.",
                    done=False,
                ),
                self._chunk("", done=True),
            ]
        step = 280
        events = [
            self._chunk(text[i : i + step], done=False)
            for i in range(0, len(text), step)
        ]
        events.append(self._chunk("", done=True))
        return events

    async def _draft_answer(
        self,
        model: str,
        user_query: str,
        observations: list[dict[str, Any]],
        temperature: float,
        history: list[dict],
    ) -> str:
        """One non-streamed answer over the observations, for the evaluator to judge."""

        messages = self._answer_messages(user_query, observations, history)
        response = await self.llm_client.chat(
            model,
            messages,
            think=False,
            stream=False,
            options={"temperature": temperature, "num_predict": 1400},
        )
        return (response["message"]["content"] or "").strip()

    def _answer_messages(
        self,
        user_query: str,
        observations: list[dict[str, Any]],
        history: list[dict],
    ) -> list[dict]:
        context = json.dumps(observations, ensure_ascii=False)
        if len(context) > 18000:
            context = context[:18000] + "…"
        return [
            {
                "role": "system",
                "content": f"""Ответь на вопрос по фактическим результатам Urban MCP.
Не выдумывай отсутствующие данные и явно отмечай пустые результаты. Если были
возвращены географические слои, скажи, какие именно слои отправлены на карту.
Не показывай внутренние JSON, имена MCP-инструментов и технический процесс.

В наблюдениях поле "aggregate" содержит ТОЧНЫЕ количества, посчитанные по всем
записям, а не по образцу: total_records — сколько всего записей, breakdown — сколько
записей приходится на каждое значение поля (например, physical_object_type.name).
Когда спрашивают, какие объекты есть и сколько их, приводи числа именно оттуда и
перечисляй категории — не пиши, что типы неизвестны, если breakdown их содержит.
Поле "summary" — лишь образец нескольких записей; не делай по нему выводов о
количестве.

Наблюдения:
{context}""",
            },
            *history,
            {"role": "user", "content": user_query},
        ]

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
