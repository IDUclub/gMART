from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from ollama import ChatResponse

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.exceptions.token_exceptions import (
    PipelineSuspendedError,
    TokenExpiredError,
)
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.pipeline_state import (
    TOKEN_REFRESH_TIMEOUT,
    PipelineStateStore,
    PipelineStatus,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient

_MCP_SOURCE = "URBAN_DATA_MCP_URL"
_EXECUTION_MODE = "urban_data_tool_call"
# Checkpoint key holding the terminal state of the loop (accepted answer). Unlike
# NormGraph's per-iteration resume, the tool-calling loop is not resumed step-by-step on
# reconnect (tool results here may embed large GeoJSON payloads that are unsuitable for a
# Redis checkpoint) — a reconnect before completion simply restarts the loop, bounded by
# MAX_TOOL_ITERATIONS.
_QA_PROGRESS = "qa_progress"
# Every embedded feature/geometry in a tool result is dropped to this placeholder before
# the result is echoed back into the LLM's context — the model reasons about counts and
# properties, not raw coordinate arrays; the untouched result is still used for the
# feature_collection events sent to the client.
_GEOMETRY_PLACEHOLDER = "<omitted for brevity>"

SYSTEM_PROMPT = (
    "Ты — ассистент по урбанистическим данным (Urban API): территории, проекты и "
    "сценарии, физические объекты, сервисы, индикаторы, социальные группы и ценности, "
    "а также справочники (типы объектов, сервисов, единицы измерения и т.п.). Отвечай "
    "на вопрос пользователя, используя доступные инструменты, чтобы получить "
    "актуальные данные. Правила:\n"
    "- Не выдумывай данные, названия и идентификаторы — используй только то, что "
    "вернули инструменты.\n"
    "- Если для ответа нужно несколько запросов, вызывай инструменты последовательно, "
    "уточняя параметры по мере получения данных.\n"
    "- Если ни один инструмент не подходит или данных недостаточно для ответа — прямо "
    "сообщи об этом, не придумывай значения.\n"
    "- Если инструменты вернули пространственные данные (слои), в текстовом ответе "
    "кратко опиши, что показано на карте.\n"
    "- Отвечай на русском языке, ясно и по существу."
)


class UrbanDataQaService(BaseLlmService):
    """
    Q&A agent over the external, grouped Urban MCP server (territories/services/physical
    objects and other urban data groups exposed by urban-mcp).

    Unlike the other pipeline-shaped agents (restriction/provision/norms), the Urban MCP
    tool catalogue is not a fixed, known-in-advance contract — it is discovered at
    runtime. The agent therefore runs a native Ollama tool-calling loop: the model is
    given the MCP's tool list and picks/calls tools itself (up to
    ``MAX_TOOL_ITERATIONS`` rounds), after which a final natural-language answer is
    streamed to the client grounded in the tool results. Any ``FeatureCollection``
    embedded in a tool result — at any nesting depth — is surfaced as a
    ``feature_collection`` event so the frontend can render it as a layer.

    Reconnect: every emitted event is buffered in Redis (``PipelineStateStore``) keyed by
    the ``request_id`` announced in the first ``pipeline_started`` event. Reconnecting
    after the pipeline has produced its final answer replays the buffered events only. A
    reconnect that happens mid-loop (rare — the loop is short) simply restarts the
    tool-calling loop rather than resuming the exact LLM conversation state.
    """

    MAX_TOOL_ITERATIONS = 5

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client: ChatStorageApiClient,
        urban_api_client: UrbanApiClient,
        state_store: PipelineStateStore,
    ) -> None:
        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.state_store = state_store

    # ------------------------------------------------------------------
    # Public entry point (reconnect handling + chat storage + history)
    # ------------------------------------------------------------------

    async def run_urban_data_qa_pipeline(
        self,
        urban_data_mcp_client: "UrbanDataMcpClient",
        token: str,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Mutable container so the inner loop can update the token on refresh and the
        # outer generator (and the final chat-storage persist call) sees the latest value.
        token_ref: list[str] = [token]
        collected: dict[str, Any] = {
            "final_answer": "",
            "tool_calls": [],
            "newly_completed": False,
        }
        is_reconnect = request_id is not None and await self.state_store.exists(
            request_id
        )

        if is_reconnect:
            logger.info(
                f"Urban data QA reconnect request_id={request_id}, replaying buffered events"
            )
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            stored = await self.state_store.get_state(request_id) or {}
            if not chat_id and stored.get("chat_id"):
                chat_id = stored["chat_id"]
            model = stored.get("model") or model
            if stored.get("temperature") is not None:
                temperature = stored["temperature"]
            user_query = stored.get("user_query") or user_query
            if stored.get("scenario_id") is not None:
                scenario_id = stored["scenario_id"]
        else:
            request_id = request_id or self.state_store.new_request_id()

        original_chat_id = chat_id

        if not is_reconnect:
            yield self._buf(request_id, self._pipeline_started_event(request_id))

            # No chat_id supplied → create a new chat tagged with scenario_id. A2A runs
            # pass persist_history=False: no chat is created and nothing is written to
            # ChatStorage (history stays read-only).
            if not chat_id and persist_history:
                project_id: int | None = None
                if scenario_id is not None:
                    try:
                        project_id = (
                            await self.urban_api_client.get_project_by_scenario(
                                token_ref[0], scenario_id
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Urban data QA: failed to resolve project_id for "
                            f"scenario_id={scenario_id}: {exc}"
                        )
                        yield self._buf(
                            request_id, self._project_lookup_failed_event(scenario_id)
                        )
                try:
                    chat_id, title = await self.create_chat(
                        token_ref[0],
                        model,
                        user_query,
                        additional_instructions=(
                            "Запрос направлен агенту вопросов по урбанистическим "
                            "данным (внешний Urban MCP)."
                        ),
                        scenario_id=scenario_id,
                        project_id=project_id,
                        resolve_project_id=False,
                    )
                    yield self._buf(
                        request_id, self._chat_created_event(chat_id, title)
                    )
                except Exception as exc:  # chat storage must not break the stream
                    logger.warning(f"Urban data QA: failed to create chat: {exc}")
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
                chat_info = await self.get_chat_messages(token_ref[0], original_chat_id)
                history = self.build_llm_history(
                    chat_info.messages, current_user_query=user_query
                )
            except Exception as exc:
                logger.warning(f"Urban data QA: failed to fetch chat history: {exc}")

        # A follow-up question in an existing chat is persisted here — create_chat
        # stores only the first one. Runs after the history fetch so the current
        # question doesn't also enter the LLM context from storage, and is skipped on
        # reconnect (the original run already stored it). Chat storage failures must
        # not break the stream.
        if persist_history and not is_reconnect and original_chat_id:
            try:
                await self.add_single_message(
                    token_ref[0],
                    original_chat_id,
                    RoleEnum.USER,
                    user_query,
                    scenario_id=scenario_id,
                )
            except Exception as exc:
                logger.warning(f"Urban data QA: failed to persist user question: {exc}")

        async for event in self._run_qa_loop(
            urban_data_mcp_client,
            model,
            temperature,
            user_query,
            history,
            collected,
            request_id,
            token_ref,
            scenario_id,
        ):
            yield event

        # Persist only when this run actually produced the answer — never on a reconnect
        # that merely replayed an already-completed pipeline (avoids duplicate messages).
        if persist_history and collected.get("newly_completed"):
            self._schedule_persist_answer(token_ref[0], chat_id, collected, scenario_id)

    # ------------------------------------------------------------------
    # Inner loop (native Ollama tool-calling: plan/execute are the model's own choice)
    # ------------------------------------------------------------------

    async def _run_qa_loop(
        self,
        urban_data_mcp_client: "UrbanDataMcpClient",
        model: str,
        temperature: float,
        user_query: str,
        history: list[dict],
        collected: dict[str, Any],
        request_id: str,
        token_ref: list[str],
        scenario_id: int | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        checkpoint = await self.state_store.get_checkpoint(request_id)
        progress = checkpoint.get(_QA_PROGRESS) or {}
        if progress.get("accepted"):
            # The pipeline already produced the final answer before the disconnect; its
            # terminal events were buffered and have just been replayed — nothing to redo.
            collected["final_answer"] = progress.get("final_answer", "")
            collected["tool_calls"] = list(progress.get("tool_calls", []))
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        yield self._buf(
            request_id,
            self._status("tools_loading", "Получаю список доступных инструментов…"),
        )
        tools_out: list[list[dict]] = []
        try:
            async for event in self._retryable_step(
                request_id,
                urban_data_mcp_client,
                token_ref,
                lambda: urban_data_mcp_client.get_tools(),
                tools_out,
            ):
                yield self._buf(request_id, event)
        except PipelineSuspendedError:
            return
        tools = tools_out[0] if tools_out else []

        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt(scenario_id)},
            *history,
            {"role": "user", "content": user_query},
        ]

        for iteration in range(1, self.MAX_TOOL_ITERATIONS + 1):
            yield self._buf(
                request_id,
                self._status(
                    "executing",
                    f"Определяю, какие данные нужны для ответа (шаг {iteration})…",
                ),
            )
            chat_out: list[ChatResponse] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    urban_data_mcp_client,
                    token_ref,
                    lambda: self.llm_client.chat(
                        model,
                        messages,
                        tools=tools,
                        options={"temperature": temperature},
                        stream=False,
                    ),
                    chat_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            response = chat_out[0]
            tool_calls = response.message.tool_calls or []
            if not tool_calls:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.message.content or "",
                    "tool_calls": self._serialize_tool_calls(tool_calls),
                }
            )
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = dict(tool_call.function.arguments or {})
                yield self._buf(
                    request_id,
                    self._status("executing", f"Запрашиваю данные: «{tool_name}»…"),
                )
                result_out: list[Any] = []
                try:
                    async for event in self._retryable_step(
                        request_id,
                        urban_data_mcp_client,
                        token_ref,
                        lambda tn=tool_name, ta=arguments: urban_data_mcp_client.execute_tool(
                            tn, ta
                        ),
                        result_out,
                    ):
                        yield self._buf(request_id, event)
                except PipelineSuspendedError:
                    return
                result = result_out[0] if result_out else None

                call_record = {"function": {"name": tool_name, "arguments": arguments}}
                collected["tool_calls"].append(call_record)
                yield self._buf(
                    request_id,
                    self._tool_call(
                        _EXECUTION_MODE, [call_record], mcp_source=_MCP_SOURCE
                    ),
                )
                for fc_event in self._feature_collections(tool_name, result):
                    yield self._buf(request_id, fc_event)

                messages.append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": json.dumps(
                            self._strip_geometries(result), ensure_ascii=False
                        ),
                    }
                )

        yield self._buf(request_id, self._status("answer_drafting", "Формирую ответ…"))
        draft_parts: list[str] = []
        async for chunk_event in self._generate_answer(model, messages, temperature):
            if text := chunk_event["content"]["text"]:
                draft_parts.append(text)
            yield self._buf(request_id, chunk_event)
        collected["final_answer"] = "".join(draft_parts).strip()
        collected["newly_completed"] = True

        await self.state_store.save_checkpoint(
            request_id,
            _QA_PROGRESS,
            {
                "accepted": True,
                "final_answer": collected["final_answer"],
                "tool_calls": collected["tool_calls"],
            },
        )
        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # Final answer generation (streaming, no tools — grounded in the gathered context)
    # ------------------------------------------------------------------

    async def _generate_answer(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
    ) -> AsyncGenerator[dict[str, Any], None]:
        response_buffer: list[str] = []
        async for part in await self.llm_client.chat(
            model,
            messages,
            options={"temperature": temperature},
            stream=True,
        ):
            part: ChatResponse
            if part.message.content:
                response_buffer.append(part.message.content)
            yield self._chunk(part.message.content or "", done=part.done)
        logger.debug(
            f"Urban data QA final answer [{model}]: {''.join(response_buffer)}"
        )

    # ------------------------------------------------------------------
    # Tool-result shaping: hide raw geometry from the LLM, keep it for the client
    # ------------------------------------------------------------------

    @classmethod
    def _strip_geometries(cls, data: Any) -> Any:
        """Recursively replace embedded geometries with a placeholder for the LLM context."""
        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection" and isinstance(
                data.get("features"), list
            ):
                return {
                    "type": "FeatureCollection",
                    "feature_count": len(data["features"]),
                    "sample_properties": [
                        feature.get("properties")
                        for feature in data["features"][:3]
                        if isinstance(feature, dict)
                    ],
                }
            if (
                data.get("type")
                in {
                    "Point",
                    "LineString",
                    "Polygon",
                    "MultiPoint",
                    "MultiLineString",
                    "MultiPolygon",
                    "GeometryCollection",
                }
                and "coordinates" in data
            ):
                return {"type": data["type"], "coordinates": _GEOMETRY_PLACEHOLDER}
            return {key: cls._strip_geometries(value) for key, value in data.items()}
        if isinstance(data, list):
            return [cls._strip_geometries(item) for item in data]
        return data

    @staticmethod
    def _feature_collections(name_prefix: str, data: Any):
        """Recursively yield feature_collection events for FeatureCollections in a tool result."""
        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection" and isinstance(
                data.get("features"), list
            ):
                yield {
                    "type": "feature_collection",
                    "content": {"name": name_prefix, "feature_collection": data},
                }
                return
            for key, value in data.items():
                yield from UrbanDataQaService._feature_collections(
                    f"{name_prefix}.{key}", value
                )
        elif isinstance(data, list):
            for index, item in enumerate(data):
                yield from UrbanDataQaService._feature_collections(
                    f"{name_prefix}[{index}]", item
                )

    @staticmethod
    def _system_prompt(scenario_id: int | None) -> str:
        """Append the current scenario_id (if any) so the model can fill scenario/project
        -scoped tool arguments (e.g. ``GetScenarioServices``, ``GetProjectById``) without
        asking the user to repeat it — those are ordinary, visible required parameters on
        the tools themselves, not hidden protocol metadata."""
        if scenario_id is None:
            return (
                f"{SYSTEM_PROMPT}\n\nТекущий scenario_id не выбран. Если пользователь "
                "просит данные по конкретному проекту/сценарию, а нужный "
                "scenario_id/project_id не следует явно из вопроса — уточни его."
            )
        return f"{SYSTEM_PROMPT}\n\nТекущий scenario_id: {scenario_id}."

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[Any]) -> list[dict]:
        return [
            {
                "function": {
                    "name": tool_call.function.name,
                    "arguments": dict(tool_call.function.arguments or {}),
                }
            }
            for tool_call in tool_calls
        ]

    # ------------------------------------------------------------------
    # Token-refresh retry wrapper (mirrors RestrictionParserService/ProvisionService)
    # ------------------------------------------------------------------

    async def _retryable_step(
        self,
        request_id: str,
        urban_data_mcp_client: "UrbanDataMcpClient",
        token_ref: list[str],
        step_fn: Callable,
        result: list,
    ) -> AsyncGenerator[dict, None]:
        while True:
            try:
                result.append(await step_fn())
                return
            except TokenExpiredError:
                logger.warning(
                    f"Token expired for request_id={request_id}, waiting for refresh"
                )
                yield self._token_expired_event(request_id)
                await self.state_store.set_status(
                    request_id, PipelineStatus.WAITING_TOKEN
                )
                try:
                    new_token = await asyncio.wait_for(
                        self.state_store.wait_for_token(request_id),
                        timeout=TOKEN_REFRESH_TIMEOUT,
                    )
                    urban_data_mcp_client.update_token(new_token)
                    token_ref[0] = new_token
                    await self.state_store.set_status(
                        request_id, PipelineStatus.RUNNING
                    )
                    logger.info(
                        f"Token refreshed for request_id={request_id}, retrying"
                    )
                except asyncio.TimeoutError:
                    await self.state_store.set_status(
                        request_id, PipelineStatus.SUSPENDED
                    )
                    yield self._pipeline_suspended_event(request_id)
                    raise PipelineSuspendedError(request_id)

    def _buf(self, request_id: str, event: dict) -> dict:
        """Fire-and-forget buffer the event for reconnect replay, then return it."""
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    # ------------------------------------------------------------------
    # Chat storage persistence (final answer + tool calls)
    # ------------------------------------------------------------------

    def _schedule_persist_answer(
        self,
        token: str,
        chat_id: str | None,
        collected: dict[str, Any],
        scenario_id: int | None,
    ) -> None:
        if not chat_id or not collected.get("final_answer"):
            return
        task = asyncio.create_task(
            self._persist_answer(token, chat_id, dict(collected), scenario_id)
        )
        task.add_done_callback(self._log_persist_result)

    async def _persist_answer(
        self,
        token: str,
        chat_id: str,
        collected: dict[str, Any],
        scenario_id: int | None,
    ) -> None:
        parts: list[TextPartRequest | ToolCallPartRequest] = []
        if collected.get("tool_calls"):
            calls = [
                self._tool_call_to_storage(step, tc)
                for step, tc in enumerate(collected["tool_calls"], start=1)
            ]
            parts.append(
                ToolCallPartRequest(
                    kind="tool_call",
                    payload=ToolCallPayload(
                        execution_mode=_EXECUTION_MODE, calls=calls
                    ),
                    mcp_source=_MCP_SOURCE,
                )
            )
        parts.append(
            TextPartRequest(
                kind="text", payload=TextPayload(text=collected["final_answer"])
            )
        )
        await self.add_complex_message(
            token, chat_id, RoleEnum.ASSISTANT, parts, scenario_id=scenario_id
        )

    @staticmethod
    def _log_persist_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.exception(f"Urban data QA: failed to persist answer: {exc}")

    # ------------------------------------------------------------------
    # Event / tool-call helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_call_to_storage(step: int, tool_call: dict) -> ToolCall:
        function_call = tool_call.get("function") or {}
        tool_name = function_call.get("name") or tool_call.get("name")
        arguments = function_call.get("arguments") or tool_call.get("arguments") or {}
        if not tool_name:
            raise ValueError(f"Tool call without tool name: {tool_call}")
        return ToolCall(step=step, tool_name=tool_name, arguments=arguments)

    @staticmethod
    def _pipeline_started_event(request_id: str) -> dict:
        return {"type": "pipeline_started", "content": {"request_id": request_id}}

    @staticmethod
    def _project_lookup_failed_event(scenario_id: int | None) -> dict:
        return {
            "type": "warning",
            "content": {
                "code": "project_id_unavailable",
                "scenario_id": scenario_id,
                "message": (
                    f"Не удалось получить идентификатор проекта (project_id) по "
                    f"scenario_id={scenario_id}. Фильтр проекта не будет сохранён, "
                    "выполнение запроса продолжается."
                ),
            },
        }

    @staticmethod
    def _token_expired_event(request_id: str) -> dict:
        return {
            "type": "token_expired",
            "content": {
                "request_id": request_id,
                "message": "Token expired. Update token to continue request procedure.",
            },
        }

    @staticmethod
    def _pipeline_suspended_event(request_id: str) -> dict:
        return {
            "type": "pipeline_suspended",
            "content": {
                "request_id": request_id,
                "message": (
                    "Выполнение приостановлено: токен не был обновлён вовремя. "
                    "Переподключитесь с тем же request_id, чтобы продолжить."
                ),
            },
        }

    @staticmethod
    def _status(status: str, text: str) -> dict:
        return {"type": "status", "content": {"status": status, "text": text}}

    @staticmethod
    def _chunk(text: str, done: bool) -> dict:
        return {"type": "chunk", "content": {"text": text, "done": done}}

    @staticmethod
    def _tool_call(
        execution_mode: str, tool_calls: list[dict], mcp_source: str | None = None
    ) -> dict:
        content: dict = {"execution_mode": execution_mode, "tool_calls": tool_calls}
        if mcp_source is not None:
            content["mcp_source"] = mcp_source
        return {"type": "tool_call", "content": content}

    @staticmethod
    def _chat_created_event(chat_id: str, chat_title: str) -> dict:
        return {
            "type": "service_event",
            "content": {
                "event_type": "storage_event",
                "event": {
                    "storage_event_type": "chat_created",
                    "chat_id": chat_id,
                    "chat_title": chat_title,
                },
            },
        }
