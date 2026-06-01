from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from ollama import ChatResponse

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    StatusPartRequest,
    StatusPayload,
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
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.pipeline_state import (
    PIPELINE_TTL,
    TOKEN_REFRESH_TIMEOUT,
    PipelineStateStore,
    PipelineStatus,
    PipelineStep,
)
from src.agents.services.provision_context import ProvisionContextBuilder
from src.agents.services.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.provision_tool_executor import ProvisionToolExecutor
from src.agents.services.service_entities.provision_plan import (
    ProvisionPlan,
    ProvisionPlanMode,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


class ProvisionService(BaseLlmService):
    """
    Service for running provision effects pipelines.
    Pipeline steps:
        1. GET_SERVICE_ID  — resolve service_type_id via IDU MCP GetServices.
        2. CALCULATE_EFFECTS — call CalculateObjectEffects on the effects MCP server.
        3. FINAL_RESPONSE  — LLM analysis of pivot data.
    """

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client: ChatStorageApiClient,
        state_store: PipelineStateStore,
    ) -> None:

        super().__init__(ollama_host, chat_storage_client)
        self.plan_builder = ProvisionPlanBuilder(self.llm_client)
        self.tool_executor = ProvisionToolExecutor()
        self.context_builder = ProvisionContextBuilder()
        self.state_store = state_store

    # ------------------------------------------------------------------
    # Public entry point (SSE outer wrapper — mirrors RestrictionParserService)
    # ------------------------------------------------------------------

    async def run_provision_pipeline(
        self,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int,
        chat_id: str | None = None,
        request_id: str | None = None,
    ) -> AsyncGenerator:
        token_ref: list[str] = [
            idu_mcp_client.mcp_client.transport.auth.token.get_secret_value()
        ]
        text_buffer: list[str] = []
        message_parts: list[
            TextPartRequest | StatusPartRequest | ToolCallPartRequest
        ] = []

        async for item in self._run_provision_pipeline(
            idu_mcp_client=idu_mcp_client,
            effects_mcp_client=effects_mcp_client,
            model=model,
            temperature=temperature,
            user_query=user_query,
            scenario_id=scenario_id,
            chat_id=chat_id,
            request_id=request_id,
            token_ref=token_ref,
        ):
            chat_id = self._chat_id_from_storage_event(item) or chat_id
            if item.get("type") == "tool_call":
                self._flush_text_buffer_to_parts(text_buffer, message_parts)
                content = item.get("content", {})
                self._add_tool_calls_to_parts(
                    message_parts,
                    content.get("tool_calls", []),
                    execution_mode=content.get("execution_mode", ""),
                    mcp_source=content.get("mcp_source"),
                )
                continue
            if item.get("type") == "chunk":
                content = item.get("content", {})
                if content.get("text"):
                    text_buffer.append(content["text"])
                if content.get("done"):
                    self._flush_text_buffer_to_parts(text_buffer, message_parts)
                yield item
                continue
            self._flush_text_buffer_to_parts(text_buffer, message_parts)
            part = self._pipeline_item_to_chat_part(item)
            if part is not None:
                message_parts.append(part)
            yield item

        if text_buffer:
            self._flush_text_buffer_to_parts(text_buffer, message_parts)
        self._schedule_add_message_parts_to_chat(
            token_ref[0],
            chat_id,
            message_parts,
            scenario_id=scenario_id,
        )

    # ------------------------------------------------------------------
    # Inner pipeline
    # ------------------------------------------------------------------

    async def _run_provision_pipeline(
        self,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int,
        token_ref: list[str],
        chat_id: str | None = None,
        request_id: str | None = None,
    ) -> AsyncGenerator:
        is_reconnect = request_id is not None and await self.state_store.exists(
            request_id
        )
        if is_reconnect:
            logger.info(f"Reconnect for request_id={request_id}, replaying events")
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            if not chat_id:
                stored = await self.state_store.get_state(request_id)
                if stored and stored.get("chat_id"):
                    chat_id = stored["chat_id"]
        else:
            request_id = request_id or self.state_store.new_request_id()

        original_chat_id = chat_id
        if not is_reconnect:
            yield self._buf(request_id, self._pipeline_started_event(request_id))

            if not chat_id:
                chat_result: list[tuple[str, str]] = []
                try:
                    async for event in self._retryable_step(
                        request_id,
                        idu_mcp_client,
                        effects_mcp_client,
                        token_ref,
                        lambda: self.create_chat(
                            token_ref[0],
                            model,
                            user_query,
                            additional_instructions=(
                                "Первый запрос пользователя был отправлен к сервису "
                                "расчёта эффектов обеспеченности."
                            ),
                            scenario_id=scenario_id,
                        ),
                        chat_result,
                    ):
                        yield self._buf(request_id, event)
                except PipelineSuspendedError:
                    return
                chat_id, title = chat_result[0]
                yield self._buf(request_id, self._chat_created_event(chat_id, title))

            await self.state_store.create(
                request_id,
                chat_id=chat_id,
                user_query=user_query,
                scenario_id=scenario_id,
                model=model,
                temperature=temperature,
            )

        logger.info(f"Provision pipeline request_id={request_id} chat_id={chat_id}")

        llm_history: list[dict] = []
        if original_chat_id:
            try:
                chat_info = await self.get_chat_messages(token_ref[0], original_chat_id)
                llm_history = self.build_llm_history(chat_info.messages)
            except Exception as exc:
                logger.warning(f"Failed to fetch chat history: {exc}")

        checkpoint = await self.state_store.get_checkpoint(request_id)

        # ── Step 0: RESOLVE_SERVICE ───────────────────────────────────
        yield self._buf(
            request_id,
            self._status("service_lookup", "Определяю сервис и параметры из запроса"),
        )
        if PipelineStep.RESOLVE_SERVICE not in checkpoint:
            plan_out: list[ProvisionPlan] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    token_ref,
                    lambda: self._resolve_service_plan(
                        idu_mcp_client, model, user_query, scenario_id, llm_history
                    ),
                    plan_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            plan = plan_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.RESOLVE_SERVICE, plan.model_dump(mode="json")
            )
        else:
            plan = ProvisionPlan.model_validate(
                checkpoint[PipelineStep.RESOLVE_SERVICE]
            )

        if plan.mode == ProvisionPlanMode.NEEDS_CLARIFICATION:
            yield self._buf(
                request_id,
                self._status(
                    "service_lookup", "Не удалось определить сервис из запроса"
                ),
            )
            yield self._buf(
                request_id,
                self._chunk(
                    plan.clarification_question
                    or "Уточните, для какого сервиса рассчитать эффекты обеспеченности.",
                    done=True,
                ),
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        service_name: str = plan.service_name  # type: ignore[assignment]
        target_population: int | None = plan.target_population

        # ── Step 1: GET_SERVICE_ID ────────────────────────────────────
        yield self._buf(
            request_id,
            self._status("service_lookup", f"Ищу сервис «{service_name}» в каталоге"),
        )
        if PipelineStep.GET_SERVICE_ID not in checkpoint:
            svc_out: list[Any] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    token_ref,
                    lambda: self.tool_executor.get_service_id(
                        idu_mcp_client, service_name
                    ),
                    svc_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            svc_result = svc_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.GET_SERVICE_ID, svc_result.data
            )
        else:
            from src.agents.services.provision_tool_executor import ProvisionStepResult

            svc_result = ProvisionStepResult(
                data=checkpoint[PipelineStep.GET_SERVICE_ID], tool_calls=[]
            )

        service_type_id: int = svc_result.data["service_type_id"]
        yield self._buf(
            request_id,
            self._tool_call(
                "service_lookup", svc_result.tool_calls, mcp_source="IDU_MCP_URL"
            ),
        )
        yield self._buf(
            request_id,
            self._status("service_lookup", f"Сервис найден: id={service_type_id}"),
        )

        # ── Step 2: CALCULATE_EFFECTS ─────────────────────────────────
        yield self._buf(
            request_id,
            self._status(
                "effects_calculation",
                "Рассчитываю эффекты обеспеченности для сценария проекта",
            ),
        )
        if PipelineStep.CALCULATE_EFFECTS not in checkpoint:
            eff_out: list[Any] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    token_ref,
                    lambda: self.tool_executor.calculate_effects(
                        effects_mcp_client,
                        service_type_id,
                        scenario_id,
                        target_population,
                    ),
                    eff_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            eff_result = eff_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.CALCULATE_EFFECTS, eff_result.data
            )
        else:
            from src.agents.services.provision_tool_executor import ProvisionStepResult

            eff_result = ProvisionStepResult(
                data=checkpoint[PipelineStep.CALCULATE_EFFECTS], tool_calls=[]
            )

        effects_data: dict = eff_result.data
        yield self._buf(
            request_id,
            self._tool_call(
                "effects_calculation",
                eff_result.tool_calls,
                mcp_source="OBJECTS_EFFECTS_MCP_URL",
            ),
        )
        yield self._buf(
            request_id,
            self._status("effects_calculation", "Расчёт эффектов завершён"),
        )
        for item in self._effects_feature_collections(effects_data):
            yield self._buf(request_id, item)

        # ── Step 3: FINAL_RESPONSE ────────────────────────────────────
        if PipelineStep.FINAL_RESPONSE not in checkpoint:
            context = self.context_builder.build_context(effects_data, service_name)
            async for chunk in self._generate_analysis(
                model, user_query, context, temperature, history=llm_history
            ):
                yield self._buf(request_id, chunk)
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # Token-refresh retry wrapper (mirrors RestrictionParserService)
    # ------------------------------------------------------------------

    async def _retryable_step(
        self,
        request_id: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
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
                    idu_mcp_client.update_token(new_token)
                    effects_mcp_client.update_token(new_token)
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
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    # ------------------------------------------------------------------
    # Service plan resolution
    # ------------------------------------------------------------------

    async def _resolve_service_plan(
        self,
        idu_mcp_client: IduMcpClient,
        model: str,
        user_query: str,
        scenario_id: int,
        history: list[dict] | None = None,
    ) -> ProvisionPlan:
        services_catalog = await self.plan_builder.get_services_catalog(
            idu_mcp_client, scenario_id
        )
        return await self.plan_builder.build_plan(
            model, user_query, services_catalog, history
        )

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    async def _generate_analysis(
        self,
        model: str,
        user_query: str,
        context: str,
        temperature: float,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[dict, None]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Проанализируй эффекты обеспеченности на основе сводных данных расчёта. "
                    "Ответ формулируй как аналитический комментарий: что изменилось, "
                    "насколько значимы изменения, какие выводы можно сделать. "
                    "Используй только числа и факты из контекста.\n\n"
                    f"Контекст расчёта:\n{context}"
                ),
            },
            *(history or []),
            {"role": "user", "content": user_query},
        ]
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
        logger.debug(f"LLM provision analysis [{model}]: {''.join(response_buffer)}")

    # ------------------------------------------------------------------
    # Chat storage helpers (mirrors RestrictionParserService)
    # ------------------------------------------------------------------

    async def _add_message_parts_to_chat(
        self,
        token: str,
        chat_id: str | None,
        parts: list[TextPartRequest | StatusPartRequest | ToolCallPartRequest],
        **metadata,
    ) -> None:
        if not chat_id or not parts:
            return
        await self.add_complex_message(
            token, chat_id, RoleEnum.ASSISTANT, parts, **metadata
        )

    def _schedule_add_message_parts_to_chat(
        self,
        token: str,
        chat_id: str | None,
        parts: list[TextPartRequest | StatusPartRequest | ToolCallPartRequest],
        **metadata,
    ) -> None:
        if not chat_id or not parts:
            return
        task = asyncio.create_task(
            self._add_message_parts_to_chat(token, chat_id, parts.copy(), **metadata)
        )
        task.add_done_callback(self._log_message_upload_result)

    @staticmethod
    def _log_message_upload_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.exception(f"Failed to upload provision response message: {exc}")

    @staticmethod
    def _flush_text_buffer_to_parts(
        text_buffer: list[str],
        parts: list[TextPartRequest | StatusPartRequest | ToolCallPartRequest],
    ) -> None:
        if not text_buffer:
            return
        parts.append(
            TextPartRequest(kind="text", payload=TextPayload(text="".join(text_buffer)))
        )
        text_buffer.clear()

    def _add_tool_calls_to_parts(
        self,
        parts: list[TextPartRequest | StatusPartRequest | ToolCallPartRequest],
        tool_calls: list[dict],
        execution_mode: str,
        mcp_source: str | None = None,
    ) -> None:
        if not tool_calls:
            return
        calls = [
            self._tool_call_to_chat_storage_call(step, tc)
            for step, tc in enumerate(tool_calls, start=1)
        ]
        parts.append(
            ToolCallPartRequest(
                kind="tool_call",
                payload=ToolCallPayload(execution_mode=execution_mode, calls=calls),
                mcp_source=mcp_source,
            )
        )

    @staticmethod
    def _pipeline_item_to_chat_part(
        item: dict,
    ) -> TextPartRequest | StatusPartRequest | None:
        item_type = item.get("type")
        content = item.get("content") or {}
        if item_type == "status":
            return StatusPartRequest(
                kind="status",
                payload=StatusPayload(
                    status=content.get("status", ""), text=content.get("text", "")
                ),
            )
        if item_type == "chunk":
            text = content.get("text") or ""
            return (
                TextPartRequest(kind="text", payload=TextPayload(text=text))
                if text
                else None
            )
        return None

    @staticmethod
    def _tool_call_to_chat_storage_call(step: int, tool_call: dict) -> ToolCall:
        function_call = tool_call.get("function") or {}
        tool_name = (
            tool_call.get("tool_name")
            or tool_call.get("name")
            or function_call.get("name")
        )
        arguments = tool_call.get("arguments") or function_call.get("arguments") or {}
        if not tool_name:
            raise ValueError(f"Tool call without tool name: {tool_call}")
        return ToolCall(step=step, tool_name=tool_name, arguments=arguments)

    @staticmethod
    def _chat_id_from_storage_event(item: dict) -> str | None:
        event_container = item.get("content") or item
        event = event_container.get("event") or {}
        if event.get("storage_event_type") == "chat_created":
            return event.get("chat_id")
        return None

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pipeline_started_event(request_id: str) -> dict:
        return {"type": "pipeline_started", "content": {"request_id": request_id}}

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

    @staticmethod
    def _status(status: str, text: str) -> dict:
        return {"type": "status", "content": {"status": status, "text": text}}

    @staticmethod
    def _chunk(text: str, done: bool) -> dict:
        return {"type": "chunk", "content": {"text": text, "done": done}}

    @staticmethod
    def _tool_call(
        execution_mode: str,
        tool_calls: list[dict],
        mcp_source: str | None = None,
    ) -> dict:
        content: dict = {"execution_mode": execution_mode, "tool_calls": tool_calls}
        if mcp_source is not None:
            content["mcp_source"] = mcp_source
        return {"type": "tool_call", "content": content}

    @staticmethod
    def _feature_collections(layers: dict[str, dict]):
        for name, fc in layers.items():
            if isinstance(fc, dict) and fc.get("type") == "FeatureCollection":
                yield {
                    "type": "feature_collection",
                    "content": {"name": name, "feature_collection": fc},
                }

    @staticmethod
    def _effects_feature_collections(effects_data: dict):
        """Yield feature_collection events from nested effects result structure."""
        for group_key in ("before_prove_data", "after_prove_data"):
            group = effects_data.get(group_key) or {}
            if isinstance(group, dict):
                for layer_name, fc in group.items():
                    if isinstance(fc, dict) and fc.get("type") == "FeatureCollection":
                        yield {
                            "type": "feature_collection",
                            "content": {
                                "name": f"{group_key}.{layer_name}",
                                "feature_collection": fc,
                            },
                        }
        effects_fc = effects_data.get("effects")
        if (
            isinstance(effects_fc, dict)
            and effects_fc.get("type") == "FeatureCollection"
        ):
            yield {
                "type": "feature_collection",
                "content": {"name": "effects", "feature_collection": effects_fc},
            }
