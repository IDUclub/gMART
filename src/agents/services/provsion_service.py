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
    TableColumn,
    TablePartRequest,
    TablePayload,
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
    PIPELINE_TTL,
    TOKEN_REFRESH_TIMEOUT,
    PipelineStateStore,
    PipelineStatus,
    PipelineStep,
)
from src.agents.services.provision_context import ProvisionContextBuilder
from src.agents.services.provision_plan_builder import ProvisionPlanBuilder
from src.agents.services.provision_tool_executor import (
    ProvisionStepResult,
    ProvisionToolExecutor,
)
from src.agents.services.service_entities.provision_plan import (
    ProvisionPlan,
    ProvisionPlanMode,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient

MessagePart = (
    TextPartRequest | StatusPartRequest | ToolCallPartRequest | TablePartRequest
)

EFFECTS_ANALYSIS_INSTRUCTIONS = (
    "Проанализируй эффекты обеспеченности на основе сводных данных расчёта. "
    "Ответ формулируй как аналитический комментарий: что изменилось, "
    "насколько значимы изменения, какие выводы можно сделать. "
    "Используй только числа и факты из контекста. "
    "Не оформляй таблицы — сводная таблица уже показана пользователю отдельно."
)
PROVISION_ANALYSIS_INSTRUCTIONS = (
    "Проанализируй текущую обеспеченность сервисом на основе сводных данных расчёта. "
    "Ответ формулируй как аналитический комментарий: насколько спрос покрыт "
    "вместимостью, есть ли дефицит и насколько он значим. "
    "Используй только числа и факты из контекста. "
    "Не оформляй таблицы — таблица показателей уже показана пользователю отдельно."
)
SUMMARY_ANALYSIS_INSTRUCTIONS = (
    "Проанализируй сводку обеспеченности городскими сервисами. "
    "Укажи, какими сервисами территория обеспечена хуже всего (наибольший дефицит) "
    "и какими лучше всего, и какие выводы можно сделать. "
    "Используй только числа и факты из контекста. "
    "Не оформляй таблицы — сводная таблица уже показана пользователю отдельно."
)
POPULATION_HINT = (
    "\n\nРасчёт выполнен по населению, восстановленному из данных Urban API. "
    "При желании укажите целевую численность населения прямо в запросе — "
    "например: «…при населении 25 000 человек» — и расчёт будет выполнен с ней."
)


class ProvisionService(BaseLlmService):
    """
    Service for running provision pipelines.

    The first LLM call classifies the user query into one of the plan modes
    (see ProvisionPlanMode); every mode then runs a deterministic,
    checkpointed pipeline:
        - effects:        GetServiceTypeIdByName -> CalculateObjectEffects
                          -> layers + strict pivot table + LLM commentary.
        - provision:      CalculateServicesProvision (single service, layers on)
                          -> layers + strict metrics table + LLM commentary.
        - summary:        CalculateServicesProvision (catalog, layers off unless
                          explicitly requested) -> strict deficit/surplus table
                          + LLM commentary.
        - list_services:  deterministic text listing the scenario catalog.
    """

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client: ChatStorageApiClient,
        urban_api_client: UrbanApiClient,
        state_store: PipelineStateStore,
    ) -> None:

        super().__init__(ollama_host, chat_storage_client, urban_api_client)
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
        message_parts: list[MessagePart] = []

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
            self._status("service_lookup", "Определяю тип запроса и параметры"),
        )
        if PipelineStep.RESOLVE_SERVICE not in checkpoint:
            resolve_out: list[tuple[ProvisionPlan, dict[str, int]]] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    token_ref,
                    lambda: self._resolve_service_plan(
                        token_ref[0], model, user_query, scenario_id, llm_history
                    ),
                    resolve_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            plan, service_types = resolve_out[0]
            await self.state_store.save_checkpoint(
                request_id,
                PipelineStep.RESOLVE_SERVICE,
                {
                    "plan": plan.model_dump(mode="json"),
                    "service_types": service_types,
                },
            )
        else:
            plan, service_types = self._load_resolve_checkpoint(
                checkpoint[PipelineStep.RESOLVE_SERVICE]
            )

        if plan.mode == ProvisionPlanMode.NEEDS_CLARIFICATION:
            yield self._buf(
                request_id,
                self._status(
                    "service_lookup", "Не удалось определить тип запроса из сообщения"
                ),
            )
            yield self._buf(
                request_id,
                self._chunk(
                    plan.clarification_question
                    or "Уточните, какой анализ обеспеченности сервисами нужен.",
                    done=True,
                ),
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        if plan.mode == ProvisionPlanMode.LIST_SERVICES:
            async for event in self._run_list_services(request_id, service_types):
                yield event
            return

        if plan.mode == ProvisionPlanMode.SUMMARY:
            async for event in self._run_summary(
                request_id,
                idu_mcp_client,
                effects_mcp_client,
                token_ref,
                checkpoint,
                plan,
                service_types,
                model,
                temperature,
                user_query,
                scenario_id,
                llm_history,
            ):
                yield event
            return

        if plan.mode == ProvisionPlanMode.PROVISION:
            async for event in self._run_single_provision(
                request_id,
                idu_mcp_client,
                effects_mcp_client,
                token_ref,
                checkpoint,
                plan,
                service_types,
                model,
                temperature,
                user_query,
                scenario_id,
                llm_history,
            ):
                yield event
            return

        # Default: ProvisionPlanMode.EFFECTS — the original before/after pipeline
        async for event in self._run_effects(
            request_id,
            idu_mcp_client,
            effects_mcp_client,
            token_ref,
            checkpoint,
            plan,
            model,
            temperature,
            user_query,
            scenario_id,
            llm_history,
        ):
            yield event

    # ------------------------------------------------------------------
    # Mode: list_services
    # ------------------------------------------------------------------

    async def _run_list_services(
        self,
        request_id: str,
        service_types: dict[str, int],
    ) -> AsyncGenerator:
        yield self._buf(
            request_id,
            self._status("service_lookup", "Собираю список доступных сервисов"),
        )
        names = sorted(service_types)
        if names:
            listing = "\n".join(f"- {name}" for name in names)
            text = (
                "В проекте и его контексте доступны следующие сервисы:\n"
                f"{listing}\n\n"
                "Могу рассчитать сводку обеспеченности по всем сервисам "
                "или обеспеченность и эффекты по конкретному сервису."
            )
        else:
            text = "В сценарии нет доступных для анализа сервисов."
        yield self._buf(request_id, self._chunk(text, done=True))
        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # Mode: summary
    # ------------------------------------------------------------------

    async def _run_summary(
        self,
        request_id: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        token_ref: list[str],
        checkpoint: dict,
        plan: ProvisionPlan,
        service_types: dict[str, int],
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int,
        llm_history: list[dict],
    ) -> AsyncGenerator:
        selected_names = plan.service_names or list(service_types)
        services_args = {
            service_types[name]: {
                "name": name,
                "as_layer": name in plan.layer_service_names,
            }
            for name in selected_names
            if name in service_types
        }
        if not services_args:
            yield self._buf(
                request_id,
                self._chunk(
                    "В сценарии нет доступных сервисов для расчёта сводки.", done=True
                ),
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        yield self._buf(
            request_id,
            self._status(
                "effects_calculation",
                f"Рассчитываю обеспеченность по {len(services_args)} сервисам",
            ),
        )
        prov_out: list[ProvisionStepResult] = []
        async for event in self._checkpointed_provision_step(
            request_id,
            idu_mcp_client,
            effects_mcp_client,
            token_ref,
            checkpoint,
            scenario_id,
            services_args,
            plan.target_population,
            prov_out,
        ):
            yield event
        if not prov_out:
            return
        prov_result = prov_out[0]

        yield self._buf(
            request_id,
            self._tool_call(
                "effects_calculation",
                prov_result.tool_calls,
                mcp_source="OBJECTS_EFFECTS_MCP_URL",
            ),
        )
        yield self._buf(
            request_id,
            self._status("effects_calculation", "Расчёт обеспеченности завершён"),
        )
        for item in self._provision_feature_collections(prov_result.data):
            yield self._buf(request_id, item)

        table = self.context_builder.build_summary_table(prov_result.data)
        if table["rows"]:
            yield self._buf(request_id, self._table(table))
        else:
            failed = [
                f"{svc.get('name', '')}: {svc.get('error', 'нет данных')}"
                for svc in (prov_result.data.get("services") or {}).values()
            ]
            yield self._buf(
                request_id,
                self._chunk(
                    "Не удалось рассчитать обеспеченность ни для одного сервиса.\n"
                    + "\n".join(failed),
                    done=True,
                ),
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        if PipelineStep.FINAL_RESPONSE not in checkpoint:
            context = self.context_builder.build_summary_context(prov_result.data)
            if plan.target_population is not None:
                context += (
                    f"\n\nРасчёт выполнен для заданной пользователем численности "
                    f"населения: {plan.target_population} человек."
                )
            async for chunk in self._generate_analysis(
                model,
                user_query,
                context,
                temperature,
                history=llm_history,
                instructions=SUMMARY_ANALYSIS_INSTRUCTIONS,
                trailing_note=(
                    POPULATION_HINT if plan.target_population is None else None
                ),
            ):
                yield self._buf(request_id, chunk)
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # Mode: provision (single service, current state)
    # ------------------------------------------------------------------

    async def _run_single_provision(
        self,
        request_id: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        token_ref: list[str],
        checkpoint: dict,
        plan: ProvisionPlan,
        service_types: dict[str, int],
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int,
        llm_history: list[dict],
    ) -> AsyncGenerator:
        service_name: str = plan.service_name  # type: ignore[assignment]
        service_type_id = service_types.get(service_name)

        if service_type_id is None:
            # Fallback for legacy checkpoints without the service_types map
            svc_out: list[ProvisionStepResult] = []
            async for event in self._get_service_id_step(
                request_id,
                idu_mcp_client,
                effects_mcp_client,
                token_ref,
                checkpoint,
                service_name,
                svc_out,
            ):
                yield event
            if not svc_out:
                return
            svc_result = svc_out[0]
            service_type_id = svc_result.data["service_type_id"]
            yield self._buf(
                request_id,
                self._tool_call(
                    "service_lookup", svc_result.tool_calls, mcp_source="IDU_MCP_URL"
                ),
            )

        yield self._buf(
            request_id,
            self._status(
                "effects_calculation",
                f"Рассчитываю текущую обеспеченность сервисом «{service_name}»",
            ),
        )
        services_args = {
            service_type_id: {"name": service_name, "as_layer": True},
        }
        prov_out: list[ProvisionStepResult] = []
        async for event in self._checkpointed_provision_step(
            request_id,
            idu_mcp_client,
            effects_mcp_client,
            token_ref,
            checkpoint,
            scenario_id,
            services_args,
            plan.target_population,
            prov_out,
        ):
            yield event
        if not prov_out:
            return
        prov_result = prov_out[0]

        yield self._buf(
            request_id,
            self._tool_call(
                "effects_calculation",
                prov_result.tool_calls,
                mcp_source="OBJECTS_EFFECTS_MCP_URL",
            ),
        )
        yield self._buf(
            request_id,
            self._status("effects_calculation", "Расчёт обеспеченности завершён"),
        )
        for item in self._provision_feature_collections(prov_result.data):
            yield self._buf(request_id, item)

        service_result = self._single_service_result(prov_result.data, service_type_id)
        summary = (service_result or {}).get("summary")
        if not summary:
            error = (service_result or {}).get("error") or "нет данных"
            yield self._buf(
                request_id,
                self._chunk(
                    f"Не удалось рассчитать обеспеченность сервисом «{service_name}»: "
                    f"{error}",
                    done=True,
                ),
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        table = self.context_builder.build_provision_metrics_table(
            summary, service_name
        )
        yield self._buf(request_id, self._table(table))

        if PipelineStep.FINAL_RESPONSE not in checkpoint:
            context = self.context_builder.build_provision_context(
                summary, service_name
            )
            if plan.target_population is not None:
                context += (
                    f"\n\nРасчёт выполнен для заданной пользователем численности "
                    f"населения: {plan.target_population} человек."
                )
            async for chunk in self._generate_analysis(
                model,
                user_query,
                context,
                temperature,
                history=llm_history,
                instructions=PROVISION_ANALYSIS_INSTRUCTIONS,
                trailing_note=(
                    POPULATION_HINT if plan.target_population is None else None
                ),
            ):
                yield self._buf(request_id, chunk)
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # Mode: effects (original before/after pipeline)
    # ------------------------------------------------------------------

    async def _run_effects(
        self,
        request_id: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        token_ref: list[str],
        checkpoint: dict,
        plan: ProvisionPlan,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int,
        llm_history: list[dict],
    ) -> AsyncGenerator:
        service_name: str = plan.service_name  # type: ignore[assignment]
        target_population: int | None = plan.target_population

        # ── Step 1: GET_SERVICE_ID ────────────────────────────────────
        yield self._buf(
            request_id,
            self._status("service_lookup", f"Ищу сервис «{service_name}» в каталоге"),
        )
        svc_out: list[ProvisionStepResult] = []
        async for event in self._get_service_id_step(
            request_id,
            idu_mcp_client,
            effects_mcp_client,
            token_ref,
            checkpoint,
            service_name,
            svc_out,
        ):
            yield event
        if not svc_out:
            return
        svc_result = svc_out[0]

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

        table = self.context_builder.build_effects_pivot_table(
            effects_data, service_name
        )
        if table is not None:
            yield self._buf(request_id, self._table(table))

        # ── Step 3: FINAL_RESPONSE ────────────────────────────────────
        if PipelineStep.FINAL_RESPONSE not in checkpoint:
            context = self.context_builder.build_context(effects_data, service_name)
            async for chunk in self._generate_analysis(
                model,
                user_query,
                context,
                temperature,
                history=llm_history,
                instructions=EFFECTS_ANALYSIS_INSTRUCTIONS,
            ):
                yield self._buf(request_id, chunk)
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # Shared checkpointed steps
    # ------------------------------------------------------------------

    async def _get_service_id_step(
        self,
        request_id: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        token_ref: list[str],
        checkpoint: dict,
        service_name: str,
        out: list,
    ) -> AsyncGenerator:
        """Resolve service_type_id via IDU MCP with checkpointing; result in out."""
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
            svc_result = ProvisionStepResult(
                data=checkpoint[PipelineStep.GET_SERVICE_ID], tool_calls=[]
            )
        out.append(svc_result)

    async def _checkpointed_provision_step(
        self,
        request_id: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        token_ref: list[str],
        checkpoint: dict,
        scenario_id: int,
        services_args: dict[int, dict],
        target_population: int | None,
        out: list,
    ) -> AsyncGenerator:
        """Run CalculateServicesProvision with checkpointing; result in out."""
        if PipelineStep.CALCULATE_PROVISION not in checkpoint:
            prov_out: list[Any] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    token_ref,
                    lambda: self.tool_executor.calculate_services_provision(
                        effects_mcp_client,
                        scenario_id,
                        services_args,
                        target_population,
                    ),
                    prov_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            prov_result = prov_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.CALCULATE_PROVISION, prov_result.data
            )
        else:
            prov_result = ProvisionStepResult(
                data=checkpoint[PipelineStep.CALCULATE_PROVISION], tool_calls=[]
            )
        out.append(prov_result)

    @staticmethod
    def _single_service_result(
        services_result: dict, service_type_id: int
    ) -> dict | None:
        """Extract the per-service entry; JSON round-trips turn int keys into str."""
        services = services_result.get("services") or {}
        return services.get(str(service_type_id)) or services.get(service_type_id)

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
        token: str,
        model: str,
        user_query: str,
        scenario_id: int,
        history: list[dict] | None = None,
    ) -> tuple[ProvisionPlan, dict[str, int]]:
        """Fetch the scenario service-type catalog and classify the user query."""
        service_types = await self.urban_api_client.get_scenario_service_types(
            token, scenario_id
        )
        plan = await self.plan_builder.build_plan(
            model, user_query, list(service_types), history
        )
        return plan, service_types

    @staticmethod
    def _load_resolve_checkpoint(
        stored: dict,
    ) -> tuple[ProvisionPlan, dict[str, int]]:
        if "plan" in stored:
            return (
                ProvisionPlan.model_validate(stored["plan"]),
                stored.get("service_types") or {},
            )
        # Legacy checkpoint shape: the plan dump was stored directly
        return ProvisionPlan.model_validate(stored), {}

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
        instructions: str = EFFECTS_ANALYSIS_INSTRUCTIONS,
        trailing_note: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        messages = [
            {
                "role": "system",
                "content": f"{instructions}\n\nКонтекст расчёта:\n{context}",
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
            yield self._chunk(
                part.message.content or "",
                done=part.done and trailing_note is None,
            )
        if trailing_note is not None:
            # Deterministic service note appended after the LLM commentary
            # (e.g. how to override the population used in the calculation).
            yield self._chunk(trailing_note, done=True)
        logger.debug(f"LLM provision analysis [{model}]: {''.join(response_buffer)}")

    # ------------------------------------------------------------------
    # Chat storage helpers (mirrors RestrictionParserService)
    # ------------------------------------------------------------------

    async def _add_message_parts_to_chat(
        self,
        token: str,
        chat_id: str | None,
        parts: list[MessagePart],
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
        parts: list[MessagePart],
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
        parts: list[MessagePart],
    ) -> None:
        if not text_buffer:
            return
        parts.append(
            TextPartRequest(kind="text", payload=TextPayload(text="".join(text_buffer)))
        )
        text_buffer.clear()

    def _add_tool_calls_to_parts(
        self,
        parts: list[MessagePart],
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
    ) -> TextPartRequest | StatusPartRequest | TablePartRequest | None:
        item_type = item.get("type")
        content = item.get("content") or {}
        if item_type == "status":
            return StatusPartRequest(
                kind="status",
                payload=StatusPayload(
                    status=content.get("status", ""), text=content.get("text", "")
                ),
            )
        if item_type == "table":
            return TablePartRequest(
                kind="table",
                payload=TablePayload(
                    name=content.get("name", ""),
                    title=content.get("title", ""),
                    columns=[
                        TableColumn(**column) for column in content.get("columns", [])
                    ],
                    rows=content.get("rows", []),
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
    def _table(table_content: dict) -> dict:
        return {"type": "table", "content": table_content}

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

    @staticmethod
    def _provision_feature_collections(services_result: dict):
        """Yield feature_collection events from a CalculateServicesProvision result."""
        for service in (services_result.get("services") or {}).values():
            layers = service.get("layers") or {}
            service_name = service.get("name", "service")
            for layer_name, fc in layers.items():
                if isinstance(fc, dict) and fc.get("type") == "FeatureCollection":
                    yield {
                        "type": "feature_collection",
                        "content": {
                            "name": f"provision.{service_name}.{layer_name}",
                            "feature_collection": fc,
                        },
                    }
