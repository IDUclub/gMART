from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

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
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.model_clients.llm_base import LlmChatResponse
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.normgraph_restriction_retriever import (
    NormGraphRestrictionRetriever,
)
from src.agents.services.pipeline_state import (
    PIPELINE_TTL,
    PipelineStateStore,
    PipelineStatus,
    PipelineStep,
)
from src.agents.services.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.restriction_context import RestrictionContextBuilder
from src.agents.services.restriction_tool_executor import RestrictionToolExecutor
from src.agents.services.service_entities.restriction_plan import (
    RestrictionPlan,
    RestrictionTaskMode,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient


def _ablation_no_catalog() -> bool:
    """Whether the domain-catalog grounding ablation is enabled (evaluation only).

    Enabled when the ``ABLATION_NO_CATALOG`` env var is a truthy value
    (``1``/``true``/``yes``/``on``). Off by default in production.
    """
    return os.getenv("ABLATION_NO_CATALOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class RestrictionParserService(BaseLlmService):
    """
    Service for running restriction execution pipelines. Inherits from BaseLlmService.
    Attributes:
        host (str): Ollama host.
        chat_storage_client (ChatStorageApiClient)
        llm_client (AsyncOllamaClient): Asynchronous ollama client.
        state_store (PipelineStateStore): Redis-backed pipeline state store.
    """

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client: ChatStorageApiClient,
        urban_api_client: UrbanApiClient,
        state_store: PipelineStateStore,
    ) -> None:

        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.plan_builder = RestrictionPlanBuilder(self.llm_client)
        self.normgraph_retriever = NormGraphRestrictionRetriever(self.llm_client)
        self.tool_executor = RestrictionToolExecutor()
        self.context_builder = RestrictionContextBuilder()
        self.state_store = state_store

    async def run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        temperature: float,
        model: str | None,
        user_query: str,
        scenario_id: int,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
    ) -> AsyncGenerator:
        """Run the standard geometry-only restrictions pipeline."""

        async for item in self._run_pipeline_entry(
            mcp_client=mcp_client,
            temperature=temperature,
            model=model,
            user_query=user_query,
            scenario_id=scenario_id,
            chat_id=chat_id,
            request_id=request_id,
            persist_history=persist_history,
            normgraph_mcp_client=None,
            history_agent="restrictions",
        ):
            yield item

    async def run_compliance_pipeline(
        self,
        mcp_client: IduMcpClient,
        temperature: float,
        model: str | None,
        user_query: str,
        scenario_id: int,
        normgraph_mcp_client: NormGraphMcpClient | None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
    ) -> AsyncGenerator:
        """Run the compliance pipeline with optional normative grounding."""

        async for item in self._run_pipeline_entry(
            mcp_client=mcp_client,
            temperature=temperature,
            model=model,
            user_query=user_query,
            scenario_id=scenario_id,
            chat_id=chat_id,
            request_id=request_id,
            persist_history=persist_history,
            normgraph_mcp_client=normgraph_mcp_client,
            history_agent="compliance",
        ):
            yield item

    async def _run_pipeline_entry(
        self,
        mcp_client: IduMcpClient,
        temperature: float,
        model: str | None,
        user_query: str,
        scenario_id: int,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
        normgraph_mcp_client: NormGraphMcpClient | None = None,
        history_agent: str = "restrictions",
    ) -> AsyncGenerator:
        # Fill in the provider's model when the caller named none; keeps REST and A2A
        # on one behaviour and out of backend-specific literals.
        model = await self.resolve_model(model)
        # Mutable container so the inner pipeline can update the token on
        # refresh and the outer generator sees the latest value.
        token_ref: list[str] = [
            mcp_client.mcp_client.transport.auth.token.get_secret_value()
        ]
        text_buffer: list[str] = []
        message_parts: list[
            TextPartRequest | StatusPartRequest | ToolCallPartRequest
        ] = []

        async for item in self._run_restriction_execution_pipline(
            mcp_client=mcp_client,
            temperature=temperature,
            model=model,
            user_query=user_query,
            scenario_id=scenario_id,
            chat_id=chat_id,
            request_id=request_id,
            token_ref=token_ref,
            persist_history=persist_history,
            normgraph_mcp_client=normgraph_mcp_client,
            history_agent=history_agent,
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
            # TODO: persist full FeatureCollection/evidence snapshots for exact
            # historical reproduction. Phase one intentionally stores MCP tool
            # calls and reruns them against the current scenario state.
            yield item

        if text_buffer:
            self._flush_text_buffer_to_parts(text_buffer, message_parts)
        # Use token_ref[0]: may have been refreshed during pipeline execution.
        # A2A runs pass persist_history=False — no ChatStorage writes at all.
        if persist_history:
            self._schedule_add_message_parts_to_chat(
                token_ref[0],
                chat_id,
                message_parts,
                scenario_id=scenario_id,
            )

    async def _run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        temperature: float,
        model: str,
        user_query: str,
        scenario_id: int,
        token_ref: list[str],
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
        normgraph_mcp_client: NormGraphMcpClient | None = None,
        history_agent: str = "restrictions",
    ) -> AsyncGenerator:
        is_reconnect = request_id is not None and await self.state_store.exists(
            request_id
        )
        if is_reconnect:
            logger.info(f"Reconnect for request_id={request_id}, replaying events")
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            # Restore chat_id from persisted state so history is available
            # even if the client didn't re-send the query parameter.
            if not chat_id:
                stored_state = await self.state_store.get_state(request_id)
                if stored_state and stored_state.get("chat_id"):
                    chat_id = stored_state["chat_id"]
                    logger.info(
                        f"Restored chat_id={chat_id} from state for request_id={request_id}"
                    )
        else:
            request_id = request_id or self.state_store.new_request_id()

        original_chat_id = chat_id
        if not is_reconnect:
            yield self._buf(request_id, self._pipeline_started_event(request_id))

            # A2A runs pass persist_history=False: no chat is created and nothing
            # is written to ChatStorage (history stays read-only).
            if not chat_id and persist_history:
                logger.info("No chat id provided, creating a new chat.")
                chat_result: list[tuple[str, str]] = []
                try:
                    async for event in self._retryable_step(
                        request_id,
                        mcp_client,
                        token_ref,
                        lambda: self.create_chat(
                            token_ref[0],
                            model,
                            user_query,
                            additional_instructions="""Первый запрос пользователя был отправлен к сервису
                                создания слоёв с ограничениями ихз запроса пользователя.
                            """,
                            scenario_id=scenario_id,
                            agent_id=history_agent,
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

        logger.info(
            f"Pipeline request_id={request_id} chat_id={chat_id} query={user_query!r}"
        )

        llm_history: list[dict] = []
        if original_chat_id:
            try:
                chat_info = await self.get_chat_messages(token_ref[0], original_chat_id)
                llm_history = self.build_llm_history(
                    chat_info.messages, current_user_query=user_query
                )
                logger.info(f"Loaded {len(llm_history)} messages from chat history")
            except Exception as exc:
                logger.warning(
                    f"Failed to fetch chat history, proceeding without it: {exc}"
                )

        # A follow-up question in an existing chat is persisted here — create_chat
        # stores only the first one. Runs after the history fetch so the current
        # question doesn't also enter the LLM context from storage, and is skipped
        # on reconnect (the original run already stored it). Chat storage failures
        # must not break the stream.
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
                logger.warning(f"Failed to persist user question: {exc}")

        checkpoint = await self.state_store.get_checkpoint(request_id)

        normgraph_restrictions: list[dict[str, Any]] = []
        if normgraph_mcp_client is not None:
            yield self._buf(
                request_id,
                self._status(
                    "norm_retrieval",
                    "Проверяю применимые канонические ограничения в NormGraph",
                ),
            )
            if PipelineStep.NORMGRAPH not in checkpoint:
                retrieval = await self.normgraph_retriever.retrieve(
                    normgraph_mcp_client,
                    model,
                    user_query,
                    history=llm_history,
                )
                normgraph_restrictions = retrieval.restrictions
                checkpoint_data = {
                    "restrictions": retrieval.restrictions,
                    "unsupported_count": retrieval.unsupported_count,
                    "tool_call": retrieval.tool_call,
                }
                await self.state_store.save_checkpoint(
                    request_id, PipelineStep.NORMGRAPH, checkpoint_data
                )
                yield self._buf(
                    request_id,
                    self._tool_call(
                        "norm_retrieval",
                        [retrieval.tool_call],
                        mcp_source="NORM_GRAPH_MCP_URL",
                    ),
                )
            else:
                normgraph_restrictions = checkpoint[PipelineStep.NORMGRAPH].get(
                    "restrictions", []
                )

        yield self._buf(
            request_id,
            self._status(
                "data_retrievement", "Получаю каталоги сервисов и физических объектов"
            ),
        )
        if PipelineStep.PLAN not in checkpoint:
            plan_out: list[RestrictionPlan] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    mcp_client,
                    token_ref,
                    lambda: self._build_plan(
                        mcp_client,
                        model,
                        user_query,
                        scenario_id,
                        llm_history,
                        normgraph_restrictions,
                    ),
                    plan_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            plan = plan_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.PLAN, plan.model_dump(mode="json")
            )
        else:
            plan = RestrictionPlan.model_validate(checkpoint[PipelineStep.PLAN])

        if plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION:
            yield self._buf(
                request_id,
                self._status(
                    "context_preparation", "Нужно уточнение параметров запроса."
                ),
            )
            yield self._buf(
                request_id,
                self._chunk(
                    plan.clarification_question or "Уточните параметры запроса.",
                    done=True,
                ),
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        if PipelineStep.PLAN_EXPLANATION not in checkpoint:
            yield self._buf(
                request_id,
                self._status(
                    "plan_explanation", "Объясняю, почему выбраны эти параметры"
                ),
            )
            async for chunk in self.generate_plan_explanation(
                model, user_query, plan, temperature, history=llm_history
            ):
                yield self._buf(request_id, chunk)
            yield self._buf(request_id, self._chunk("\n\n", done=False))
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.PLAN_EXPLANATION, True
            )

        yield self._buf(
            request_id,
            self._status(
                "data_retrievement", "Получаю необходимые слои по утверждённому плану"
            ),
        )
        if PipelineStep.LAYERS not in checkpoint:
            layers_out: list[Any] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    mcp_client,
                    token_ref,
                    lambda: self.tool_executor.retrieve_layers_for_plan(
                        mcp_client, plan, scenario_id
                    ),
                    layers_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            layers_result = layers_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.LAYERS, layers_result.tool_result
            )
        else:
            from src.agents.services.service_entities.restriction_entities import (
                GeometryToolCallResult,
            )

            layers_result = GeometryToolCallResult(
                tool_result=checkpoint[PipelineStep.LAYERS],
                tool_calls=[],
                messages=[],
            )

        yield self._buf(
            request_id,
            self._tool_call(
                "data_retrievement", layers_result.tool_calls, mcp_source="IDU_MCP_URL"
            ),
        )
        for item in self._feature_collections(layers_result.tool_result):
            yield self._buf(request_id, item)
        layers = layers_result.tool_result

        yield self._buf(
            request_id,
            self._status(
                "buffer_creation", "Начинаю построение буферов зон с ограничениями"
            ),
        )
        if PipelineStep.BUFFERS not in checkpoint:
            buffers_out: list[Any] = []
            try:
                async for event in self._retryable_step(
                    request_id,
                    mcp_client,
                    token_ref,
                    lambda: self.tool_executor.run_buffer_plan(
                        mcp_client, plan, layers
                    ),
                    buffers_out,
                ):
                    yield self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            buffers_result = buffers_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.BUFFERS, buffers_result.tool_result
            )
        else:
            from src.agents.services.service_entities.restriction_entities import (
                GeometryToolCallResult,
            )

            buffers_result = GeometryToolCallResult(
                tool_result=checkpoint[PipelineStep.BUFFERS],
                tool_calls=[],
                messages=[],
            )

        yield self._buf(
            request_id,
            self._tool_call(
                "buffer_creation", buffers_result.tool_calls, mcp_source="IDU_MCP_URL"
            ),
        )
        yield self._buf(
            request_id,
            self._status(
                "buffer_creation", "Построил необходимые буферы с ограничениями."
            ),
        )
        for item in self._feature_collections(buffers_result.tool_result):
            yield self._buf(request_id, item)

        if plan.mode == RestrictionTaskMode.BUFFERS_ONLY:
            context = await self.context_builder.generate_buffers_context(
                buffers_result.tool_result
            )
        else:
            yield self._buf(
                request_id,
                self._status(
                    "restriction_formation",
                    "Начинаю извлечение нормативных ограничений.",
                ),
            )
            if PipelineStep.RESTRICTIONS not in checkpoint:
                restr_out: list[Any] = []
                try:
                    async for event in self._retryable_step(
                        request_id,
                        mcp_client,
                        token_ref,
                        lambda: self.tool_executor.run_restriction_plan(
                            mcp_client, plan, layers, buffers_result.tool_result
                        ),
                        restr_out,
                    ):
                        yield self._buf(request_id, event)
                except PipelineSuspendedError:
                    return
                restriction_result = restr_out[0]
                await self.state_store.save_checkpoint(
                    request_id,
                    PipelineStep.RESTRICTIONS,
                    restriction_result.tool_result,
                )
            else:
                from src.agents.services.service_entities.restriction_entities import (
                    GeometryToolCallResult,
                )

                restriction_result = GeometryToolCallResult(
                    tool_result=checkpoint[PipelineStep.RESTRICTIONS],
                    tool_calls=[],
                    messages=[],
                )

            yield self._buf(
                request_id,
                self._tool_call(
                    "restriction_formation",
                    restriction_result.tool_calls,
                    mcp_source="IDU_MCP_URL",
                ),
            )
            yield self._buf(
                request_id,
                self._status(
                    "restriction_formation",
                    "Извлечение нормативных ограничений завершено.",
                ),
            )
            for item in self._feature_collections(restriction_result.tool_result):
                yield self._buf(request_id, item)
            context = await self.context_builder.generate_restrictions_context(
                restriction_result.tool_result["generators"],
                restriction_result.tool_result["objects"],
            )

        if PipelineStep.FINAL_RESPONSE not in checkpoint:
            async for chunk in self.generate_final_response(
                model, user_query, context, temperature, history=llm_history
            ):
                yield self._buf(request_id, chunk)
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    async def _retryable_step(
        self,
        request_id: str,
        mcp_client: IduMcpClient,
        token_ref: list[str],
        step_fn: Callable,
        result: list,
    ) -> AsyncGenerator[dict, None]:
        """Execute a step without replacing the pipeline's M2M credentials.

        ``mcp_client`` and ``token_ref`` remain in the signature for call-site
        compatibility.  Authentication refresh belongs to ``ServiceTokenAuth``;
        a downstream 401 must surface as a service-auth failure instead of asking
        the browser for a new user JWT.
        """
        del request_id, mcp_client, token_ref
        result.append(await step_fn())
        if False:  # pragma: no cover
            yield {}

    def _buf(self, request_id: str, event: dict) -> dict:
        """Fire-and-forget: buffer the event to Redis and return it for yielding."""
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    async def generate_plan_explanation(
        self,
        model: str,
        user_query: str,
        plan: RestrictionPlan,
        temperature: float,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[dict[str, str | dict[str, str | None | bool]], None]:
        messages = [
            {
                "role": "system",
                "content": f"""Коротко и дружелюбно объясни пользователю, почему для его запроса выбраны такие параметры.
                Пиши обычным человеческим языком, без технических терминов.
                Не упоминай JSON, модель, инструмент, пайплайн, схему, поля или внутренние названия.
                Не спорь с пользователем и не перегружай деталями.
                Объясни:
                - что выбрано как источник построения зон;
                - какой радиус используется и откуда он взят;
                - будут ли строиться только буферы или также ограничения для других объектов;
                - если есть целевые объекты, почему они выбраны.

                На этом этапе расчёт ещё не выполнен. Не сообщай количество найденных
                объектов, не перечисляй адреса и не приводи «примерные» результаты.
                Объясняй только выбранные параметры будущей проверки.

                Данные для объяснения:
                {self._plan_summary(plan)}
                """,
            },
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        response_buffer: list[str] = []
        async for part in await self.llm_client.chat(
            model,
            messages,
            think=False,
            options={"temperature": min(temperature, 0.4)},
            stream=True,
        ):
            part: LlmChatResponse
            if part.message.content:
                response_buffer.append(part.message.content)
                yield self._chunk(part.message.content, done=False)
        logger.debug(f"LLM plan explanation [{model}]: {''.join(response_buffer)}")

    async def generate_final_response(
        self,
        model: str,
        user_query: str,
        context: str,
        temperature: float,
        history: list[dict] | None = None,
    ) -> AsyncGenerator[dict[str, str | dict[str, str | None | bool]], None]:
        messages = [
            {
                "role": "system",
                "content": f"""Дай комментарий к запросу пользователя на основе контекста статистики сгенерированных слоёв.
                Ответ давай только в виде обычного текста. Внимательно анализируй предоставленную в контексте информацию.
                Сообщи общее число затронутых объектов. Для каждого объекта из
                affected_objects назови его понятное имя, составной object_id, применённое
                ограничение и причину попадания. Если details_truncated=true, явно скажи,
                что полный перечень находится в возвращённом GeoJSON. Если объектов нет,
                сообщи об этом прямо. Не показывай программный код.
                В качестве нормативных отсылок используй название документа, номер пункта
                и restriction_id только тогда, когда они есть в evidence/provenance.

                Контекст для ответа:

                {context}
                """,
            },
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        response_buffer: list[str] = []
        async for part in await self.llm_client.chat(
            model,
            messages,
            think=False,
            options={"temperature": temperature},
            stream=True,
        ):
            part: LlmChatResponse
            if part.message.content:
                response_buffer.append(part.message.content)
                yield self._chunk(part.message.content, done=False)
        if not response_buffer:
            fallback = self._fallback_final_response(context)
            response_buffer.append(fallback)
            yield self._chunk(fallback, done=False)
        yield self._chunk("", done=True)
        logger.debug(f"LLM final response [{model}]: {''.join(response_buffer)}")

    @staticmethod
    def _fallback_final_response(context: str) -> str:
        """Return a useful user-facing result if Ollama emits no content."""

        match = re.search(r'"affected_count"\s*:\s*(\d+)', context)
        if match:
            affected_count = int(match.group(1))
            if affected_count == 0:
                return (
                    "Проверка завершена: объектов, попавших под заданные ограничения, "
                    "не найдено. Геометрии зон и источников возвращены вместе с результатом."
                )
            return (
                f"Проверка завершена: под заданные ограничения попали "
                f"{affected_count} объектов. Полный перечень объектов возвращён в GeoJSON; "
                "для каждого объекта там указаны понятное имя, составной идентификатор, "
                "применённое ограничение и причина геометрического пересечения."
            )
        return (
            "Проверка завершена. Полный результат возвращён в GeoJSON вместе с объектами "
            "и атрибутами, объясняющими причины попадания под ограничения."
        )

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
            logger.exception(f"Failed to upload restriction response message: {exc}")

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
            self._tool_call_to_chat_storage_call(step, tool_call)
            for step, tool_call in enumerate(tool_calls, start=1)
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
            if not text:
                return None
            return TextPartRequest(kind="text", payload=TextPayload(text=text))
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

    async def _build_plan(
        self,
        mcp_client: IduMcpClient,
        model: str,
        user_query: str,
        scenario_id: int,
        history: list[dict] | None = None,
        normgraph_restrictions: list[dict[str, Any]] | None = None,
    ) -> RestrictionPlan:
        # Ablation switch (evaluation only): when ABLATION_NO_CATALOG is set the
        # plan is built WITHOUT the domain-catalog grounding, so the effect of
        # catalog grounding on plan validity / entity correctness can be measured.
        # Env-gated (not per-request) so an ablation arm runs as a separate pass
        # and the public request contract is untouched.
        if _ablation_no_catalog():
            services_catalog: list[str] = []
            physical_objects_catalog: list[str] = []
        else:
            services_catalog, physical_objects_catalog = (
                await self.plan_builder.get_entity_catalogs(mcp_client, scenario_id)
            )
        return await self.plan_builder.build_plan(
            model,
            user_query,
            scenario_id,
            services_catalog,
            physical_objects_catalog,
            history=history,
            normgraph_restrictions=normgraph_restrictions,
        )

    @staticmethod
    def _pipeline_started_event(request_id: str) -> dict:
        return {
            "type": "pipeline_started",
            "content": {"request_id": request_id},
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

    # Backend result dicts key the restriction layers by internal English names;
    # translate them to human-readable titles so they are not shown to the user
    # verbatim (the effect layers already arrive under catalog names).
    _RESERVED_LAYER_NAMES = {
        "objects": "Объекты в зоне ограничений",
        "generators": "Источники ограничений",
    }

    @classmethod
    def _feature_collections(cls, layers: dict[str, dict]):
        for name, feature_collection in layers.items():
            display = cls._RESERVED_LAYER_NAMES.get(str(name), name)
            yield {
                "type": "feature_collection",
                "content": {
                    "name": display,
                    "feature_collection": feature_collection,
                },
            }

    @staticmethod
    def _plan_summary(plan: RestrictionPlan) -> dict:
        return {
            "mode": plan.mode.value,
            "sources": [entity.name for entity in plan.source_entities],
            "targets": [entity.name for entity in plan.target_entities],
            "buffers": [
                {
                    "source": rule.source_name,
                    "distance_m": rule.buffer_size,
                    "title": rule.title,
                    "origin": rule.origin,
                    "restriction_id": rule.restriction_id,
                    "provenance": (
                        rule.provenance.model_dump(mode="json")
                        if rule.provenance
                        else None
                    ),
                }
                for rule in plan.buffer_rules
            ],
            "restrictions": [
                {
                    "source": rule.source_name,
                    "targets": rule.target_names,
                    "title": rule.title,
                    "description": rule.description,
                    "origin": rule.origin,
                    "restriction_id": rule.restriction_id,
                    "provenance": (
                        rule.provenance.model_dump(mode="json")
                        if rule.provenance
                        else None
                    ),
                }
                for rule in plan.restriction_rules
            ],
            "selection_reasons": [
                {"step": reason.step, "reason": reason.reason}
                for reason in plan.selection_reasons
            ],
        }
