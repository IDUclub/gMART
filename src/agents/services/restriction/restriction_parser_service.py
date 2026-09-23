from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncGenerator, Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    FilePartRequest,
    StatusPartRequest,
    StatusPayload,
    StructuredPartRequest,
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.common.files.temporary_file_store import TemporaryFileStore
from src.agents.model_clients.llm_base import LlmChatResponse
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.compilance.compliance_dedup import group_checks
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_metrics import COMPLIANCE_METRICS
from src.agents.services.compilance.compliance_report import (
    REPORT_MIME_TYPE,
    REPORT_SLOT,
    REPORT_TITLE,
    build_compliance_report,
    report_filename,
)
from src.agents.services.compilance.compliance_result_harness import (
    ComplianceResultHarness,
    PreparedComplianceFollowUp,
)
from src.agents.services.compilance.compliance_sources import (
    source_reference,
    source_references,
)
from src.agents.services.layer_attributes import compact_layer, compact_layer_event
from src.agents.services.normgraph.normgraph_restriction_retriever import (
    NormGraphRestrictionRetriever,
)
from src.agents.services.pipeline_state import (
    PipelineStateStore,
    PipelineStatus,
    PipelineStep,
)
from src.agents.services.restriction.restriction_catalog import (
    RestrictionPlanBuilder,
    normalize_name,
)
from src.agents.services.restriction.restriction_context import (
    RestrictionContextBuilder,
)
from src.agents.services.restriction.restriction_tool_executor import (
    RestrictionToolExecutor,
)
from src.agents.services.service_entities.compliance import (
    ComplianceResult,
    ComplianceSummary,
    VerificationCoverage,
)
from src.agents.services.service_entities.restriction_plan import (
    RestrictionPlan,
    RestrictionTaskMode,
)
from src.common.service_auth import user_id_from_jwt

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
        file_store: TemporaryFileStore | None = None,
    ) -> None:

        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.plan_builder = RestrictionPlanBuilder(self.llm_client)
        self.normgraph_retriever = NormGraphRestrictionRetriever(self.llm_client)
        self.tool_executor = RestrictionToolExecutor()
        self.compliance_executor = ComplianceTemplateExecutor()
        self.compliance_result_harness = ComplianceResultHarness()
        self.context_builder = RestrictionContextBuilder()
        self.state_store = state_store
        self.file_store = file_store

    async def run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        token: str,
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
            token=token,
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
        token: str,
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
            token=token,
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
        token: str,
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
        # User context is passed explicitly. The MCP transport contains only
        # dynamic M2M authentication and must never be inspected for a user JWT.
        token_ref = [token]
        text_buffer: list[str] = []
        message_parts: list[
            TextPartRequest
            | StatusPartRequest
            | ToolCallPartRequest
            | StructuredPartRequest
            | FilePartRequest
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
            part = self._pipeline_item_to_chat_part(
                item, text_only=history_agent == "compliance"
            )
            if part is not None:
                message_parts.append(part)
            # Compliance details remain in the live stream and pipeline journal.
            # Chat history keeps only tool calls and text to avoid oversized BSON.
            yield item

        if text_buffer:
            self._flush_text_buffer_to_parts(text_buffer, message_parts)
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
                yield compact_layer_event(event, history_agent)
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
            yield await self._buf(request_id, self._pipeline_started_event(request_id))

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
                        yield await self._buf(request_id, event)
                except PipelineSuspendedError:
                    return
                chat_id, title = chat_result[0]
                yield await self._buf(
                    request_id, self._chat_created_event(chat_id, title)
                )

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
        chat_messages: list[dict[str, Any]] = []
        if original_chat_id:
            try:
                chat_info = await self.get_chat_messages(token_ref[0], original_chat_id)
                chat_messages = chat_info.messages
                llm_history = self.build_llm_history(
                    chat_messages, current_user_query=user_query
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

        if history_agent == "compliance" and not is_reconnect:
            prepared_follow_up = self.compliance_result_harness.prepare_follow_up(
                user_query, chat_messages, llm_history
            )
            if prepared_follow_up is not None:
                yield await self._buf(
                    request_id,
                    self._status(
                        "compliance_result_analysis",
                        "Анализирую результат последней проверки",
                    ),
                )
                async for chunk in self.generate_compliance_follow_up(
                    model,
                    prepared_follow_up,
                    temperature,
                ):
                    yield await self._buf(request_id, chunk)
                await self.state_store.set_status(request_id, PipelineStatus.DONE)
                return

        checkpoint = await self.state_store.get_checkpoint(request_id)

        normgraph_restrictions: list[dict[str, Any]] = []
        skipped_without_plan = 0
        if normgraph_mcp_client is not None:
            yield await self._buf(
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
                    require_check_plan=history_agent == "compliance",
                    retrieve_all=history_agent == "compliance",
                )
                normgraph_restrictions = retrieval.restrictions
                skipped_without_plan = retrieval.unsupported_count
                checkpoint_data = {
                    "restrictions": retrieval.restrictions,
                    "unsupported_count": retrieval.unsupported_count,
                    "tool_call": retrieval.tool_call,
                }
                await self.state_store.save_checkpoint(
                    request_id, PipelineStep.NORMGRAPH, checkpoint_data
                )
                yield await self._buf(
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

                skipped_without_plan = checkpoint[PipelineStep.NORMGRAPH].get(
                    "unsupported_count", 0
                )

        # Compliance consumes persisted CheckPlans only. Mentioning a threshold
        # must never route the entire normative corpus into the LLM plan builder.
        if history_agent == "compliance":
            if normgraph_mcp_client is None:
                yield await self._buf(
                    request_id,
                    self._chunk(
                        "Проверка соответствия не выполнена: NormGraph не подключён.",
                        done=True,
                    ),
                )
                await self.state_store.set_status(request_id, PipelineStatus.DONE)
                return
            async for event in self._run_executable_compliance(
                mcp_client=mcp_client,
                request_id=request_id,
                scenario_id=scenario_id,
                restrictions=normgraph_restrictions,
                checkpoint=checkpoint,
                skipped_without_plan=skipped_without_plan,
                owner=self._report_owner(token_ref[0]),
            ):
                yield event
            return

        yield await self._buf(
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
                    yield await self._buf(request_id, event)
            except PipelineSuspendedError:
                return
            plan = plan_out[0]
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.PLAN, plan.model_dump(mode="json")
            )
        else:
            plan = RestrictionPlan.model_validate(checkpoint[PipelineStep.PLAN])

        if plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION:
            yield await self._buf(
                request_id,
                self._status(
                    "context_preparation", "Нужно уточнение параметров запроса."
                ),
            )
            yield await self._buf(
                request_id,
                {
                    "type": "clarification",
                    "content": {
                        "question": plan.clarification_question
                        or "Уточните параметры запроса."
                    },
                },
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        if PipelineStep.PLAN_EXPLANATION not in checkpoint:
            yield await self._buf(
                request_id,
                self._status(
                    "plan_explanation", "Объясняю, почему выбраны эти параметры"
                ),
            )
            async for chunk in self.generate_plan_explanation(
                model, user_query, plan, temperature, history=llm_history
            ):
                yield await self._buf(request_id, chunk)
            yield await self._buf(request_id, self._chunk("\n\n", done=False))
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.PLAN_EXPLANATION, True
            )

        yield await self._buf(
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
                    yield await self._buf(request_id, event)
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

        yield await self._buf(
            request_id,
            self._tool_call(
                "data_retrievement", layers_result.tool_calls, mcp_source="IDU_MCP_URL"
            ),
        )
        layers = layers_result.tool_result
        visible_layers = layers
        if plan.mode == RestrictionTaskMode.RESTRICTIONS:
            source_names = {
                normalize_name(entity.name) for entity in plan.source_entities
            }
            # Targets are published only after checking their intersections.
            visible_layers = {
                name: layer
                for name, layer in layers.items()
                if normalize_name(name) in source_names
            }
        for item in self._feature_collections(visible_layers):
            yield await self._buf(request_id, item)

        yield await self._buf(
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
                    yield await self._buf(request_id, event)
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

        yield await self._buf(
            request_id,
            self._tool_call(
                "buffer_creation", buffers_result.tool_calls, mcp_source="IDU_MCP_URL"
            ),
        )
        yield await self._buf(
            request_id,
            self._status(
                "buffer_creation", "Построил необходимые буферы с ограничениями."
            ),
        )
        if plan.mode == RestrictionTaskMode.BUFFERS_ONLY:
            # Restriction checks publish these geometries in the final generators layer.
            for item in self._feature_collections(buffers_result.tool_result):
                yield await self._buf(request_id, item)
            # A buffer count is not a count of affected objects. Report the
            # actual returned layers directly, including empty collections.
            counts = [
                f"«{name}»: {len(layer.get('features', []))}"
                for name, layer in buffers_result.tool_result.items()
            ]
            yield await self._buf(
                request_id,
                self._chunk(
                    "Построены буферы. Количество геометрий по слоям: "
                    + "; ".join(counts)
                    + ". Пересечения с целевыми объектами в этом расчёте не проверялись.",
                    done=True,
                ),
            )
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return
        else:
            yield await self._buf(
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
                        yield await self._buf(request_id, event)
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

            yield await self._buf(
                request_id,
                self._tool_call(
                    "restriction_formation",
                    restriction_result.tool_calls,
                    mcp_source="IDU_MCP_URL",
                ),
            )
            yield await self._buf(
                request_id,
                self._status(
                    "restriction_formation",
                    "Извлечение нормативных ограничений завершено.",
                ),
            )
            for item in self._feature_collections(restriction_result.tool_result):
                yield await self._buf(request_id, item)
            context = await self.context_builder.generate_restrictions_context(
                restriction_result.tool_result["generators"],
                restriction_result.tool_result["objects"],
            )

        if PipelineStep.FINAL_RESPONSE not in checkpoint:
            async for chunk in self.generate_final_response(
                model, user_query, context, temperature, history=llm_history
            ):
                yield await self._buf(request_id, chunk)
            await self.state_store.save_checkpoint(
                request_id, PipelineStep.FINAL_RESPONSE, True
            )

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    async def _run_executable_compliance(
        self,
        *,
        mcp_client: IduMcpClient,
        request_id: str,
        scenario_id: int,
        restrictions: list[dict[str, Any]],
        checkpoint: dict[str, Any],
        skipped_without_plan: int = 0,
        owner: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Execute each NormGraph CheckPlan independently and emit replayable results."""

        if PipelineStep.VERDICT_AGGREGATION in checkpoint:
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        yield await self._buf(
            request_id,
            self._status(
                "check_plan_validation",
                "Проверяю версии и параметры нормативных планов",
            ),
        )
        plans: list[dict[str, Any]] = []
        for hit in restrictions:
            # Apply the same gate to old checkpoints restored on reconnect.
            if not NormGraphRestrictionRetriever.has_executable_plan(hit):
                skipped_without_plan += 1
                continue
            raw_plan = hit["check_plan"]
            source = dict(raw_plan["source"])
            provenance = hit.get("provenance") or {}
            # Older stored plans may lack source labels still present in the hit.
            for field, value, limit in (
                ("document_name", provenance.get("name"), 300),
                ("clause_number", provenance.get("numbering"), 100),
                ("extraction_text", hit.get("extraction_text"), 8000),
            ):
                if not (source.get(field) or "").strip() and isinstance(value, str):
                    source[field] = value.strip()[:limit]
            raw_plan = {**raw_plan, "source": source}
            plans.append(raw_plan)
        groups = await group_checks(
            plans,
            mcp_client,
            scenario_id,
            resolver=getattr(
                getattr(self, "compliance_executor", None), "catalog_resolver", None
            ),
        )
        duplicates = len(plans) - len(groups)
        plans = [group.plan for group in groups]
        sources_by_id = {
            group.plan["source"]["restriction_id"]: group.sources for group in groups
        }
        for raw_plan in plans:
            yield await self._buf(
                request_id,
                {
                    "type": "check_plan",
                    "content": {
                        "restriction_id": raw_plan["source"]["restriction_id"],
                        "plan": raw_plan,
                        "equivalent_sources": sources_by_id[
                            raw_plan["source"]["restriction_id"]
                        ],
                    },
                },
            )
        if skipped_without_plan:
            yield await self._buf(
                request_id,
                self._status(
                    "check_plan_validation",
                    f"Пропущено норм без исполнимого плана: {skipped_without_plan}. "
                    f"К проверке принято: {len(plans)}.",
                ),
            )
        await self.state_store.save_checkpoint(
            request_id, PipelineStep.CHECK_PLAN_VALIDATION, plans
        )

        yield await self._buf(
            request_id,
            self._status(
                "requirements_resolution",
                "Проверяю наличие слоёв, атрибутов и полноту данных",
            ),
        )
        results: list[ComplianceResult] = []
        layer_name_counts: dict[str, int] = {}
        resolution_events: list[dict[str, Any]] = []
        yield await self._buf(
            request_id,
            {
                "type": "compliance_progress",
                "content": self._compliance_progress(results, len(plans)),
            },
        )
        for index, raw_plan in enumerate(plans, start=1):
            execution_calls: list[dict[str, Any]] = []
            timings_ms: dict[str, float] = {}
            try:
                execution = await self.compliance_executor.execute(
                    mcp_client, raw_plan, scenario_id
                )
                result = execution.result
                execution_calls = execution.tool_calls
                timings_ms = execution.timings_ms
            except Exception as exc:  # one norm must not erase the others
                logger.bind(
                    restriction_id=(raw_plan.get("source") or {}).get("restriction_id"),
                    template=raw_plan.get("template"),
                ).exception("Compliance template failed")
                error_text = f"{type(exc).__name__} {exc}".lower()
                COMPLIANCE_METRICS.observe_downstream_error(
                    "urban_api" if "urban" in error_text else "idu_mcp"
                )
                result = self._failed_compliance_result(raw_plan, str(exc))
            planner_status = str(raw_plan.get("planner_status") or "unsupported")
            COMPLIANCE_METRICS.observe(
                result,
                planner_status=planner_status,
                timings_ms=timings_ms,
            )
            logger.bind(
                request_id=request_id,
                restriction_id=result.restriction_id,
                template=result.template,
                template_version=result.template_version,
                planner_status=planner_status,
                verification_status=result.verification_status,
                compliance_status=result.compliance_status,
                timings_ms=timings_ms,
                applicable_objects=result.coverage.applicable_objects,
                checked_objects=result.coverage.checked_objects,
                unchecked_objects=result.coverage.unchecked_objects,
                fill_rate=result.coverage.fill_rate,
                violated_objects=result.summary.violated_objects,
            ).info("Compliance norm completed")
            result.source["equivalent_sources"] = sources_by_id[result.restriction_id]
            results.append(result)
            resolution_content = {
                "restriction_id": result.restriction_id,
                "effective_requirements": result.effective_requirements.model_dump(
                    mode="json"
                ),
                "resolved_requirements": [
                    item.model_dump(mode="json")
                    for item in result.resolved_requirements
                ],
                "missing_requirements": result.missing_requirements,
            }
            resolution_events.append(resolution_content)
            yield await self._buf(
                request_id,
                {"type": "requirement_resolution", "content": resolution_content},
            )
            yield await self._buf(
                request_id,
                self._status(
                    "template_execution",
                    f"Выполняю {result.template}@v{result.template_version}",
                ),
            )
            if execution_calls:
                yield await self._buf(
                    request_id,
                    self._tool_call(
                        "template_execution", execution_calls, "IDU_MCP_URL"
                    ),
                )
            yield await self._buf(
                request_id,
                {
                    "type": "compliance_result",
                    "content": self._compliance_result_payload(result),
                },
            )
            progress = self._compliance_progress(results, len(plans))
            yield await self._buf(
                request_id,
                {"type": "compliance_progress", "content": progress},
            )
            yield await self._buf(
                request_id,
                self._status(
                    "template_execution",
                    f"Проверено {index} из {len(plans)} норм",
                ),
            )
            # Map geometry has one transport: non-empty violation layer events.
            # Successful and non-executed checks must not open empty UI layers.
            if (
                result.compliance_status == "violated"
                and result.summary.violated_objects > 0
                and (result.violated_features or {}).get("features")
            ):
                name = f"Нарушение нормы — {source_reference(result.source)}"
                layer_name_counts[name] = layer_name_counts.get(name, 0) + 1
                if layer_name_counts[name] > 1:
                    name += f" ({layer_name_counts[name]})"
                for item in self._feature_collections({name: result.violated_features}):
                    yield await self._buf(request_id, item)

        await self.state_store.save_checkpoint(
            request_id, PipelineStep.REQUIREMENTS_RESOLUTION, resolution_events
        )
        await self.state_store.save_checkpoint(
            request_id,
            PipelineStep.TEMPLATE_EXECUTION,
            [
                RestrictionParserService._compliance_result_payload(item)
                for item in results
            ],
        )
        yield await self._buf(
            request_id,
            self._status("verdict_aggregation", "Собираю итог по всем нормам"),
        )
        summary = self._compliance_summary(request_id, results)
        summary["duplicate_checks"] = duplicates
        summary["equivalent_sources"] = {
            rid: sources for rid, sources in sources_by_id.items() if len(sources) > 1
        }
        await self.state_store.save_checkpoint(
            request_id, PipelineStep.VERDICT_AGGREGATION, summary
        )
        yield await self._buf(
            request_id, {"type": "compliance_summary", "content": summary}
        )
        summary_text = self._compliance_summary_text(summary)
        if duplicates:
            summary_text += f" Повторных проверок объединено: {duplicates}."
            for sources in summary["equivalent_sources"].values():
                labels = list(dict.fromkeys(source_reference(s) for s in sources))
                summary_text += " Эквивалентные нормы: " + "; ".join(labels) + "."
        if skipped_without_plan:
            summary_text += (
                f" Пропущено норм без исполнимого плана: {skipped_without_plan}."
            )
        yield await self._buf(
            request_id,
            self._chunk(summary_text, done=True),
        )
        # The report link closes the stream, after the final answer text.
        report_event = self._compliance_report_event(
            {**summary, "skipped_without_plan": skipped_without_plan},
            scenario_id=scenario_id,
            owner=owner,
        )
        if report_event is not None:
            yield await self._buf(request_id, report_event)
        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    def _compliance_report_event(
        self,
        summary: dict[str, Any],
        *,
        scenario_id: int,
        owner: str | None,
    ) -> dict | None:
        """Store the Markdown report and describe it as a ``file`` event."""

        file_store: TemporaryFileStore | None = getattr(self, "file_store", None)
        if file_store is None or not owner:
            return None
        report = build_compliance_report(summary)
        if report is None:
            return None
        filename = report_filename(scenario_id, datetime.now(timezone.utc))
        try:
            stored = file_store.save(
                REPORT_SLOT,
                report.encode("utf-8"),
                owner=owner,
                filename=filename,
                mime_type=REPORT_MIME_TYPE,
            )
        except OSError:
            # The verdicts are already delivered; a missing file must not fail them.
            logger.exception("Could not store the compliance report")
            return None
        return {
            "type": "file",
            "content": {
                "name": REPORT_SLOT,
                "title": REPORT_TITLE,
                "role": "result",
                "url": stored.url,
                "download_url": stored.download_url,
                "filename": filename,
                "mime_type": REPORT_MIME_TYPE,
                "source_service": "gmart",
            },
        }

    @staticmethod
    def _report_owner(token: str) -> str | None:
        try:
            return user_id_from_jwt(token)
        except ValueError:
            logger.warning("Compliance report is skipped: token has no user id")
            return None

    @staticmethod
    def _failed_compliance_result(
        raw_plan: dict[str, Any], reason: str
    ) -> ComplianceResult:
        source = raw_plan.get("source") or {}
        return ComplianceResult(
            restriction_id=str(source.get("restriction_id") or "unknown"),
            template=str(raw_plan.get("template") or "unknown"),
            template_version=int(raw_plan.get("template_version") or 1),
            verification_status="unverifiable",
            compliance_status="unknown",
            coverage=VerificationCoverage(
                applicable_objects=0,
                checked_objects=0,
                unchecked_objects=0,
                fill_rate=0,
            ),
            summary=ComplianceSummary(violated_objects=0, passed_objects=0),
            missing_requirements=["template_execution_failed"],
            warnings=[reason[:1000]],
            source=source,
        )

    @staticmethod
    def _compliance_summary(
        request_id: str, results: list[ComplianceResult]
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "total_norms": len(results),
            "violated_norms": sum(
                item.compliance_status == "violated" for item in results
            ),
            "passed_norms": sum(item.compliance_status == "passed" for item in results),
            "unverifiable_norms": sum(
                item.verification_status == "unverifiable" for item in results
            ),
            "unsupported_norms": sum(
                item.verification_status == "unsupported" for item in results
            ),
            "not_applicable_norms": sum(
                item.verification_status == "not_applicable" for item in results
            ),
            "partial_norms": sum(
                item.verification_status == "partial" for item in results
            ),
            "results": [
                RestrictionParserService._compliance_result_payload(item)
                for item in results
            ],
        }

    @staticmethod
    def _compliance_progress(
        results: list[ComplianceResult], total_norms: int
    ) -> dict[str, int]:
        completed = len(results)
        return {
            "total_norms": total_norms,
            "completed_norms": completed,
            "pending_norms": max(total_norms - completed, 0),
            "passed_norms": sum(item.compliance_status == "passed" for item in results),
            "violated_norms": sum(
                item.compliance_status == "violated" for item in results
            ),
            "unverifiable_norms": sum(
                item.verification_status == "unverifiable" for item in results
            ),
            "unsupported_norms": sum(
                item.verification_status == "unsupported" for item in results
            ),
        }

    @staticmethod
    def _compliance_result_payload(result: ComplianceResult) -> dict[str, Any]:
        """Keep result metadata; geometry is emitted only as feature_collection."""
        return result.model_dump(
            mode="json", exclude={"violated_features", "passed_features"}
        )

    @staticmethod
    def _compliance_summary_text(summary: dict[str, Any]) -> str:
        if summary["total_norms"] == 0:
            return "Нормы с исполнимыми планами не найдены. Проверка соответствия не выполнена; отсутствие проверок не подтверждает отсутствие нарушений."
        parts = [
            f"Проверка завершена для {summary['total_norms']} норм.",
            f"Нарушено: {summary['violated_norms']}.",
            f"На проверенной части без нарушений: {summary['passed_norms']}.",
            f"Не удалось проверить: {summary['unverifiable_norms']}.",
            f"Не поддерживается: {summary['unsupported_norms']}.",
        ]
        if summary["partial_norms"]:
            parts.append(
                f"С частичным покрытием: {summary['partial_norms']}; вывод относится только к проверенной части."
            )
        violations = []
        for result in summary.get("results", []):
            count = result.get("summary", {}).get("violated_objects", 0)
            if result.get("compliance_status") != "violated" or count <= 0:
                continue
            source = result.get("source") or {}
            label = "; ".join(source_references(source))
            text = " ".join((source.get("extraction_text") or "").split())
            detail = f"- {label}: нарушений на объектах — {count}."
            if text:
                detail += f" Требование: {text}"
            if result.get("verification_status") == "partial":
                unchecked = result.get("coverage", {}).get("unchecked_objects", 0)
                detail += f" Не проверено объектов: {unchecked}."
            violations.append(detail)
        overview = " ".join(parts)
        if violations:
            return overview + "\n\nНарушенные нормы:\n\n" + "\n".join(violations)
        return overview

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

    async def _buf(self, request_id: str, event: dict) -> dict:
        """Persist before emitting: buffer the event to Redis and return it for yielding."""
        await self.state_store.buffer_event(request_id, event)
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

    async def generate_compliance_follow_up(
        self,
        model: str,
        prepared: PreparedComplianceFollowUp,
        temperature: float,
    ) -> AsyncGenerator[dict[str, str | dict[str, str | None | bool]], None]:
        """Answer from the persisted result without invoking compliance tools."""

        response_buffer: list[str] = []
        async for part in await self.llm_client.chat(
            model,
            prepared.messages,
            think=False,
            options={"temperature": min(temperature, 0.2)},
            stream=True,
        ):
            part: LlmChatResponse
            if part.message.content:
                response_buffer.append(part.message.content)
        raw_answer = "".join(
            response_buffer
        ) or self.compliance_result_harness.fallback_answer(prepared.summary)
        answer = self.compliance_result_harness.normalize_answer(
            prepared.summary, raw_answer
        )
        yield self._chunk(answer, done=False)
        yield self._chunk("", done=True)
        logger.debug(f"LLM compliance follow-up [{model}]: {answer}")

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
        parts: list[
            TextPartRequest
            | StatusPartRequest
            | ToolCallPartRequest
            | StructuredPartRequest
            | FilePartRequest
        ],
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
        parts: list[
            TextPartRequest
            | StatusPartRequest
            | ToolCallPartRequest
            | StructuredPartRequest
            | FilePartRequest
        ],
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
        parts: list[
            TextPartRequest
            | StatusPartRequest
            | ToolCallPartRequest
            | StructuredPartRequest
            | FilePartRequest
        ],
    ) -> None:
        if not text_buffer:
            return
        parts.append(
            TextPartRequest(kind="text", payload=TextPayload(text="".join(text_buffer)))
        )
        text_buffer.clear()

    def _add_tool_calls_to_parts(
        self,
        parts: list[
            TextPartRequest
            | StatusPartRequest
            | ToolCallPartRequest
            | StructuredPartRequest
            | FilePartRequest
        ],
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
        *,
        text_only: bool = False,
    ) -> (
        TextPartRequest
        | StatusPartRequest
        | StructuredPartRequest
        | FilePartRequest
        | None
    ):
        item_type = item.get("type")
        if item_type == "file":
            # History keeps the link like GenBuilder: without role and download_url.
            payload = {
                key: value
                for key, value in (item.get("content") or {}).items()
                if key not in {"role", "download_url"}
            }
            return FilePartRequest(kind="file", payload=payload)
        if text_only and item_type not in {"chunk", "clarification"}:
            return None
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
        if item_type == "clarification":
            return TextPartRequest(
                kind="text",
                payload=TextPayload(
                    text=content.get("question") or "Уточните параметры запроса."
                ),
            )
        if item_type in {
            "check_plan",
            "requirement_resolution",
            "compliance_result",
            "compliance_summary",
        }:
            return StructuredPartRequest(
                kind="data",
                payload={"event_type": item_type, "content": content},
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
                    "feature_collection": compact_layer(
                        feature_collection, "restrictions"
                    ),
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
