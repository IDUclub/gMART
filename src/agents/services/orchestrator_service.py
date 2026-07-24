from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    TextPartRequest,
    TextPayload,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.dvd_rag_service import DvdRagService
from src.agents.services.normgraph_rag_service import NormGraphRagService
from src.agents.services.orchestrator_catalog import (
    AGENT_CATALOG,
    AgentCatalogEntry,
    available_agents,
)
from src.agents.services.orchestrator_plan_builder import OrchestratorPlanBuilder
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus
from src.agents.services.provsion_service import ProvisionService
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities.orchestrator_plan import (
    OrchestratorAgent,
    OrchestratorPlan,
    OrchestratorPlanMode,
    OrchestratorStep,
)
from src.agents.services.urban_data_qa_service import UrbanDataQaService

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
    from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
    from src.agents.mcp_clients.urban_data_mcp_client import UrbanDataMcpClient

# Inner sub-agent event types that are never forwarded to the client: the outer
# stream announces the step itself (step_started) and owns the chat lifecycle.
_SUPPRESSED_INNER_EVENTS = {"pipeline_started", "service_event"}


class OrchestratorService(BaseLlmService):
    """
    Single entry point routing a user request across the gMART agents.

    An LLM planner maps the request onto a sequential plan of 1..3 steps over the
    restriction / provision / documents / norms agents (or a clarification
    question when nothing fits). Each step invokes the corresponding pipeline
    service **in-process** with ``persist_history=False`` and its own
    ``request_id``; the sub-agent's events are forwarded verbatim inside
    ``step_event`` envelopes. Between steps only a short text digest of the
    previous results is passed (no GeoJSON threading in v1).

    The orchestrator owns the ChatStorage lifecycle: it creates the chat, stores
    the user question and persists one combined assistant message with a text
    part per step, so follow-up requests in the same chat give the planner the
    full dialogue context.

    Reconnect (v1): every emitted event is buffered in Redis keyed by the outer
    ``request_id``; reconnecting with the same ``request_id`` replays the
    buffered events only — unfinished steps are not resumed.
    """

    DIGEST_MAX_CHARS = 1500

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client: ChatStorageApiClient,
        urban_api_client: UrbanApiClient,
        state_store: PipelineStateStore,
        restriction_service: RestrictionParserService,
        provision_service: ProvisionService,
        dvd_service: DvdRagService,
        normgraph_service: NormGraphRagService,
        urban_data_service: UrbanDataQaService,
        app_config: AgentsAppConfig,
    ) -> None:
        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.state_store = state_store
        self.restriction_service = restriction_service
        self.provision_service = provision_service
        self.dvd_service = dvd_service
        self.normgraph_service = normgraph_service
        self.urban_data_service = urban_data_service
        self.app_config = app_config
        self.plan_builder = OrchestratorPlanBuilder(self.llm_client)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_orchestration_pipeline(
        self,
        idu_mcp_client: "IduMcpClient",
        effects_mcp_client: "EffectsMcpClient",
        dvd_mcp_client: "DvdMcpClient | None",
        normgraph_mcp_client: "NormGraphMcpClient | None",
        urban_data_mcp_client: "UrbanDataMcpClient | None",
        token: str,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        is_reconnect = request_id is not None and await self.state_store.exists(
            request_id
        )
        if is_reconnect:
            logger.info(
                f"Orchestrator reconnect request_id={request_id}, "
                "replaying buffered events"
            )
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            return
        request_id = request_id or self.state_store.new_request_id()

        original_chat_id = chat_id
        yield self._buf(request_id, self._pipeline_started_event(request_id))

        # No chat_id supplied → create a new chat. Chat storage failures must not
        # break the stream: the pipeline keeps going without persistence.
        if not chat_id and persist_history:
            try:
                chat_id, title = await self.create_chat(
                    token,
                    model,
                    user_query,
                    additional_instructions=(
                        "Запрос направлен агенту-оркестратору, распределяющему "
                        "задачи между агентами платформы."
                    ),
                    scenario_id=scenario_id,
                )
                yield self._buf(request_id, self._chat_created_event(chat_id, title))
            except Exception as exc:
                logger.warning(f"Orchestrator: failed to create chat: {exc}")
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
            except Exception as exc:
                logger.warning(f"Orchestrator: failed to fetch chat history: {exc}")

        # A follow-up question in an existing chat is persisted here — create_chat
        # stores only the first one. Runs after the history fetch so the current
        # question doesn't also enter the planner context from storage.
        if persist_history and original_chat_id:
            try:
                await self.add_single_message(
                    token,
                    original_chat_id,
                    RoleEnum.USER,
                    user_query,
                    scenario_id=scenario_id,
                )
            except Exception as exc:
                logger.warning(f"Orchestrator: failed to persist user question: {exc}")

        # ── Planning ───────────────────────────────────────────────────
        yield self._buf(
            request_id,
            self._status("planning", "Определяю, какие агенты нужны для запроса…"),
        )
        agents = available_agents(self.app_config, scenario_id)
        plan = await self.plan_builder.build_plan(model, user_query, agents, history)

        if plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION:
            question = plan.clarification_question or ""
            yield self._buf(request_id, self._clarification_event(question))
            if persist_history:
                self._schedule_persist_text(token, chat_id, question, scenario_id)
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        # ── Execution ──────────────────────────────────────────────────
        yield self._buf(request_id, self._plan_event(plan))

        summary_steps: list[dict[str, Any]] = []
        digests: list[tuple[OrchestratorStep, str]] = []
        aborted = False

        for step_number, step in enumerate(plan.steps, start=1):
            if aborted:
                summary_steps.append(
                    self._summary_step(step_number, step, "skipped", "")
                )
                continue

            effective_query = self._compose_step_query(step, digests)
            step_request_id = self.state_store.new_request_id()
            yield self._buf(
                request_id,
                self._step_started_event(
                    step_number, step, step_request_id, effective_query
                ),
            )

            status = "completed"
            collected: dict[str, Any] = {"chunks": {}, "notes": []}
            try:
                pipeline = self._build_step_pipeline(
                    step,
                    effective_query,
                    step_request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    dvd_mcp_client,
                    normgraph_mcp_client,
                    urban_data_mcp_client,
                    token,
                    model,
                    temperature,
                    scenario_id,
                )
                async for item in pipeline:
                    if item.get("type") in _SUPPRESSED_INNER_EVENTS:
                        continue
                    self._collect_digest(collected, item)
                    yield self._buf(
                        request_id,
                        self._step_event(step_number, step, item),
                    )
                    if item.get("type") == "error":
                        status = "failed"
                        break
                    if item.get("type") == "pipeline_suspended":
                        status = "suspended"
                        break
            except PipelineSuspendedError:
                status = "suspended"
            except Exception as exc:
                logger.opt(exception=exc).error(
                    f"Orchestrator: step {step_number} ({step.agent}) failed"
                )
                status = "failed"

            digest = self._digest_from_collected(collected)
            yield self._buf(
                request_id,
                self._step_finished_event(step_number, step, status, digest),
            )
            summary_steps.append(self._summary_step(step_number, step, status, digest))
            if status == "completed":
                digests.append((step, digest))
            else:
                # Later steps consume earlier digests; running them after a
                # failure would produce misleading results — abort the plan.
                aborted = True

        yield self._buf(request_id, self._final_event(summary_steps))
        await self.state_store.set_status(
            request_id, PipelineStatus.FAILED if aborted else PipelineStatus.DONE
        )
        if persist_history:
            self._schedule_persist_summary(token, chat_id, summary_steps, scenario_id)

    # ------------------------------------------------------------------
    # Step dispatch (in-process pipeline invocation)
    # ------------------------------------------------------------------

    def _build_step_pipeline(
        self,
        step: OrchestratorStep,
        user_query: str,
        step_request_id: str,
        idu_mcp_client: "IduMcpClient",
        effects_mcp_client: "EffectsMcpClient",
        dvd_mcp_client: "DvdMcpClient | None",
        normgraph_mcp_client: "NormGraphMcpClient | None",
        urban_data_mcp_client: "UrbanDataMcpClient | None",
        token: str,
        model: str,
        temperature: float,
        scenario_id: int | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if step.agent == OrchestratorAgent.RESTRICTION:
            if scenario_id is None:
                raise ValueError("restriction step requires scenario_id")
            return self.restriction_service.run_restriction_execution_pipline(
                mcp_client=idu_mcp_client,
                temperature=temperature,
                model=model,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        if step.agent == OrchestratorAgent.PROVISION:
            if scenario_id is None:
                raise ValueError("provision step requires scenario_id")
            return self.provision_service.run_provision_pipeline(
                idu_mcp_client=idu_mcp_client,
                effects_mcp_client=effects_mcp_client,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        if step.agent == OrchestratorAgent.DOCUMENTS:
            if dvd_mcp_client is None:
                raise ValueError("documents step requires DVD_MCP_SERVER")
            return self.dvd_service.run_document_qa_pipeline(
                dvd_mcp_client=dvd_mcp_client,
                token=token,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        if step.agent == OrchestratorAgent.NORMS:
            if normgraph_mcp_client is None:
                raise ValueError("norms step requires NORM_GRAPH_MCP_SERVER")
            return self.normgraph_service.run_norms_qa_pipeline(
                normgraph_mcp_client=normgraph_mcp_client,
                token=token,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        if step.agent == OrchestratorAgent.URBAN_DATA:
            if urban_data_mcp_client is None:
                raise ValueError("urban_data step requires URBAN_DATA_MCP_SERVER")
            return self.urban_data_service.run_urban_data_qa_pipeline(
                urban_data_mcp_client=urban_data_mcp_client,
                token=token,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        raise ValueError(f"Unknown orchestrator agent: {step.agent}")

    # ------------------------------------------------------------------
    # Text digest between steps
    # ------------------------------------------------------------------

    def _compose_step_query(
        self,
        step: OrchestratorStep,
        digests: list[tuple[OrchestratorStep, str]],
    ) -> str:
        if not digests:
            return step.task
        context_lines = "\n".join(
            f"[Шаг {number}, {self._agent_title(prev.agent)}] {digest}"
            for number, (prev, digest) in enumerate(digests, start=1)
            if digest
        )
        if not context_lines:
            return step.task
        return (
            f"{step.task}\n\nКонтекст — результаты предыдущих шагов:\n{context_lines}"
        )

    @staticmethod
    def _collect_digest(collected: dict[str, Any], item: dict[str, Any]) -> None:
        content = item.get("content") or {}
        if not isinstance(content, dict):
            return
        if item.get("type") == "chunk":
            # DVD/norms tag chunks with the draft iteration; only the last
            # (accepted) draft belongs in the digest, so texts are keyed by it.
            iteration = int(content.get("iteration") or 0)
            collected["chunks"].setdefault(iteration, []).append(
                content.get("text") or ""
            )
        elif item.get("type") == "feature_collection" and content.get("name"):
            collected["notes"].append(f"Построен слой «{content['name']}».")
        elif item.get("type") == "table" and (
            content.get("title") or content.get("name")
        ):
            title = content.get("title") or content.get("name")
            collected["notes"].append(f"Сформирована таблица «{title}».")

    def _digest_from_collected(self, collected: dict[str, Any]) -> str:
        chunks: dict[int, list[str]] = collected["chunks"]
        text = "".join(chunks[max(chunks)]).strip() if chunks else ""
        parts = [part for part in (text, " ".join(collected["notes"])) if part]
        digest = "\n".join(parts)
        if len(digest) > self.DIGEST_MAX_CHARS:
            digest = digest[: self.DIGEST_MAX_CHARS - 1].rstrip() + "…"
        return digest

    # ------------------------------------------------------------------
    # Chat storage persistence (combined assistant answer)
    # ------------------------------------------------------------------

    def _schedule_persist_summary(
        self,
        token: str,
        chat_id: str | None,
        summary_steps: list[dict[str, Any]],
        scenario_id: int | None,
    ) -> None:
        text_blocks = [
            f"Шаг {step['step']} — {self._agent_title(step['agent'])}: "
            f"{step['task']}\n\n"
            + (step["summary"] or f"(шаг не выполнен: {step['status']})")
            for step in summary_steps
        ]
        if not text_blocks:
            return
        self._schedule_persist_parts(
            token,
            chat_id,
            [
                TextPartRequest(kind="text", payload=TextPayload(text=block))
                for block in text_blocks
            ],
            scenario_id,
        )

    def _schedule_persist_text(
        self,
        token: str,
        chat_id: str | None,
        text: str,
        scenario_id: int | None,
    ) -> None:
        if not text:
            return
        self._schedule_persist_parts(
            token,
            chat_id,
            [TextPartRequest(kind="text", payload=TextPayload(text=text))],
            scenario_id,
        )

    def _schedule_persist_parts(
        self,
        token: str,
        chat_id: str | None,
        parts: list[TextPartRequest],
        scenario_id: int | None,
    ) -> None:
        if not chat_id:
            return
        task = asyncio.create_task(
            self.add_complex_message(
                token, chat_id, RoleEnum.ASSISTANT, parts, scenario_id=scenario_id
            )
        )
        task.add_done_callback(self._log_persist_result)

    @staticmethod
    def _log_persist_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.exception(f"Orchestrator: failed to persist answer: {exc}")

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _buf(self, request_id: str, event: dict) -> dict:
        """Fire-and-forget buffer the event for reconnect replay, then return it."""
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    @staticmethod
    def _agent_title(agent: OrchestratorAgent | str) -> str:
        entry: AgentCatalogEntry | None = AGENT_CATALOG.get(OrchestratorAgent(agent))
        return entry.title if entry else str(agent)

    @staticmethod
    def _pipeline_started_event(request_id: str) -> dict:
        return {"type": "pipeline_started", "content": {"request_id": request_id}}

    @staticmethod
    def _status(status: str, text: str) -> dict:
        return {"type": "status", "content": {"status": status, "text": text}}

    def _plan_event(self, plan: OrchestratorPlan) -> dict:
        return {
            "type": "plan",
            "content": {
                "steps": [
                    {
                        "step": number,
                        "agent": step.agent.value,
                        "agent_title": self._agent_title(step.agent),
                        "task": step.task,
                    }
                    for number, step in enumerate(plan.steps, start=1)
                ]
            },
        }

    @staticmethod
    def _step_started_event(
        step_number: int,
        step: OrchestratorStep,
        step_request_id: str,
        task: str,
    ) -> dict:
        return {
            "type": "step_started",
            "content": {
                "step": step_number,
                "agent": step.agent.value,
                "step_request_id": step_request_id,
                "task": task,
            },
        }

    @staticmethod
    def _step_event(
        step_number: int, step: OrchestratorStep, item: dict[str, Any]
    ) -> dict:
        return {
            "type": "step_event",
            "content": {
                "step": step_number,
                "agent": step.agent.value,
                "event": item,
            },
        }

    @staticmethod
    def _step_finished_event(
        step_number: int, step: OrchestratorStep, status: str, summary: str
    ) -> dict:
        return {
            "type": "step_finished",
            "content": {
                "step": step_number,
                "agent": step.agent.value,
                "status": status,
                "summary": summary,
            },
        }

    @staticmethod
    def _clarification_event(question: str) -> dict:
        return {"type": "clarification", "content": {"question": question}}

    @staticmethod
    def _summary_step(
        step_number: int, step: OrchestratorStep, status: str, summary: str
    ) -> dict[str, Any]:
        return {
            "step": step_number,
            "agent": step.agent.value,
            "task": step.task,
            "status": status,
            "summary": summary,
        }

    @staticmethod
    def _final_event(summary_steps: list[dict[str, Any]]) -> dict:
        return {"type": "orchestrator_final", "content": {"steps": summary_steps}}

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
