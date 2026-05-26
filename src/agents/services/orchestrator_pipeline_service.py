from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from loguru import logger

from src.agents.model_clients.base_client import BaseLlmClient
from src.agents.services.critic_service import CriticService, CriticVerdict
from src.agents.services.orchestrator_service import (
    OrchestratorIntent,
    OrchestratorService,
)
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus

if TYPE_CHECKING:
    from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
    from src.agents.services.provsion_service import ProvisionService
    from src.agents.services.restriction_parser_service import RestrictionParserService


class OrchestratorPipelineService(BaseLlmClient):
    """
    Pipeline service for the orchestrator REST (frontend) endpoint.

    Classifies user intent via LLM, optionally decomposes compound queries,
    delegates to RestrictionParserService and/or ProvisionService, and runs a
    critic agent after each sub-pipeline to verify result quality.

    Critic / retry logic
    --------------------
    After each sub-pipeline's final ``chunk done:true`` event the critic
    evaluates the collected text against the user's (sub-)query.  If the
    verdict is ``"poor"`` the orchestrator emits a ``critique`` event and
    automatically re-runs the sub-pipeline once with a refined query produced
    by the critic.  At most ``CriticService.MAX_RETRIES`` automatic retries
    are performed per sub-pipeline.

    Reconnect / token-refresh support
    ----------------------------------
    A ``pipeline_started`` event is emitted with the orchestrator-level
    ``request_id``.  The client must persist this ID and pass it back on
    reconnect.  Retry state (``*_retry_request_id``, ``*_refined_query``) is
    persisted to Redis *before* the retry starts so that reconnect always
    resumes from the correct point.

    Compatible with ``stream_with_error_handling``:
    ``run_orchestrator_pipeline`` uses keyword-only params (``*``) and inherits
    ``execute_request`` from ``BaseLlmClient`` for error-recovery prompts.

    Attributes:
        host (str): Ollama host URL.
        orchestrator_service (OrchestratorService): LLM intent classifier.
        restriction_service (RestrictionParserService): Restriction pipeline.
        provision_service (ProvisionService): Provision effects pipeline.
        critic_service (CriticService): LLM-based result quality evaluator.
        state_store (PipelineStateStore): Redis-backed pipeline state store.
    """

    def __init__(
        self,
        host: str,
        orchestrator_service: OrchestratorService,
        restriction_service: RestrictionParserService,
        provision_service: ProvisionService,
        critic_service: CriticService,
        state_store: PipelineStateStore,
    ) -> None:
        super().__init__(host=host)
        self.orchestrator_service = orchestrator_service
        self.restriction_service = restriction_service
        self.provision_service = provision_service
        self.critic_service = critic_service
        self.state_store = state_store

    # ── Public entry point ────────────────────────────────────────────────────

    async def run_orchestrator_pipeline(
        self,
        *,
        model: str,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
        temperature: float,
        user_query: str,
        scenario_id: int,
        project_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """
        Run the orchestrator pipeline: classify intent → decompose (if compound)
        → delegate to sub-pipelines → evaluate each result with critic → retry
        once if quality is poor.

        Args:
            model (str): Ollama model name.
            idu_mcp_client (IduMcpClient): MCP client for IDU geospatial tools.
            effects_mcp_client (EffectsMcpClient): MCP client for effects calculation.
            temperature (float): LLM sampling temperature.
            user_query (str): Natural-language user request.
            scenario_id (int): Urban API scenario ID.
            project_id (int | None): Project ID for provision effects.
            chat_id (str | None): Existing Chat Storage chat UUID.
            request_id (str | None): Orchestrator request ID for reconnect.
        Yields:
            dict: SSE event dict compatible with OrchestratorResponse schema.
        """
        # ── Reconnect detection ───────────────────────────────────────────────
        is_reconnect = request_id is not None and await self.state_store.exists(
            request_id
        )

        # Fields populated differently on fresh run vs reconnect:
        restriction_request_id: str | None
        provision_request_id: str | None
        restriction_retry_request_id: str | None
        provision_retry_request_id: str | None
        restriction_refined_query: str | None
        provision_refined_query: str | None

        if is_reconnect:
            logger.info(
                f"OrchestratorPipelineService: reconnect for request_id={request_id}"
            )
            for event in await self.state_store.get_buffered_events(request_id):
                yield event

            orch_state = await self.state_store.get_state(request_id) or {}
            intent = OrchestratorIntent(
                needs_restriction=bool(orch_state.get("needs_restriction", True)),
                needs_provision=bool(orch_state.get("needs_provision", False)),
                restriction_query=orch_state.get("restriction_query"),
                provision_query=orch_state.get("provision_query"),
            )
            restriction_request_id = orch_state.get("restriction_request_id")
            provision_request_id = orch_state.get("provision_request_id")
            restriction_retry_request_id = orch_state.get(
                "restriction_retry_request_id"
            )
            provision_retry_request_id = orch_state.get("provision_retry_request_id")
            restriction_refined_query = orch_state.get("restriction_refined_query")
            provision_refined_query = orch_state.get("provision_refined_query")
            chat_id = chat_id or orch_state.get("chat_id")
            # Restore pipeline parameters from state so retry sub-pipelines always
            # use the original values regardless of what the reconnecting client sends.
            # Bug fix: project_id was not persisted before; now it is — restore it.
            if project_id is None and orch_state.get("project_id") is not None:
                project_id = int(orch_state["project_id"])
            model = orch_state.get("model") or model
            temperature = float(orch_state.get("temperature") or temperature)
            scenario_id = int(orch_state.get("scenario_id") or scenario_id)

        else:
            # ── Fresh run ─────────────────────────────────────────────────────
            request_id = request_id or self.state_store.new_request_id()
            restriction_request_id = self.state_store.new_request_id()
            provision_request_id = self.state_store.new_request_id()
            restriction_retry_request_id = None
            provision_retry_request_id = None
            restriction_refined_query = None
            provision_refined_query = None

            yield self._buf(request_id, self._pipeline_started_event(request_id))

            # ── Phase 1: intent classification ────────────────────────────────
            yield self._buf(request_id, self._routing("Определяю тип запроса..."))

            try:
                intent = await self.orchestrator_service.classify_intent(
                    user_query, model
                )
            except Exception as exc:
                logger.warning(
                    f"OrchestratorPipelineService: classification error ({exc}). "
                    "Defaulting to restriction-only."
                )
                intent = OrchestratorIntent(
                    needs_restriction=True, needs_provision=False
                )

            if intent.is_empty:
                yield self._buf(
                    request_id,
                    self._routing(
                        "Запрос не относится к геопространственному анализу. "
                        "Пожалуйста, уточните задачу: задайте зоны ограничений "
                        "или запросите расчёт обеспеченности сервисами."
                    ),
                )
                yield self._buf(
                    request_id,
                    {
                        "type": "chunk",
                        "content": {
                            "text": (
                                "Не удалось определить тип запроса. "
                                "Уточните задачу — например: «Зона ограничения вокруг школ 200 м» "
                                "или «Рассчитай эффекты обеспеченности для детских садов»."
                            ),
                            "done": True,
                        },
                    },
                )
                return

            # ── Phase 1.5: query decomposition (compound requests only) ───────
            if intent.is_compound:
                yield self._buf(
                    request_id,
                    self._routing(
                        "Составной запрос: разбиваю на подзадачи для каждого агента..."
                    ),
                )
                restriction_query, provision_query = (
                    await self.orchestrator_service.decompose_query(user_query, model)
                )
                intent.restriction_query = restriction_query
                intent.provision_query = provision_query

            # ── Phase 2: routing notification ─────────────────────────────────
            labels: list[str] = []
            if intent.needs_restriction:
                labels.append("restriction-creation-agent")
            if intent.needs_provision:
                labels.append("provision-effects-agent")
            yield self._buf(
                request_id,
                self._routing(f"Перенаправляю запрос: {', '.join(labels)}."),
            )

            # Persist state before delegating so any reconnect can restore it.
            await self.state_store.create_generic(
                request_id,
                {
                    "status": PipelineStatus.RUNNING,
                    "needs_restriction": intent.needs_restriction,
                    "needs_provision": intent.needs_provision,
                    "restriction_query": intent.restriction_query,
                    "provision_query": intent.provision_query,
                    "restriction_request_id": restriction_request_id,
                    "provision_request_id": provision_request_id,
                    # Retry fields — populated later if critic triggers a retry:
                    "restriction_retry_request_id": None,
                    "restriction_refined_query": None,
                    "provision_retry_request_id": None,
                    "provision_refined_query": None,
                    "chat_id": chat_id,
                    "user_query": user_query,
                    "scenario_id": scenario_id,
                    "project_id": project_id,
                    "model": model,
                    "temperature": temperature,
                },
            )

        # ── Phase 3: restriction sub-pipeline ─────────────────────────────────
        if intent.needs_restriction:
            effective_query = intent.restriction_query or user_query

            yield self._routing("Запускаю агент ограничений...")
            restriction_text: list[str] = []
            async for (
                event
            ) in self.restriction_service.run_restriction_execution_pipline(
                mcp_client=idu_mcp_client,
                temperature=temperature,
                model=model,
                user_query=effective_query,
                scenario_id=scenario_id,
                chat_id=chat_id,
                request_id=restriction_request_id,
            ):
                chat_id = self._chat_id_from_event(event) or chat_id
                if not is_reconnect:
                    self._collect_chunk_text(event, restriction_text)
                yield event

            if not is_reconnect:
                # ── Critic evaluation ─────────────────────────────────────────
                verdict = await self.critic_service.evaluate(
                    user_query=effective_query,
                    response_text="".join(restriction_text),
                    agent_name="restriction-creation-agent",
                    model=model,
                )
                will_retry = (
                    verdict.needs_retry and restriction_retry_request_id is None
                )
                yield self._critique_event(
                    "restriction-creation-agent", verdict, retried=will_retry
                )

                if will_retry:
                    restriction_refined_query = verdict.refined_query
                    restriction_retry_request_id = self.state_store.new_request_id()
                    # Persist retry state BEFORE starting retry (for reconnect safety).
                    # chat_id is included so a reconnect during the retry run uses the
                    # correct chat (which may have been created during the first run).
                    await self.state_store.update_state(
                        request_id,
                        {
                            "restriction_retry_request_id": restriction_retry_request_id,
                            "restriction_refined_query": restriction_refined_query,
                            "chat_id": chat_id,
                        },
                    )
                    yield self._routing(
                        "Агент ограничений: обнаружены недочёты, выполняю повторный запуск..."
                    )
                    async for (
                        event
                    ) in self.restriction_service.run_restriction_execution_pipline(
                        mcp_client=idu_mcp_client,
                        temperature=temperature,
                        model=model,
                        user_query=restriction_refined_query,
                        scenario_id=scenario_id,
                        chat_id=chat_id,
                        request_id=restriction_retry_request_id,
                    ):
                        chat_id = self._chat_id_from_event(event) or chat_id
                        yield event

            elif (
                is_reconnect
                and restriction_retry_request_id
                and restriction_refined_query
            ):
                # Reconnect: retry was already decided — resume / replay it.
                yield self._routing(
                    "Агент ограничений: обнаружены недочёты, выполняю повторный запуск..."
                )
                async for (
                    event
                ) in self.restriction_service.run_restriction_execution_pipline(
                    mcp_client=idu_mcp_client,
                    temperature=temperature,
                    model=model,
                    user_query=restriction_refined_query,
                    scenario_id=scenario_id,
                    chat_id=chat_id,
                    request_id=restriction_retry_request_id,
                ):
                    chat_id = self._chat_id_from_event(event) or chat_id
                    yield event

        # ── Phase 4: provision sub-pipeline ───────────────────────────────────
        if intent.needs_provision:
            if project_id is None:
                yield self._routing(
                    "Расчёт обеспеченности пропущен: параметр project_id не передан. "
                    "Укажите project_id в запросе для получения эффектов обеспеченности."
                )
            else:
                effective_query = intent.provision_query or user_query

                yield self._routing("Запускаю агент обеспеченности...")
                provision_text: list[str] = []
                async for event in self.provision_service.run_provision_pipeline(
                    idu_mcp_client=idu_mcp_client,
                    effects_mcp_client=effects_mcp_client,
                    temperature=temperature,
                    model=model,
                    user_query=effective_query,
                    project_id=project_id,
                    scenario_id=scenario_id,
                    chat_id=chat_id,
                    request_id=provision_request_id,
                ):
                    if not is_reconnect:
                        self._collect_chunk_text(event, provision_text)
                    yield event

                if not is_reconnect:
                    # ── Critic evaluation ─────────────────────────────────────
                    verdict = await self.critic_service.evaluate(
                        user_query=effective_query,
                        response_text="".join(provision_text),
                        agent_name="provision-effects-agent",
                        model=model,
                    )
                    will_retry = (
                        verdict.needs_retry and provision_retry_request_id is None
                    )
                    yield self._critique_event(
                        "provision-effects-agent", verdict, retried=will_retry
                    )

                    if will_retry:
                        provision_refined_query = verdict.refined_query
                        provision_retry_request_id = self.state_store.new_request_id()
                        # Include chat_id so a reconnect during the retry uses the
                        # correct chat (which may have been created during Phase 3/4).
                        await self.state_store.update_state(
                            request_id,
                            {
                                "provision_retry_request_id": provision_retry_request_id,
                                "provision_refined_query": provision_refined_query,
                                "chat_id": chat_id,
                            },
                        )
                        yield self._routing(
                            "Агент обеспеченности: обнаружены недочёты, выполняю повторный запуск..."
                        )
                        async for (
                            event
                        ) in self.provision_service.run_provision_pipeline(
                            idu_mcp_client=idu_mcp_client,
                            effects_mcp_client=effects_mcp_client,
                            temperature=temperature,
                            model=model,
                            user_query=provision_refined_query,
                            project_id=project_id,
                            scenario_id=scenario_id,
                            chat_id=chat_id,
                            request_id=provision_retry_request_id,
                        ):
                            yield event

                elif (
                    is_reconnect
                    and provision_retry_request_id
                    and provision_refined_query
                ):
                    yield self._routing(
                        "Агент обеспеченности: обнаружены недочёты, выполняю повторный запуск..."
                    )
                    async for event in self.provision_service.run_provision_pipeline(
                        idu_mcp_client=idu_mcp_client,
                        effects_mcp_client=effects_mcp_client,
                        temperature=temperature,
                        model=model,
                        user_query=provision_refined_query,
                        project_id=project_id,
                        scenario_id=scenario_id,
                        chat_id=chat_id,
                        request_id=provision_retry_request_id,
                    ):
                        yield event

        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _buf(self, request_id: str, event: dict) -> dict:
        """Buffer *event* to Redis (fire-and-forget) and return it for yielding."""
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    @staticmethod
    def _pipeline_started_event(request_id: str) -> dict:
        return {"type": "pipeline_started", "content": {"request_id": request_id}}

    @staticmethod
    def _routing(text: str) -> dict:
        """Build an orchestrator routing event (not buffered — emitted fresh)."""
        return {"type": "routing", "content": {"text": text}}

    @staticmethod
    def _critique_event(agent: str, verdict: CriticVerdict, retried: bool) -> dict:
        """Build a critique event (not buffered — re-emitted from state on reconnect)."""
        return {
            "type": "critique",
            "content": {
                "agent": agent,
                "quality": verdict.quality,
                "feedback": verdict.feedback,
                "retried": retried,
            },
        }

    @staticmethod
    def _collect_chunk_text(event: dict, buffer: list[str]) -> None:
        """Append text from a chunk event into *buffer* for critic evaluation."""
        if event.get("type") == "chunk":
            text = (event.get("content") or {}).get("text") or ""
            if text:
                buffer.append(text)

    @staticmethod
    def _chat_id_from_event(event: dict) -> str | None:
        """Extract chat_id from a chat_created storage event, if present."""
        content = event.get("content") or {}
        inner_event = content.get("event") or {}
        if inner_event.get("storage_event_type") == "chat_created":
            return inner_event.get("chat_id")
        return None
