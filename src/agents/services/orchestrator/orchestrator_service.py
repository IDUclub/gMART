from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import TYPE_CHECKING, Any

from fastapi.encoders import jsonable_encoder
from loguru import logger

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    StructuredPartRequest,
    TablePartRequest,
    TablePayload,
    TextPartRequest,
    TextPayload,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.common.exceptions.base_exceptions import AgentsNotFound
from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.runtime.budget import BudgetExceeded, RunBudget, budget_scope
from src.agents.runtime.tools import stream_planned
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.dvd.dvd_rag_service import DvdRagService
from src.agents.services.normgraph.normgraph_rag_service import NormGraphRagService
from src.agents.services.orchestrator.analysis import AnalyticalRun, artifact_parts
from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import (
    GoalClarification,
    GoalManager,
    GoalState,
)
from src.agents.services.orchestrator.analysis_support import (
    blocker_text,
    configured_limits,
    context_scope,
    missing_input,
)
from src.agents.services.orchestrator.orchestrator_catalog import (
    AGENT_CATALOG,
    AgentCatalogEntry,
    available_agents,
)
from src.agents.services.orchestrator.orchestrator_plan_builder import (
    OrchestratorPlanBuilder,
)
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService
from src.agents.services.service_entities.orchestrator_plan import (
    OrchestratorAgent,
    OrchestratorPlan,
    OrchestratorPlanMode,
    OrchestratorStep,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
    from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
    from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient

# Inner sub-agent event types that are never forwarded to the client: the outer
# stream announces the step itself (step_started) and owns the chat lifecycle.
_SUPPRESSED_INNER_EVENTS = {"pipeline_started", "service_event"}


class OrchestratorService(BaseLlmService):
    """Route simple tasks and run bounded analytical investigations through SDK.

    Specialists execute in-process with independent request IDs. The orchestrator
    owns chat persistence, artifact delivery and replay. Analytical tasks review
    evidence between steps and can continue from a scoped saved context.
    See docs/analytical-orchestrator.md for limits and the SSE contract.
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
        app_config: AgentsAppConfig,
        scenario_data_service: ScenarioDataService | None = None,
        planning_services: dict | None = None,
        variant_provision_service=None,
    ) -> None:
        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.state_store = state_store
        self.restriction_service = restriction_service
        self.provision_service = provision_service
        self.scenario_data_service = scenario_data_service
        self.dvd_service = dvd_service
        self.normgraph_service = normgraph_service
        self.app_config = app_config
        self.planning_services = planning_services or {}
        self.variant_provision_service = variant_provision_service
        self.plan_builder = OrchestratorPlanBuilder(self.llm_client)
        self.goal_manager = GoalManager(self.llm_client)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run_orchestration_pipeline(
        self,
        idu_mcp_client: "IduMcpClient",
        effects_mcp_client: "EffectsMcpClient",
        dvd_mcp_client: "DvdMcpClient | None",
        normgraph_mcp_client: "NormGraphMcpClient | None",
        token: str,
        model: str | None,
        temperature: float,
        user_query: str,
        scenario_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
        urban_mcp_client: "UrbanMcpClient | None" = None,
        budget_tokens: int | None = None,
        budget_seconds: float | None = None,
        continue_from: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if request_id and await self.state_store.exists(request_id):
            await self.check_replay_owner(request_id, token)
            for event in await self.state_store.get_buffered_events(
                request_id, owner=context_scope(token, "owner")
            ):
                yield event
            return
        request_id = request_id or self.state_store.new_request_id()
        claimed = await self.state_store.create(
            request_id,
            chat_id=chat_id,
            user_query=user_query,
            scenario_id=scenario_id,
            model=model,
            temperature=temperature,
            owner=context_scope(token, "owner"),
            claim_id=self.state_store.new_request_id(),
        )
        if not claimed:
            await self.check_replay_owner(request_id, token)
            for event in await self.state_store.get_buffered_events(
                request_id, owner=context_scope(token, "owner")
            ):
                yield event
            return
        args = dict(
            idu_mcp_client=idu_mcp_client,
            effects_mcp_client=effects_mcp_client,
            dvd_mcp_client=dvd_mcp_client,
            normgraph_mcp_client=normgraph_mcp_client,
            token=token,
            model=model,
            temperature=temperature,
            user_query=user_query,
            scenario_id=scenario_id,
            chat_id=chat_id,
            request_id=request_id,
            persist_history=persist_history,
            urban_mcp_client=urban_mcp_client,
            continue_from=continue_from,
        )
        budget = RunBudget(configured_limits(budget_tokens, budget_seconds))
        with budget_scope(budget):
            try:
                async with asyncio.timeout(budget.remaining_seconds):
                    async with aclosing(
                        self._run_orchestration_pipeline(**args)
                    ) as events:
                        async for event in events:
                            yield event
            except (BudgetExceeded, TimeoutError, ValueError, LlmResponseError) as exc:
                logger.warning(
                    "Orchestrator control failed ({}): {}",
                    type(exc).__name__,
                    str(exc).splitlines()[0][:250] if str(exc) else "deadline",
                )
                missing = [
                    missing_input(
                        exc.resource
                        if isinstance(exc, BudgetExceeded)
                        else "time" if isinstance(exc, TimeoutError) else "planning"
                    )
                ]
                saved = await self.state_store.get_analysis_context(
                    context_scope(token, "run:" + request_id)
                )
                context = AnalysisContext(saved)
                steps = (saved or {}).get("steps", [])
                if not saved:
                    # A deadline may fire while the outer SSE buffer is being
                    # written, outside the specialist generator's timeout scope.
                    started = {}
                    for event in await self.state_store.get_buffered_events(
                        request_id, owner=context_scope(token, "owner")
                    ):
                        content = event.get("content", {})
                        if event["type"] == "step_started":
                            started[content["step"]] = content
                        elif (
                            event["type"] == "step_event" and content["step"] in started
                        ):
                            info = started[content["step"]]
                            context.add_artifact(
                                content["event"],
                                content["step"],
                                info["step_request_id"],
                            )
                        elif (
                            event["type"] == "step_finished"
                            and content["step"] in started
                        ):
                            info = started[content["step"]]
                            context.finish(
                                content["step"],
                                info["task"],
                                scenario_id,
                                content["status"],
                                content["summary"],
                                info["step_request_id"],
                            )
                            steps.append({**content, "task": info["task"]})
                    context.query = user_query
                    await self.state_store.save_analysis_context(
                        context_scope(token, "run:" + request_id), context.dump()
                    )
                answer = blocker_text(
                    missing, sum(c["status"] == "completed" for c in context.completed)
                )
                final = {
                    "type": "orchestrator_final",
                    "content": {
                        "steps": steps,
                        "status": "blocked",
                        "answer": answer,
                        "artifacts": context.index(),
                        "missing": [m.model_dump() for m in missing],
                        "budget": budget.snapshot(),
                        "continue_from": request_id,
                        "goal": GoalState(context).view() if context.goal else None,
                    },
                }
                if persist_history:
                    state = await self.state_store.get_state(request_id)
                    history_chat_id = chat_id or (state or {}).get("chat_id")
                    parts = [
                        TextPartRequest(kind="text", payload=TextPayload(text=answer)),
                        *artifact_parts(context),
                        StructuredPartRequest(
                            kind="data",
                            payload={
                                "event_type": "analysis_context",
                                "content": {
                                    **context.dump(),
                                    "continue_from": request_id,
                                    "status": "blocked",
                                    "scenario_id": scenario_id,
                                },
                            },
                        ),
                    ]
                    if history_chat_id:
                        try:
                            await self.add_complex_message(
                                token,
                                history_chat_id,
                                RoleEnum.ASSISTANT,
                                parts,
                                scenario_id=scenario_id,
                            )
                        except Exception:
                            logger.opt(exception=False).warning(
                                "Could not persist blocked analysis to ChatStorage"
                            )
                            yield await self._buf(
                                request_id,
                                {
                                    "type": "warning",
                                    "content": {
                                        "code": "history_unavailable",
                                        "message": "История чата недоступна. Сохранённый анализ можно продолжить по идентификатору текущего запроса до истечения срока хранения.",
                                    },
                                },
                            )
                await self.state_store.set_status(request_id, PipelineStatus.FAILED)
                yield await self._buf(request_id, final)

    async def _run_orchestration_pipeline(
        self,
        idu_mcp_client: "IduMcpClient",
        effects_mcp_client: "EffectsMcpClient",
        dvd_mcp_client: "DvdMcpClient | None",
        normgraph_mcp_client: "NormGraphMcpClient | None",
        token: str,
        model: str | None,
        temperature: float,
        user_query: str,
        scenario_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
        urban_mcp_client: "UrbanMcpClient | None" = None,
        continue_from: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Fill in the provider's model when the caller named none; keeps REST and A2A
        # on one behaviour and out of backend-specific literals.
        model = await self.resolve_model(model)
        request_id = request_id or self.state_store.new_request_id()

        original_chat_id = chat_id
        yield await self._buf(request_id, self._pipeline_started_event(request_id))

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
                    agent_id="orchestrator",
                )
                yield await self._buf(
                    request_id, self._chat_created_event(chat_id, title)
                )
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
            owner=context_scope(token, "owner"),
        )

        history: list[dict] = []
        context = AnalysisContext()
        saved = None
        scope = context_scope(
            token, "run:" + continue_from if continue_from else chat_id
        )
        if scope:
            saved = await self.state_store.get_analysis_context(scope)
            if saved:
                context = AnalysisContext(saved)
                if scenario_id is None:
                    scenario_id = saved.get("scenario_id")
        if original_chat_id:
            try:
                chat_info = await self.get_chat_messages(token, original_chat_id)
                history = self.build_llm_history(
                    chat_info.messages, current_user_query=user_query
                )
                if not saved and not continue_from:
                    for message in reversed(chat_info.messages):
                        if message.get("role") != "assistant":
                            continue
                        snapshot = next(
                            (
                                part.get("payload", {}).get("content")
                                for part in message.get("parts", [])
                                if part.get("kind") == "data"
                                and part.get("payload", {}).get("event_type")
                                == "analysis_context"
                            ),
                            None,
                        )
                        if snapshot:
                            context = AnalysisContext(snapshot)
                            break
            except Exception as exc:
                logger.warning(f"Orchestrator: failed to fetch chat history: {exc}")

        if continue_from and not saved:
            question = "Сохранённый анализ недоступен или срок хранения истёк. Укажите исходные сценарии, показатели и изменённые условия, чтобы восстановить задачу."
            yield await self._buf(request_id, self._clarification_event(question))
            await self.state_store.set_status(request_id, PipelineStatus.FAILED)
            return
        goal_mode = os.getenv("ORCHESTRATOR_ANALYSIS_MODE", "goal") == "goal"
        if context.completed:
            # Full historical tables never enter the prompt. Their loss-aware
            # index and selected evidence replace unbounded dialogue history.
            history = [
                {"role": "user", "content": context.query},
                {
                    "role": "assistant",
                    "content": json.dumps(context.view(), ensure_ascii=False),
                },
            ]
        elif saved and context.query:
            history = [{"role": "user", "content": context.query}]

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
        yield await self._buf(
            request_id,
            self._status("planning", "Определяю, какие агенты нужны для запроса…"),
        )
        agents = available_agents(self.app_config, scenario_id)
        if context.completed or goal_mode:
            plan = OrchestratorPlan(mode=OrchestratorPlanMode.EXECUTE, analytical=True)
        else:
            plan = await self.plan_builder.build_plan(
                model, user_query, agents, history, scenario_id=scenario_id
            )

        if plan.mode == OrchestratorPlanMode.NEEDS_CLARIFICATION:
            question = plan.clarification_question or ""
            yield await self._buf(request_id, self._clarification_event(question))
            if persist_history:
                self._schedule_persist_text(token, chat_id, question, scenario_id)
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        # ── Execution ──────────────────────────────────────────────────
        yield await self._buf(request_id, self._plan_event(plan))

        if plan.analytical:
            if goal_mode:
                if not (continue_from and context.goal):
                    goal_query = (
                        f"{context.query}\nУточнение для продолжения: {user_query}"
                        if continue_from and context.query
                        else user_query
                    )
                    # A failed goal call is still a resumable analytical run.
                    # Preserve the inherited evidence and both user requests
                    # before inference; the previous goal cannot stand in for
                    # an as-yet uncreated goal for the follow-up.
                    pending_query = (
                        context.query + "\nУточнение: " + goal_query
                        if context.query
                        and context.query != goal_query
                        and not continue_from
                        else goal_query
                    )
                    pending = {
                        **context.dump(),
                        "goal": None,
                        "query": pending_query,
                        "scenario_id": scenario_id,
                        "steps": [],
                        "remaining": [],
                    }
                    for key in ("run:" + request_id, chat_id):
                        pending_scope = context_scope(token, key)
                        if pending_scope:
                            await self.state_store.save_analysis_context(
                                pending_scope, pending
                            )
                    goal = await self.goal_manager.create(
                        model, goal_query, agents, scenario_id, history
                    )
                    if isinstance(goal, GoalClarification):
                        yield await self._buf(
                            request_id, self._clarification_event(goal.question)
                        )
                        if persist_history:
                            self._schedule_persist_text(
                                token, chat_id, goal.question, scenario_id
                            )
                        await self.state_store.set_status(
                            request_id, PipelineStatus.DONE
                        )
                        return
                    GoalState(context, goal)
                    context.query = goal_query
                # Successful operations survive explicit continuation. A blocked
                # requirement may be retried after the user supplies new input.
                elif context.goal:
                    GoalState(context).resume()
                plan = OrchestratorPlan(
                    mode=OrchestratorPlanMode.EXECUTE, analytical=True
                )
            args = dict(
                idu_mcp_client=idu_mcp_client,
                effects_mcp_client=effects_mcp_client,
                dvd_mcp_client=dvd_mcp_client,
                normgraph_mcp_client=normgraph_mcp_client,
                urban_mcp_client=urban_mcp_client,
                token=token,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                chat_id=chat_id,
                request_id=request_id,
                persist_history=persist_history,
            )
            async with aclosing(
                AnalyticalRun(self, plan, agents, args, context).events()
            ) as events:
                async for event in events:
                    yield await self._buf(request_id, event)
            return

        summary_steps: list[dict[str, Any]] = []
        digests: list[tuple[OrchestratorStep, str]] = []
        table_parts: list[TablePartRequest] = []
        aborted = False

        for step_number, step in enumerate(plan.steps, start=1):
            if aborted:
                summary_steps.append(
                    self._summary_step(step_number, step, "skipped", "")
                )
                continue

            effective_query = self._compose_step_query(step, digests)
            step_request_id = self.state_store.new_request_id()
            yield await self._buf(
                request_id,
                self._step_started_event(
                    step_number, step, step_request_id, effective_query
                ),
            )

            status = "completed"
            collected: dict[str, Any] = {"chunks": {}, "notes": []}
            step_tables: list[TablePartRequest] = []
            try:
                pipeline = self._build_step_pipeline(
                    step,
                    effective_query,
                    step_request_id,
                    idu_mcp_client,
                    effects_mcp_client,
                    dvd_mcp_client,
                    normgraph_mcp_client,
                    urban_mcp_client,
                    token,
                    model,
                    temperature,
                    step.scenario_id or scenario_id,
                )
                async with aclosing(
                    stream_planned(f"orchestrator.{step.agent}", pipeline)
                ) as events:
                    async for item in events:
                        if item.get("type") in _SUPPRESSED_INNER_EVENTS:
                            continue
                        self._collect_digest(collected, item)
                        artifact_id = context.add_artifact(
                            item, step_number, step_request_id
                        )
                        if artifact_id:
                            item = {
                                **item,
                                "content": {
                                    **item["content"],
                                    "artifact_id": artifact_id,
                                },
                            }
                        table_part = self._table_part(item)
                        if table_part is not None:
                            step_tables.append(table_part)
                        yield await self._buf(
                            request_id,
                            self._step_event(step_number, step, item),
                        )
                        if item.get("type") in {"error", "pipeline_failed"}:
                            status = "failed"
                            break
                        if item.get("type") in {
                            "clarification",
                            "clarification_required",
                        }:
                            status = "needs_clarification"
                            content = item.get("content") or {}
                            collected = {
                                "chunks": {},
                                "notes": [
                                    content.get("question")
                                    or content.get("text")
                                    or "Для выполнения шага требуется уточнение пользователя."
                                ],
                            }
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
            if status in {"failed", "suspended"}:
                # Streamed text can be an unverified draft. Never turn a failed
                # draft into a saved answer or evidence for a later step.
                digest = (
                    "Шаг не выполнен: агент сообщил об ошибке. Результат не подтверждён."
                    if status == "failed"
                    else "Шаг приостановлен. Результат ещё не подтверждён."
                )
            yield await self._buf(
                request_id,
                self._step_finished_event(step_number, step, status, digest),
            )
            summary_steps.append(self._summary_step(step_number, step, status, digest))
            context.finish(
                step_number,
                step.task,
                step.scenario_id or scenario_id,
                status,
                digest,
                step_request_id,
            )
            if status == "completed":
                digests.append((step, digest))
                table_parts.extend(step_tables)
            else:
                # Later steps consume earlier digests; running them after a
                # failure would produce misleading results — abort the plan.
                aborted = True

        if aborted:
            yield await self._buf(
                request_id,
                self._clarification_event(
                    blocker_text([missing_input("service")], len(digests))
                ),
            )
        final_event = self._final_event(summary_steps)
        final_event["content"]["artifacts"] = context.index()
        yield await self._buf(request_id, final_event)
        await self.state_store.set_status(
            request_id, PipelineStatus.FAILED if aborted else PipelineStatus.DONE
        )
        if persist_history:
            self._schedule_persist_summary(
                token,
                chat_id,
                summary_steps,
                scenario_id,
                table_parts=artifact_parts(context),
            )

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
        urban_mcp_client: "UrbanMcpClient | None",
        token: str,
        model: str,
        temperature: float,
        scenario_id: int | None,
        input_artifacts: dict | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if (
            step.agent == OrchestratorAgent.PROVISION
            and input_artifacts
            and self.variant_provision_service
        ):
            return self.variant_provision_service.run(
                token=token,
                user_query=user_query,
                scenario_id=scenario_id,
                model=model,
                temperature=temperature,
                request_id=step_request_id,
                input_artifacts=input_artifacts,
                urban_mcp_client=urban_mcp_client,
            )
        if step.agent in self.planning_services:
            return self.planning_services[step.agent].run(
                token=token,
                user_query=user_query,
                scenario_id=scenario_id,
                model=model,
                temperature=temperature,
                request_id=step_request_id,
                input_artifacts=input_artifacts,
                urban_mcp_client=urban_mcp_client,
            )
        if step.agent == OrchestratorAgent.COMPLIANCE:
            if scenario_id is None:
                raise ValueError("compliance step requires scenario_id")
            from src.agents.services.planning.variant_compliance import (
                contains_variant,
                run_variant_compliance,
            )

            if input_artifacts and contains_variant(input_artifacts):
                return run_variant_compliance(
                    self.restriction_service,
                    self.llm_client,
                    input_artifacts,
                    mcp_client=idu_mcp_client,
                    normgraph_mcp_client=normgraph_mcp_client,
                    token=token,
                    temperature=temperature,
                    model=model,
                    user_query=user_query,
                    scenario_id=scenario_id,
                    request_id=step_request_id,
                    persist_history=False,
                )
            return self.restriction_service.run_compliance_pipeline(
                mcp_client=idu_mcp_client,
                normgraph_mcp_client=normgraph_mcp_client,
                token=token,
                temperature=temperature,
                model=model,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        if step.agent == OrchestratorAgent.RESTRICTION:
            if scenario_id is None:
                raise ValueError("restriction step requires scenario_id")
            return self.restriction_service.run_restriction_execution_pipline(
                mcp_client=idu_mcp_client,
                token=token,
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
                token=token,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
            )
        if step.agent == OrchestratorAgent.SCENARIO_DATA:
            if urban_mcp_client is None or self.scenario_data_service is None:
                raise ValueError("scenario_data step requires URBAN_MCP_SERVER")
            return self.scenario_data_service.run_scenario_data_pipeline(
                urban_mcp_client=urban_mcp_client,
                token=token,
                model=model,
                temperature=temperature,
                user_query=user_query,
                scenario_id=scenario_id,
                request_id=step_request_id,
                persist_history=False,
                **(
                    {"entity_selection": step.entity_selection}
                    if step.entity_selection
                    else {}
                ),
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
        raise ValueError(f"Unknown orchestrator agent: {step.agent}")

    # ------------------------------------------------------------------
    # Text digest between steps
    # ------------------------------------------------------------------

    def _compose_step_query(
        self,
        step: OrchestratorStep,
        digests: list[tuple[OrchestratorStep, str]],
    ) -> str:
        if not digests or step.agent == OrchestratorAgent.SCENARIO_DATA:
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
        *,
        table_parts: list[TablePartRequest] | None = None,
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
            ]
            + list(table_parts or []),
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
        parts: list[TextPartRequest | TablePartRequest],
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
    def _table_part(item: dict[str, Any]) -> TablePartRequest | None:
        if item.get("type") != "table" or not isinstance(item.get("content"), dict):
            return None
        try:
            return TablePartRequest(
                kind="table", payload=TablePayload.model_validate(item["content"])
            )
        except Exception as exc:
            logger.warning(f"Orchestrator: could not persist table event: {exc}")
            return None

    @staticmethod
    def _log_persist_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.exception(f"Orchestrator: failed to persist answer: {exc}")

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    async def check_replay_owner(self, request_id: str, token: str) -> None:
        state = await self.state_store.get_state(request_id)
        if state and state.get("owner") != context_scope(token, "owner"):
            # Legacy unowned runs are deliberately not public replay caches.
            raise AgentsNotFound("Сохранённый запрос недоступен")

    async def _buf(self, request_id: str, event: dict) -> dict:
        """Persist the event for reconnect replay before returning it."""
        event = jsonable_encoder(event)
        await self.state_store.buffer_event(request_id, event)
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
