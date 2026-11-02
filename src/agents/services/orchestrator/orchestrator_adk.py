"""The autonomous orchestrator: a Google ADK agent that uses the gMART agents as tools.

The JSON planner decides the whole plan in one model call. This orchestrator is
an ADK ``LlmAgent`` instead: every sub-agent (restriction, compliance, provision,
scenario data, documents, norms, PZZ, project files) is one of its tools, and it
keeps calling them — reading each step's digest — until the request is answered
or it has to ask the user (``ask_clarification``). The sub-agents themselves stay
deterministic pipelines.

The SSE contract of ``/orchestrator`` does not change:

* the tool calls of the agent's first turn are the plan: they pass the planner's
  deterministic corrections (:meth:`OrchestratorPlanBuilder.refine_plan`) and are
  announced as the ``plan`` event;
* every tool call is one step with ``step_started`` / ``step_event`` /
  ``step_finished``, run by the same code as a planned step;
* steps of the first plan run in order, each receiving the previous digests, and
  only a plan made solely of norms/documents runs concurrently — exactly as
  before. Concurrent steps still stream one after another, in step order;
* a suspended step or one that needs the user's answer ends the run; a failed
  step does not, as with the planner.

Steps the agent adds after seeing results simply continue the numbering.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from bisect import insort
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types
from loguru import logger
from pydantic import ValidationError

from src.agents.llm_agents.adk_model import GmartAdkLlm
from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.services.orchestrator.orchestrator_catalog import AgentCatalogEntry
from src.agents.services.service_entities.orchestrator_plan import (
    MAX_PLAN_STEPS,
    OrchestratorAgent,
    OrchestratorPlan,
    OrchestratorPlanMode,
    OrchestratorStep,
)

if TYPE_CHECKING:
    from src.agents.services.orchestrator.orchestrator_service import (
        OrchestratorService,
    )

ENGINE_ENV = "ORCHESTRATOR_ENGINE"
ENGINE_ADK = "adk"
ENGINE_PLANNER = "planner"

CLARIFY_TOOL = "ask_clarification"
_AGENT_TOOL_PREFIX = "run_"
_APP_NAME = "gmart-orchestrator"
_USER_ID = "gmart"
# Agents that answer from their own sources and never need another step's digest.
_INDEPENDENT_AGENTS = {OrchestratorAgent.NORMS, OrchestratorAgent.DOCUMENTS}
_FAILED_STEP_CONTEXT = (
    "Шаг не выполнен, его результата нет. Не подставляй вместо него значения "
    "или предположения."
)
_STEP_LIMIT_NOTE = (
    f"Лимит шагов ({MAX_PLAN_STEPS}) исчерпан: этот шаг не выполнялся. "
    "Заверши работу ответом ГОТОВО."
)


def orchestrator_engine() -> str:
    """``ORCHESTRATOR_ENGINE``: ``adk`` (default) or ``planner`` (JSON plan)."""

    value = (os.getenv(ENGINE_ENV) or ENGINE_ADK).strip().lower()
    if value not in {ENGINE_ADK, ENGINE_PLANNER}:
        logger.warning(f"Unknown {ENGINE_ENV}={value!r}; using {ENGINE_ADK!r}")
        return ENGINE_ADK
    return value


def agent_tool_name(agent: OrchestratorAgent) -> str:
    return f"{_AGENT_TOOL_PREFIX}{agent.value}"


def _agent_of(tool_name: str | None) -> OrchestratorAgent | None:
    if not tool_name or not tool_name.startswith(_AGENT_TOOL_PREFIX):
        return None
    try:
        return OrchestratorAgent(tool_name.removeprefix(_AGENT_TOOL_PREFIX))
    except ValueError:
        return None


class AdkUnavailableError(Exception):
    """The model server could not run the tool-calling agent (before any step).

    The orchestrator then plans the request with the JSON planner instead, so a
    server started without tool-call support keeps working.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def tools_unsupported(self) -> bool:
        """The server rejected the request itself, not a transient failure."""
        return self.status_code in _UNSUPPORTED_STATUSES


# Answers meaning "this server does not do tool calling" (vLLM without
# --enable-auto-tool-choice answers 400). Remembered for the process, so the
# requests that follow go to the planner without a failing call first.
_UNSUPPORTED_STATUSES = {400, 422, 501}
_tools_unsupported = False


def adk_available() -> bool:
    return orchestrator_engine() == ENGINE_ADK and not _tools_unsupported


def mark_tools_unsupported(error: AdkUnavailableError) -> None:
    global _tools_unsupported
    if error.tools_unsupported and not _tools_unsupported:
        _tools_unsupported = True
        logger.warning(
            "The model server does not support tool calling; the orchestrator "
            "uses the JSON planner until restart"
        )


# ---------------------------------------------------------------------- #
# Ordered event bus
# ---------------------------------------------------------------------- #
_DONE = object()


class OrderedStepBus:
    """Streams step events in step order while steps may run concurrently.

    The lowest unfinished step streams live; a later step's events wait in its
    buffer until every earlier step has finished. Events outside any step
    (the plan) go out at once.
    """

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self._pending: list[int] = []  # registered steps not yet fully streamed
        self._buffers: dict[int, list[dict[str, Any]]] = {}
        self._finished: set[int] = set()

    def put(self, event: dict[str, Any]) -> None:
        self.queue.put_nowait(event)

    def open(self, step: int) -> None:
        if step not in self._buffers:
            insort(self._pending, step)
            self._buffers[step] = []

    def emit(self, step: int, event: dict[str, Any]) -> None:
        if self._pending and self._pending[0] == step:
            self.queue.put_nowait(event)
        else:
            self._buffers[step].append(event)

    def close(self, step: int) -> None:
        self._finished.add(step)
        while self._pending:
            head = self._pending[0]
            for event in self._buffers.pop(head, []):
                self.queue.put_nowait(event)
            if head not in self._finished:
                # The new head streams live from now on.
                self._buffers[head] = []
                return
            self._pending.pop(0)

    def done(self) -> None:
        self.queue.put_nowait(_DONE)


# ---------------------------------------------------------------------- #
# One orchestration run
# ---------------------------------------------------------------------- #
@dataclass
class AdkRunState:
    summary_steps: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    digests: list[tuple[OrchestratorStep, str]] = field(default_factory=list)
    table_parts: list[Any] = field(default_factory=list)
    file_events: list[dict[str, Any]] = field(default_factory=list)
    clarification: str | None = None
    final_text: str = ""
    plan: OrchestratorPlan | None = None
    aborted: bool = False
    call_numbers: dict[str, int] = field(default_factory=dict)
    finished: dict[int, asyncio.Event] = field(default_factory=dict)
    sequential: bool = True
    next_number: int = 1
    steps_started: int = 0

    def ordered_summary(self) -> list[dict[str, Any]]:
        return [
            summary for _, summary in sorted(self.summary_steps, key=lambda s: s[0])
        ]


class AdkOrchestration:
    """Runs one request through the ADK orchestrator agent.

    :meth:`events` yields the orchestrator's top-level SSE events (not yet
    buffered); afterwards :attr:`state` holds the steps' summaries, tables,
    files and a clarification question, if the agent asked one.
    """

    def __init__(
        self,
        service: OrchestratorService,
        *,
        model: str,
        user_query: str,
        history: list[dict],
        agents: list[AgentCatalogEntry],
        step_kwargs: dict[str, Any],
        scenario_id: int | None,
        pzz_inputs: Any = None,
        compliance_uploads: int = 0,
    ) -> None:
        self.service = service
        self.model = model
        self.user_query = user_query
        self.history = history
        self.agents = agents
        self.step_kwargs = step_kwargs
        self.state = AdkRunState()
        self.bus = OrderedStepBus()
        self.instruction = service.plan_builder.build_tool_instruction(
            agents, scenario_id, pzz_inputs, compliance_uploads
        )

    # ------------------------------------------------------------------ #
    def build_agent(self) -> LlmAgent:
        llm = GmartAdkLlm(
            model=self.model,
            llm_client=self.service.llm_client,
            request_kind="orchestrator.route",
            think=False,
            options={"temperature": 0, "num_predict": 2048},
        )
        tools = [FunctionTool(self._agent_tool(entry)) for entry in self.agents]
        tools.append(FunctionTool(self._clarify_tool()))
        return LlmAgent(
            name="orchestrator",
            description="Маршрутизатор запросов между агентами gMART.",
            model=llm,
            instruction=self.instruction,
            tools=tools,
            after_model_callback=self._after_model,
        )

    def _user_message(self) -> str:
        if not self.history:
            return self.user_query
        return json.dumps(
            {
                "completed_dialogue_context": self.history,
                "current_request": self.user_query,
            },
            ensure_ascii=False,
        )

    async def events(self) -> AsyncGenerator[dict[str, Any], None]:
        runner = InMemoryRunner(agent=self.build_agent(), app_name=_APP_NAME)
        session = await runner.session_service.create_session(
            app_name=_APP_NAME, user_id=_USER_ID, session_id=uuid.uuid4().hex
        )
        failure: list[BaseException] = []

        async def drive() -> None:
            try:
                async for event in runner.run_async(
                    user_id=_USER_ID,
                    session_id=session.id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=self._user_message())]
                    ),
                    run_config=RunConfig(max_llm_calls=MAX_PLAN_STEPS + 3),
                ):
                    if event.author == "orchestrator" and event.content:
                        text = "".join(
                            part.text or ""
                            for part in event.content.parts or []
                            if part.text and not part.thought
                        )
                        if text and not event.get_function_calls():
                            self.state.final_text = text.strip()
            except BaseException as exc:  # noqa: BLE001 - re-raised by the consumer
                failure.append(exc)
            finally:
                self.bus.done()

        task = asyncio.create_task(drive())
        yielded = False
        try:
            while True:
                event = await self.bus.queue.get()
                if event is _DONE:
                    break
                yielded = True
                yield event
        finally:
            if not task.done():
                task.cancel()
            try:
                await task
            except BaseException:  # noqa: BLE001 - cancellation of our own task
                pass
        if failure:
            exc = failure[0]
            if not self.state.steps_started:
                if isinstance(exc, LlmResponseError) and not yielded:
                    raise AdkUnavailableError(str(exc), exc.status_code) from exc
                raise exc
            # The steps' results are already streamed; a model failure while
            # deciding what to do next must not throw them away.
            if isinstance(exc, LlmCallsLimitExceededError):
                logger.warning("ADK orchestrator reached its model-call limit")
            else:
                logger.opt(exception=exc).error(
                    "ADK orchestrator failed after its steps; finishing with them"
                )
        if self.state.plan is None and self.state.clarification is None:
            # The model answered in words without calling anything: that answer is
            # a question (or an explanation) for the user, as a clarification.
            text = self.state.final_text
            self.state.clarification = (
                text
                if text and text.strip(" .!").upper() != "ГОТОВО"
                else self.service.plan_builder._clarification_text(self.agents)
            )

    # ------------------------------------------------------------------ #
    # first turn: the plan
    # ------------------------------------------------------------------ #
    def _after_model(self, callback_context: Any, llm_response: Any) -> Any:
        content = llm_response.content
        calls = [
            part.function_call
            for part in (content.parts if content else []) or []
            if part.function_call is not None
        ]
        if not calls:
            return None
        known = {agent_tool_name(entry.key) for entry in self.agents} | {CLARIFY_TOOL}
        if self.state.plan is not None:
            # Later turns: drop calls of tools the agent does not have.
            if all(call.name in known for call in calls):
                return None
            parts = [
                part
                for part in content.parts
                if part.function_call is None or part.function_call.name in known
            ]
            llm_response.content = types.Content(role="model", parts=parts)
            return llm_response
        plan = self._first_plan(calls)
        llm_response.content = types.Content(
            role="model", parts=self._plan_calls(plan, calls)
        )
        if plan.mode == OrchestratorPlanMode.EXECUTE:
            self.state.plan = plan
            self.state.sequential = not (
                len(plan.steps) > 1
                and all(step.agent in _INDEPENDENT_AGENTS for step in plan.steps)
            )
            self.bus.put(self.service._plan_event(plan))
        return llm_response

    def _first_plan(self, calls: list[types.FunctionCall]) -> OrchestratorPlan:
        steps: list[OrchestratorStep] = []
        question = None
        unknown = False
        for call in calls:
            args = call.args or {}
            if call.name == CLARIFY_TOOL:
                question = question or str(args.get("question") or "").strip()
                continue
            agent = _agent_of(call.name)
            try:
                steps.append(
                    OrchestratorStep(agent=agent, task=str(args.get("task") or ""))
                )
            except (ValidationError, ValueError):
                unknown = True
        if steps and not unknown:
            plan = OrchestratorPlan(mode=OrchestratorPlanMode.EXECUTE, steps=steps)
        else:
            plan = OrchestratorPlan(
                mode=OrchestratorPlanMode.NEEDS_CLARIFICATION,
                clarification_question=None if unknown else question,
            )
        return self.service.plan_builder.refine_plan(plan, self.agents, self.user_query)

    def _plan_calls(
        self, plan: OrchestratorPlan, calls: list[types.FunctionCall]
    ) -> list[types.Part]:
        """The first turn's calls, rewritten to the refined plan, numbered."""

        if plan.mode != OrchestratorPlanMode.EXECUTE:
            return [
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=CLARIFY_TOOL,
                        args={"question": plan.clarification_question or ""},
                    )
                )
            ]
        ids = [call.id for call in calls if _agent_of(call.name) is not None]
        parts = []
        for number, step in enumerate(plan.steps, start=1):
            call_id = (ids[number - 1] if number <= len(ids) else None) or (
                f"call_{uuid.uuid4().hex[:12]}"
            )
            self.state.call_numbers[call_id] = number
            self.state.finished[number] = asyncio.Event()
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call_id,
                        name=agent_tool_name(step.agent),
                        args={"task": step.task},
                    )
                )
            )
        self.state.next_number = len(plan.steps) + 1
        return parts

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _agent_tool(self, entry: AgentCatalogEntry):
        orchestration = self

        async def run_agent(task: str, tool_context: ToolContext) -> dict:
            return await orchestration._run_step(entry.key, task, tool_context)

        run_agent.__name__ = agent_tool_name(entry.key)
        run_agent.__doc__ = (
            f"{entry.title}. {entry.description}\n\n"
            "Args:\n"
            "    task: Самодостаточная формулировка подзадачи на русском со всеми "
            "нужными деталями: агент видит только её."
        )
        return run_agent

    def _clarify_tool(self):
        orchestration = self

        async def ask_clarification(question: str, tool_context: ToolContext) -> dict:
            """Задать пользователю уточняющий вопрос, когда запрос нельзя выполнить
            доступными агентами или он неоднозначен.

            Args:
                question: Вопрос пользователю: чего не хватает или что неясно.
            """
            orchestration.state.clarification = (
                question or ""
            ).strip() or orchestration.service.plan_builder._clarification_text(
                orchestration.agents
            )
            tool_context.actions.skip_summarization = True
            return {"status": "asked"}

        return ask_clarification

    def _number_for(self, tool_context: ToolContext) -> int:
        call_id = tool_context.function_call_id or ""
        if call_id in self.state.call_numbers:
            return self.state.call_numbers[call_id]
        number = self.state.next_number
        self.state.next_number += 1
        self.state.call_numbers[call_id] = number
        return number

    async def _run_step(
        self, agent: OrchestratorAgent, task: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        number = self._number_for(tool_context)
        if number > MAX_PLAN_STEPS:
            tool_context.actions.skip_summarization = True
            return {"status": "skipped", "summary": _STEP_LIMIT_NOTE}
        try:
            step = OrchestratorStep(agent=agent, task=task)
        except ValidationError:
            return {"status": "failed", "summary": "Пустая формулировка task."}
        previous = self.state.finished.get(number - 1)
        if self.state.sequential and previous is not None:
            # Planned steps run in order so each one gets the earlier digests.
            await previous.wait()
        try:
            return await self._execute(number, step, tool_context)
        finally:
            done = self.state.finished.setdefault(number, asyncio.Event())
            done.set()

    async def _execute(
        self, number: int, step: OrchestratorStep, tool_context: ToolContext
    ) -> dict[str, Any]:
        from src.agents.services.orchestrator.orchestrator_service import StepOutcome

        service = self.service
        if self.state.aborted:
            self.state.summary_steps.append(
                (number, service._summary_step(number, step, "skipped", ""))
            )
            tool_context.actions.skip_summarization = True
            return {"status": "skipped", "summary": ""}
        self.state.steps_started += 1
        digests = self.state.digests
        effective_query = service._compose_step_query(step, digests)
        step_request_id = service.state_store.new_request_id()
        make_pipeline = partial(
            service._build_step_pipeline,
            step,
            effective_query,
            step_request_id,
            context_note=service._digest_lines(digests) or None,
            **self.step_kwargs,
        )
        outcome = StepOutcome()
        self.bus.open(number)
        try:
            async for event in service._execute_step(
                number,
                step,
                step_request_id,
                effective_query,
                make_pipeline,
                outcome,
                self.state.file_events,
            ):
                self.bus.emit(number, event)
        finally:
            self.bus.close(number)
        self.state.summary_steps.append(
            (
                number,
                service._summary_step(number, step, outcome.status, outcome.digest),
            )
        )
        if outcome.status == "completed":
            digests.append((step, outcome.digest))
            self.state.table_parts.extend(outcome.tables)
        elif outcome.status == "failed":
            digests.append((step, _FAILED_STEP_CONTEXT))
        else:
            # Waiting for the user: later steps would run on an unfinished premise.
            self.state.aborted = True
            tool_context.actions.skip_summarization = True
        return {"status": outcome.status, "summary": outcome.digest}
