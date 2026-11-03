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
* the steps of one turn run in order, each receiving the previous digests;
  only a turn made solely of norms/documents runs concurrently, its steps seeing
  the digests that existed before it — exactly as the planner did. Concurrent
  steps still stream one after another, in step order;
* a suspended step or one that needs the user's answer ends the run: later
  steps are skipped, concurrent ones cancelled. A failed step does not, as with
  the planner.

Steps the agent adds after seeing results simply continue the numbering. Every
call gets an id of the orchestrator's own: servers that send none (Ollama) or
repeat them would otherwise mix up the steps.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
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
        """The server cannot call tools at all, not just this one request.

        A 400 is also what a too long context or a malformed request gets, so a
        400/422 counts only when the server says it is about tools.
        """
        if self.status_code == 501:
            return True
        return self.status_code in _REJECTED_STATUSES and bool(
            _TOOLS_UNSUPPORTED.search(str(self))
        )


# Answers meaning "this server does not do tool calling" — vLLM without
# --enable-auto-tool-choice: '"auto" tool choice requires
# --enable-auto-tool-choice and --tool-call-parser to be set' (400); Ollama: 'model
# does not support tools' (400). Remembered for the process, so the requests that
# follow go to the planner without a failing call first.
_REJECTED_STATUSES = {400, 422}
_TOOLS_UNSUPPORTED = re.compile(
    r"tool[ _-]?choice"
    r"|tool[ _-]?call[ _-]?parser"
    r"|support(?:s|ed)?\s+(?:for\s+)?(?:tools|tool[ _-]?call|function[ _-]?call)"
    r"|(?:tools?|(?:tool|function)[ _-]?call(?:ing|s)?)\s+(?:are|is)\s+not\s+supported",
    re.IGNORECASE,
)
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
        self._discarded: set[int] = set()

    def put(self, event: dict[str, Any]) -> None:
        self.queue.put_nowait(event)

    def open(self, step: int) -> None:
        if step not in self._buffers and step not in self._discarded:
            insort(self._pending, step)
            self._buffers[step] = []

    def emit(self, step: int, event: dict[str, Any]) -> None:
        if step in self._discarded:
            return
        if self._pending and self._pending[0] == step:
            self.queue.put_nowait(event)
        else:
            self._buffers[step].append(event)

    def discard(self, step: int) -> None:
        """Drop a step that will not be reported: its held events never go out."""

        self._discarded.add(step)
        self._buffers.pop(step, None)
        if step in self._pending:
            head = self._pending[0] == step
            self._pending.remove(step)
            if head:
                self._advance()

    def close(self, step: int) -> None:
        if step in self._discarded:
            return
        self._finished.add(step)
        self._advance()

    def _advance(self) -> None:
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
class StepSlot:
    """Where one tool call stands among the steps.

    ``wait_for``: the step that must finish first (steps of a turn run in order).
    ``context``: for concurrent steps, how many digests existed when their turn
    was planned — they all see exactly those, whichever finishes first.
    """

    number: int
    wait_for: int | None = None
    context: int | None = None


@dataclass
class AdkRunState:
    summaries: dict[int, dict[str, Any]] = field(default_factory=dict)
    digests: list[tuple[OrchestratorStep, str]] = field(default_factory=list)
    step_tables: dict[int, list[Any]] = field(default_factory=dict)
    step_files: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    clarification: str | None = None
    final_text: str = ""
    plan: OrchestratorPlan | None = None
    aborted: bool = False
    slots: dict[str, StepSlot] = field(default_factory=dict)
    finished: dict[int, asyncio.Event] = field(default_factory=dict)
    running: dict[int, asyncio.Task] = field(default_factory=dict)
    # Steps after one that waits for the user: reported as skipped.
    cancelled: set[int] = field(default_factory=set)
    next_number: int = 1
    steps_started: int = 0

    def ordered_summary(self) -> list[dict[str, Any]]:
        return [self.summaries[number] for number in sorted(self.summaries)]

    def _reported(self, per_step: dict[int, list[Any]]) -> list[Any]:
        return [
            item
            for number in sorted(per_step)
            if number not in self.cancelled
            for item in per_step[number]
        ]

    @property
    def table_parts(self) -> list[Any]:
        return self._reported(self.step_tables)

    @property
    def file_events(self) -> list[dict[str, Any]]:
        return self._reported(self.step_files)


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
        if self.state.plan is None:
            plan = self._first_plan(calls)
            if plan.mode == OrchestratorPlanMode.EXECUTE:
                parts = self._schedule([(step.agent, step.task) for step in plan.steps])
                self.state.plan = plan
                self.bus.put(self.service._plan_event(plan))
            else:
                parts = [_clarify_call(plan.clarification_question or "")]
            llm_response.content = types.Content(role="model", parts=parts)
            return llm_response
        # Later turns: the calls of tools the agent has, numbered on.
        texts = [part for part in content.parts if part.function_call is None]
        requested: list[tuple[OrchestratorAgent, str]] = []
        clarify: list[types.Part] = []
        known = {entry.key for entry in self.agents}
        for call in calls:
            args = call.args or {}
            if call.name == CLARIFY_TOOL:
                clarify.append(_clarify_call(str(args.get("question") or "")))
                continue
            agent = _agent_of(call.name)
            if agent in known:
                requested.append((agent, str(args.get("task") or "")))
        llm_response.content = types.Content(
            role="model", parts=texts + self._schedule(requested) + clarify
        )
        return llm_response

    def _schedule(
        self, requested: list[tuple[OrchestratorAgent, str]]
    ) -> list[types.Part]:
        """Number one turn's agent calls and decide how they run.

        They run one after another, each seeing the earlier digests, unless
        every one of them is norms/documents: those start together and see the
        digests from before the turn (none in the first plan), as the planner's
        concurrent plan did. Every call gets an id of its own.
        """

        valid = [(agent, task) for agent, task in requested if task.strip()]
        concurrent = len(valid) > 1 and all(
            agent in _INDEPENDENT_AGENTS for agent, _ in valid
        )
        context = len(self.state.digests) if concurrent else None
        previous: int | None = None
        parts = []
        for agent, task in requested:
            call_id = _new_call_id()
            if task.strip():
                number = self.state.next_number
                self.state.next_number += 1
                self.state.slots[call_id] = StepSlot(
                    number, None if concurrent else previous, context
                )
                self.state.finished[number] = asyncio.Event()
                previous = number
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=call_id, name=agent_tool_name(agent), args={"task": task}
                    )
                )
            )
        return parts

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

    def _slot_for(self, tool_context: ToolContext) -> StepSlot:
        call_id = tool_context.function_call_id or ""
        slot = self.state.slots.get(call_id)
        if slot is None:
            # A call the orchestrator did not number (should not happen): next.
            slot = StepSlot(self.state.next_number)
            self.state.next_number += 1
            self.state.slots[call_id or _new_call_id()] = slot
        return slot

    async def _run_step(
        self, agent: OrchestratorAgent, task: str, tool_context: ToolContext
    ) -> dict[str, Any]:
        try:
            step = OrchestratorStep(agent=agent, task=task)
        except ValidationError:
            return {"status": "failed", "summary": "Пустая формулировка task."}
        slot = self._slot_for(tool_context)
        number = slot.number
        try:
            if number > MAX_PLAN_STEPS:
                tool_context.actions.skip_summarization = True
                return {"status": "skipped", "summary": _STEP_LIMIT_NOTE}
            if slot.wait_for is not None:
                previous = self.state.finished.get(slot.wait_for)
                if previous is not None:
                    await previous.wait()
            return await self._execute(slot, step, tool_context)
        finally:
            self.state.finished.setdefault(number, asyncio.Event()).set()

    def _skip(
        self, number: int, step: OrchestratorStep, tool_context: ToolContext
    ) -> dict[str, Any]:
        self.state.summaries[number] = self.service._summary_step(
            number, step, "skipped", ""
        )
        tool_context.actions.skip_summarization = True
        return {"status": "skipped", "summary": ""}

    def _abort_after(self, number: int) -> None:
        """Step ``number`` waits for the user: nothing after it is reported.

        Later steps that have not started are skipped when their turn comes; a
        concurrent one already running is cancelled, and one that already
        finished is reported as skipped — its held events never go out — just
        as the planner discarded its run-ahead steps.
        """

        self.state.aborted = True
        for later in range(number + 1, self.state.next_number):
            self.state.cancelled.add(later)
            self.bus.discard(later)
            task = self.state.running.get(later)
            if task is not None:
                task.cancel()
            summary = self.state.summaries.get(later)
            if summary is not None:
                self.state.summaries[later] = {
                    **summary,
                    "status": "skipped",
                    "summary": "",
                }

    async def _execute(
        self, slot: StepSlot, step: OrchestratorStep, tool_context: ToolContext
    ) -> dict[str, Any]:
        from src.agents.services.orchestrator.orchestrator_service import StepOutcome

        service = self.service
        number = slot.number
        if self.state.aborted or number in self.state.cancelled:
            return self._skip(number, step, tool_context)
        self.state.steps_started += 1
        digests = (
            self.state.digests
            if slot.context is None
            else self.state.digests[: slot.context]
        )
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
        files = self.state.step_files.setdefault(number, [])
        self.bus.open(number)

        async def stream() -> None:
            try:
                async for event in service._execute_step(
                    number,
                    step,
                    step_request_id,
                    effective_query,
                    make_pipeline,
                    outcome,
                    files,
                ):
                    self.bus.emit(number, event)
                if outcome.status not in {"completed", "failed"}:
                    # Waiting for the user: later steps would run on an
                    # unfinished premise. Decided before this step's close
                    # releases the next step's held events.
                    self._abort_after(number)
            finally:
                self.bus.close(number)

        # Its own task, so a clarification of an earlier concurrent step can
        # cancel it without cancelling ADK's tool call.
        running = asyncio.create_task(stream())
        self.state.running[number] = running
        try:
            await asyncio.wait([running])
        except asyncio.CancelledError:
            running.cancel()
            raise
        finally:
            self.state.running.pop(number, None)
        if running.cancelled() or number in self.state.cancelled:
            return self._skip(number, step, tool_context)
        running.result()
        self.state.summaries[number] = service._summary_step(
            number, step, outcome.status, outcome.digest
        )
        if outcome.status == "completed":
            self.state.digests.append((step, outcome.digest))
            self.state.step_tables[number] = list(outcome.tables)
        elif outcome.status == "failed":
            self.state.digests.append((step, _FAILED_STEP_CONTEXT))
        else:
            tool_context.actions.skip_summarization = True
        return {"status": outcome.status, "summary": outcome.digest}


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _clarify_call(question: str) -> types.Part:
    return types.Part(
        function_call=types.FunctionCall(
            id=_new_call_id(), name=CLARIFY_TOOL, args={"question": question}
        )
    )
