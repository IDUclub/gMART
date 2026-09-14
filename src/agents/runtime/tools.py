"""Execute already selected tools and specialist workflows through the SDK.

The planner has already chosen the operation. Re-asking inference to choose it
would change a validated plan. PlannedCallModel delivers that one call to Runner;
SDK owns invocation and cancellation, and stop_on_first_tool prevents replanning.
Arguments, tokens, GeoJSON and native results live in local request context, never
in model history. Public MCP names and arguments are still recorded by services.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from openai.types.responses import ResponseFunctionToolCall

from agents import Agent, Model, RunContextWrapper, Runner, Usage, function_tool
from agents.items import ModelResponse
from src.agents.runtime.budget import current_budget
from src.agents.runtime.runner import run_config


class PlannedCallModel(Model):
    """Deterministic SDK model for one validated operation, with no network I/O."""

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        **kwargs,
    ):
        if len(tools) != 1 or handoffs:
            raise ValueError("A planned operation must expose exactly one tool")
        return ModelResponse(
            output=[
                ResponseFunctionToolCall(
                    type="function_call",
                    name=tools[0].name,
                    arguments="{}",
                    call_id=f"call_{uuid4().hex}",
                )
            ],
            usage=Usage(),
            response_id=None,
        )

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError("Planned operations stream their domain events")
        yield  # pragma: no cover


@dataclass
class OperationContext:
    operation: Callable[[], Awaitable[Any]]
    result: Any = None
    error: Exception | None = None


async def execute_planned(name: str, operation: Callable[[], Awaitable[Any]]) -> Any:
    """Invoke once; exceptions propagate to the existing retry/checkpoint boundary."""

    budget = current_budget.get()
    if budget is not None and name.startswith(("mcp.", "urban.")):
        budget.tool()

    @function_tool(name_override="execute", failure_error_function=None)
    async def execute(context: RunContextWrapper[OperationContext]) -> str:
        """Выполнить выбранную и проверенную операцию текущего шага."""
        try:
            context.context.result = await context.context.operation()
        except Exception as exc:
            # SDK wraps tool exceptions in UserError. Keep the original typed
            # domain exception for token refresh, REST mapping and checkpoints.
            context.context.error = exc
            return "failed"
        return "completed"

    context = OperationContext(operation)
    agent = Agent(
        name=name,
        model=PlannedCallModel(),
        tools=[execute],
        tool_use_behavior="stop_on_first_tool",
    )
    await Runner.run(
        agent, [], context=context, max_turns=1, run_config=run_config(name)
    )
    if context.error is not None:
        raise context.error
    return context.result


async def stream_planned(name: str, events: AsyncIterator[dict]) -> AsyncIterator[dict]:
    """Delegate a fixed specialist step while preserving SSE envelopes/backpressure."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    finished = object()

    async def operation():
        async with aclosing(events):
            async for event in events:
                acknowledged = asyncio.Event()
                await queue.put((event, acknowledged))
                # Do not start the next domain operation until the caller has
                # processed this event (it may stop on clarification/failure).
                await acknowledged.wait()

    async def run():
        try:
            await execute_planned(name, operation)
        finally:
            # A cancelled consumer no longer drains this queue.
            if not asyncio.current_task().cancelling():
                await queue.put(finished)

    task = asyncio.create_task(run(), name=f"agent:{name}")
    try:
        while True:
            item = await queue.get()
            if item is finished:
                await task
                return
            event, acknowledged = item
            yield event
            acknowledged.set()
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
