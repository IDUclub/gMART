"""Harness: what every model call in a gMART workflow goes through.

Two layers, both standard ADK extension points:

* model callbacks, attached per agent, decide what a step sends: which
  model and temperature the request asked for (:func:`request_params`) and
  which conversation the model sees (:func:`conversation`);
* :class:`AuditPlugin`, registered once on the ``App``, observes every model
  call and every node of every run — the place for usage accounting, rate
  limiting, redaction or tracing, instead of code spread over the services.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.genai import types

# Session-state keys every gMART workflow reads.
MODEL = "model"
TEMPERATURE = "temperature"
USER_QUERY = "user_query"
HISTORY = "history"


def request_params(
    temperature: Callable[[float], float] | None = None,
) -> Callable[[CallbackContext, LlmRequest], None]:
    """Sets the requested model and, optionally, a temperature derived from
    the requested one (``None`` keeps the agent's fixed temperature)."""

    def apply(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        state = callback_context.state
        if state.get(MODEL):
            llm_request.model = state[MODEL]
        if temperature is not None and state.get(TEMPERATURE) is not None:
            llm_request.config.temperature = temperature(state[TEMPERATURE])

    return apply


def conversation(
    history: bool = True, query: bool = True
) -> Callable[[CallbackContext, LlmRequest], None]:
    """Replaces what the model sees with the chat history and the user query.

    A workflow node otherwise sends its predecessor's output as the user
    turn; gMART steps answer the user, so the user turn is the query itself.
    """

    def apply(callback_context: CallbackContext, llm_request: LlmRequest) -> None:
        state = callback_context.state
        contents: list[types.Content] = []
        if history:
            for message in state.get(HISTORY) or []:
                role = "model" if message["role"] == "assistant" else "user"
                contents.append(
                    types.Content(
                        role=role, parts=[types.Part(text=message["content"])]
                    )
                )
        if query and state.get(USER_QUERY):
            contents.append(
                types.Content(role="user", parts=[types.Part(text=state[USER_QUERY])])
            )
        llm_request.contents = contents

    return apply


@dataclass
class ModelCall:
    agent: str
    model: str | None
    messages: int
    seconds: float = 0.0
    output_chars: int = 0


@dataclass
class RunAudit:
    model_calls: list[ModelCall] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)


class AuditPlugin(BasePlugin):
    """Records, per invocation, the model calls and the nodes that ran."""

    def __init__(self, name: str = "gmart_audit") -> None:
        super().__init__(name=name)
        self.runs: dict[str, RunAudit] = defaultdict(RunAudit)
        self._started: dict[tuple[str, str], float] = {}

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        key = (callback_context.invocation_id, callback_context.agent_name)
        self._started[key] = time.perf_counter()
        system = (
            1 if llm_request.config and llm_request.config.system_instruction else 0
        )
        self.runs[callback_context.invocation_id].model_calls.append(
            ModelCall(
                agent=callback_context.agent_name,
                model=llm_request.model,
                messages=system + len(llm_request.contents or []),
            )
        )
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        if llm_response.partial:
            return None
        key = (callback_context.invocation_id, callback_context.agent_name)
        call = self.runs[callback_context.invocation_id].model_calls[-1]
        call.seconds = time.perf_counter() - self._started.pop(key, time.perf_counter())
        parts = llm_response.content.parts if llm_response.content else []
        call.output_chars = sum(len(part.text or "") for part in parts or [])
        return None

    async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
        node = event.node_info.name if event.node_info else None
        nodes = self.runs[invocation_context.invocation_id].nodes
        if node and (not nodes or nodes[-1] != node):
            nodes.append(node)
        return None
