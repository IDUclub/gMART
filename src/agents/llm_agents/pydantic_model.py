"""A pydantic-ai ``Model`` that sends its requests through gMART's LLM adapter.

pydantic-ai owns the agent side of a step: the message history, the typed output,
its validation and the retry feedback when the model's answer does not validate.
The wire side stays with :class:`BaseLlmAdapter`, which already speaks to vLLM and
Ollama with everything those servers need (request priority, deadline pacing, the
``think`` spelling per server, a bounded retry of truncated JSON). Going through
the adapter, rather than through pydantic-ai's own OpenAI client, keeps that
behaviour in one place and keeps every test double and the benchmark capture —
all of which stand in for the adapter — working unchanged.

The model is built on :class:`FunctionModel`, whose request plumbing (settings and
parameter preparation, streamed parts) is exactly what a model backed by a local
callable needs; the two callables below translate pydantic-ai messages to the
adapter's chat messages and back.

gMART-specific knobs travel in the run's model settings under ``gmart_*`` keys
(see :class:`GmartModelSettings`); generic pydantic-ai settings (``temperature``,
``max_tokens``, ``top_p``) are mapped onto the adapter's ``options``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings

from src.agents.llm_agents import wire
from src.agents.model_clients.llm_base import BaseLlmAdapter

# Generic pydantic-ai settings and the adapter option each one maps to.
_OPTION_SETTINGS = {
    "temperature": "temperature",
    "max_tokens": "num_predict",
    "top_p": "top_p",
}
# Server finish reasons pydantic-ai understands; anything else stays in
# ``provider_details["done_reason"]`` only (see ``_finish_reason``).
_FINISH_REASONS = {"stop": "stop", "tool_calls": "tool_call", "tool_call": "tool_call"}


class GmartModelSettings(ModelSettings, total=False):
    """Model settings carrying the adapter's own call arguments."""

    gmart_request_kind: str | None
    """Stage key: picks the request priority and labels logs/benchmarks."""
    gmart_think: bool | None
    """``think=`` for the adapter (reasoning on/off)."""
    gmart_options: dict[str, Any] | None
    """Adapter ``options`` (``temperature``, ``num_predict``, ``num_ctx``, …), sent as given."""
    gmart_format: Any
    """Adapter ``format``: a JSON schema, ``"json"``, or ``None`` to derive it from the output type."""
    gmart_extra: dict[str, Any] | None
    """Further adapter keywords (``reasoning_effort``, ``keep_alive``, …)."""


class GmartChatModel(FunctionModel):
    """pydantic-ai model backed by a :class:`BaseLlmAdapter` and a model name.

    One instance serves one agent run: it remembers the last server reply
    (``last_done_reason``, ``last_thinking``) so the stage can report them, which a
    shared instance could not do safely under concurrency.
    """

    def __init__(self, llm_client: BaseLlmAdapter, model_name: str) -> None:
        self.llm_client = llm_client
        self.served_model = model_name
        self.last_done_reason: str | None = None
        self.last_thinking: str | None = None
        self.last_text = ""
        self.calls = 0
        super().__init__(
            self._request,
            stream_function=self._stream,
            model_name=model_name,
            profile=ModelProfile(
                supports_json_schema_output=True,
                supports_json_object_output=True,
                supports_tools=True,
                default_structured_output_mode="native",
            ),
        )

    @property
    def system(self) -> str:
        return "gmart"

    # ------------------------------------------------------------------ #
    # pydantic-ai -> adapter
    # ------------------------------------------------------------------ #
    def _chat_call(
        self, messages: list[ModelMessage], info: AgentInfo, *, stream: bool
    ) -> dict[str, Any]:
        settings: dict[str, Any] = dict(info.model_settings or {})
        params = info.model_request_parameters
        call: dict[str, Any] = {
            "model": self.served_model,
            "messages": to_chat_messages(messages, info.instructions),
            "stream": stream,
        }
        options = dict(settings.get("gmart_options") or {})
        for setting, option in _OPTION_SETTINGS.items():
            if setting in settings and option not in options:
                options[option] = settings[setting]
        if options:
            call["options"] = options
        if settings.get("gmart_think") is not None:
            call["think"] = settings["gmart_think"]
        if settings.get("gmart_request_kind"):
            call["request_kind"] = settings["gmart_request_kind"]
        schema = settings.get("gmart_format")
        if schema is None and params.output_mode == "native" and params.output_object:
            schema = params.output_object.json_schema
        if schema is not None:
            call["format"] = schema
        tools = [_tool_schema(tool) for tool in params.function_tools]
        if tools:
            call["tools"] = tools
        call.update(settings.get("gmart_extra") or {})
        return call

    async def _request(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        call = self._chat_call(messages, info, stream=False)
        self.calls += 1
        reply = await self.llm_client.chat(**call)
        content = wire.content_of(reply)
        self.last_done_reason = wire.done_reason_of(reply)
        self.last_thinking = wire.thinking_of(reply)
        params = info.model_request_parameters
        if params.output_mode == "native" or call.get("format") is not None:
            content = wire.strip_json_fence(content)
        self.last_text = content
        parts: list[ModelResponsePart] = []
        if self.last_thinking:
            parts.append(ThinkingPart(content=self.last_thinking))
        if content:
            parts.append(TextPart(content=content))
        for tool_call in wire.tool_calls_of(reply):
            parts.append(
                ToolCallPart(
                    tool_name=tool_call["name"],
                    args=tool_call["arguments"],
                    tool_call_id=tool_call["id"],
                )
            )
        if not any(isinstance(part, (TextPart, ToolCallPart)) for part in parts):
            # Keep an (empty) text part: the stage reports an empty answer itself
            # instead of pydantic-ai turning it into a "token limit" exception.
            parts.append(TextPart(content=""))
        return ModelResponse(
            parts=parts,
            finish_reason=_finish_reason(self.last_done_reason),
            provider_details={"done_reason": self.last_done_reason},
        )

    async def _stream(
        self, messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        call = self._chat_call(messages, info, stream=True)
        self.calls += 1
        thinking: list[str] = []
        yielded = False
        async for part in await self.llm_client.chat(**call):
            if piece := wire.thinking_of(part):
                thinking.append(piece)
            if wire.field(part, "done"):
                self.last_done_reason = wire.done_reason_of(part)
            if content := wire.content_of(part):
                yielded = True
                yield content
        self.last_thinking = "".join(thinking) or None
        if not yielded:
            # FunctionModel requires at least one item from a stream.
            yield ""


def _finish_reason(done_reason: str | None) -> Any:
    # "length", "repetition", "incomplete" are deliberately not mapped: pydantic-ai
    # raises on an empty "length" reply, and the stages decide what a truncated
    # answer means for their pipeline (most retry with a larger budget).
    return _FINISH_REASONS.get(done_reason or "")


def _tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters_json_schema,
        },
    }


def _user_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(str(item) for item in content if isinstance(item, str))


def to_chat_messages(
    messages: list[ModelMessage], instructions: str | None = None
) -> list[dict[str, Any]]:
    """pydantic-ai history -> the adapter's ``{"role", "content"}`` messages.

    The order is kept exactly, so a stage built from a chat-message list sends
    that same list (see :func:`from_chat_messages`).
    """

    result: list[dict[str, Any]] = []
    if instructions:
        result.append({"role": "system", "content": instructions})
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    result.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    result.append({"role": "user", "content": _user_text(part.content)})
                elif isinstance(part, ToolReturnPart):
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": part.tool_call_id,
                            "content": part.model_response_str(),
                        }
                    )
                elif isinstance(part, RetryPromptPart):
                    if part.tool_name:
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": part.tool_call_id,
                                "content": part.model_response(),
                            }
                        )
                    else:
                        result.append({"role": "user", "content": part.model_response()})
                # Bookkeeping parts (instruction/availability deltas…) carry nothing
                # for a chat-completions server.
        elif isinstance(message, ModelResponse):
            text = "".join(p.content for p in message.parts if isinstance(p, TextPart))
            calls = [
                {
                    "id": p.tool_call_id,
                    "type": "function",
                    "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
                }
                for p in message.parts
                if isinstance(p, ToolCallPart)
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                entry["tool_calls"] = calls
            result.append(entry)
    return result


def from_chat_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[ModelMessage], str | None]:
    """Adapter chat messages -> (pydantic-ai history, current user prompt).

    A trailing user message becomes the run's prompt; everything before it the
    history, each message as its own part so the order survives the round trip.
    """

    items = list(messages)
    prompt: str | None = None
    if items and items[-1].get("role") == "user":
        prompt = str(items.pop().get("content") or "")
    history: list[ModelMessage] = []
    for item in items:
        role = item.get("role")
        content = item.get("content")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        if role == "system":
            history.append(ModelRequest(parts=[SystemPromptPart(content=content)]))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            history.append(ModelRequest(parts=[UserPromptPart(content=content)]))
    return history, prompt
