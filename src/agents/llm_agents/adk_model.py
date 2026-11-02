"""A Google ADK ``BaseLlm`` that sends its requests through gMART's LLM adapter.

ADK agents speak ``google.genai`` contents; the model servers gMART runs on
(vLLM, Ollama) speak the OpenAI chat API through :class:`BaseLlmAdapter`. This
model translates one ADK turn into one ``adapter.chat`` call and back:

* the system instruction becomes the leading ``system`` message;
* ``user``/``model`` contents become ``user``/``assistant`` messages, function
  calls become assistant ``tool_calls`` and function responses ``tool`` messages;
* the agent's function declarations become OpenAI ``tools``.

Going through the adapter keeps request priority, pacing and the ``think``
spelling identical to every other agent, and lets tests stand in for the adapter.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import ConfigDict, Field

from src.agents.llm_agents import wire

_FINISH_REASONS = {
    "stop": types.FinishReason.STOP,
    "tool_calls": types.FinishReason.STOP,
    "length": types.FinishReason.MAX_TOKENS,
    "content_filter": types.FinishReason.SAFETY,
}


class GmartAdkLlm(BaseLlm):
    """ADK model backed by a :class:`BaseLlmAdapter`.

    Attributes:
        llm_client: The adapter (or any object with its ``chat`` coroutine).
        request_kind: Stage key sent with every call (priority, logs).
        think: ``think=`` for the adapter; routing needs no reasoning trace.
        options: Adapter options (``temperature``, ``num_predict``…).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_client: Any = Field(exclude=True)
    request_kind: str = "orchestrator.route"
    think: bool | None = False
    options: dict[str, Any] = Field(default_factory=dict)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        call: dict[str, Any] = {
            "model": self.model,
            "messages": to_chat_messages(llm_request),
            "stream": False,
            "request_kind": self.request_kind,
        }
        if self.think is not None:
            call["think"] = self.think
        if self.options:
            call["options"] = dict(self.options)
        tools = to_chat_tools(llm_request)
        if tools:
            call["tools"] = tools
        reply = await self.llm_client.chat(**call)
        response = to_llm_response(reply)
        # A rough estimate (≈4 chars per token) for ADK's telemetry; the adapter
        # keeps the exact counts for pacing.
        response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=sum(
                len(str(m.get("content") or "")) for m in call["messages"]
            )
            // 4,
            candidates_token_count=len(wire.content_of(reply)) // 4,
        )
        yield response


# ---------------------------------------------------------------------- #
# ADK -> adapter
# ---------------------------------------------------------------------- #
def _instruction_text(instruction: Any) -> str:
    if instruction is None:
        return ""
    if isinstance(instruction, str):
        return instruction
    if isinstance(instruction, types.Content):
        return "".join(part.text or "" for part in instruction.parts or [])
    if isinstance(instruction, list):
        return "\n".join(_instruction_text(item) for item in instruction)
    text = getattr(instruction, "text", None)
    return text if isinstance(text, str) else str(instruction)


def to_chat_messages(llm_request: LlmRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = _instruction_text(llm_request.config.system_instruction)
    if system:
        messages.append({"role": "system", "content": system})
    for content in llm_request.contents:
        texts: list[str] = []
        calls: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for part in content.parts or []:
            if part.function_call is not None:
                call = part.function_call
                calls.append(
                    {
                        "id": call.id or f"call_{call.name}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.args or {}, ensure_ascii=False, default=str
                            ),
                        },
                    }
                )
            elif part.function_response is not None:
                response = part.function_response
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": response.id or f"call_{response.name}",
                        "content": json.dumps(
                            response.response or {}, ensure_ascii=False, default=str
                        ),
                    }
                )
            elif part.text and not part.thought:
                texts.append(part.text)
        if content.role == "model":
            if texts or calls:
                entry: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
                if calls:
                    entry["tool_calls"] = calls
                messages.append(entry)
        else:
            messages.extend(results)
            if texts:
                messages.append({"role": "user", "content": "\n".join(texts)})
    return messages


def _lower_types(schema: Any) -> Any:
    """genai schemas spell types in capitals (``OBJECT``); JSON Schema does not."""

    if isinstance(schema, dict):
        return {
            key: (
                value.lower()
                if key == "type" and isinstance(value, str)
                else _lower_types(value)
            )
            for key, value in schema.items()
        }
    if isinstance(schema, list):
        return [_lower_types(item) for item in schema]
    return schema


def _parameters(declaration: types.FunctionDeclaration) -> dict[str, Any]:
    if declaration.parameters_json_schema:
        parameters = _lower_types(dict(declaration.parameters_json_schema))
    elif declaration.parameters is not None:
        parameters = _lower_types(
            declaration.parameters.model_dump(mode="json", exclude_none=True)
        )
    else:
        parameters = {}
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    return parameters


def to_chat_tools(llm_request: LlmRequest) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in llm_request.config.tools or []:
        for declaration in getattr(tool, "function_declarations", None) or []:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": _parameters(declaration),
                    },
                }
            )
    return tools


# ---------------------------------------------------------------------- #
# adapter -> ADK
# ---------------------------------------------------------------------- #
def to_llm_response(reply: Any) -> LlmResponse:
    parts: list[types.Part] = []
    content = wire.content_of(reply)
    if content:
        parts.append(types.Part(text=content))
    for call in wire.tool_calls_of(reply):
        try:
            args = json.loads(call["arguments"] or "{}")
        except ValueError:
            args = {}
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    id=call["id"],
                    name=call["name"],
                    args=args if isinstance(args, dict) else {},
                )
            )
        )
    done_reason = wire.done_reason_of(reply)
    return LlmResponse(
        content=types.Content(role="model", parts=parts) if parts else None,
        finish_reason=_FINISH_REASONS.get(done_reason or ""),
        custom_metadata={"done_reason": done_reason},
        turn_complete=True,
    )
