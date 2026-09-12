"""SDK Model bridge; backend-specific sampling and error mapping stay in adapters.

Each instance belongs to one run, so concurrent requests cannot share responses,
reasoning settings or stream termination state. No OpenAI-hosted model is selected.
"""

from __future__ import annotations

import time
from uuid import uuid4

from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)

from agents import Model, ModelBehaviorError, Usage
from agents.items import ModelResponse
from src.agents.model_clients.llm_base import (
    LlmChatResponse,
    LlmMessage,
    closing_stream,
)


def output_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid4().hex}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def response_usage(response):
    raw = (
        response.get("usage")
        if isinstance(response, dict)
        else getattr(response, "usage", None)
    )

    def field(name, default=0):
        return (
            raw.get(name, default)
            if isinstance(raw, dict)
            else getattr(raw, name, default)
        ) or 0

    prompt = field("prompt_tokens") or getattr(response, "prompt_eval_count", 0) or 0
    output = field("completion_tokens") or getattr(response, "eval_count", 0) or 0
    return Usage(
        requests=1,
        input_tokens=prompt,
        output_tokens=output,
        total_tokens=prompt + output,
    )


class BackendModel(Model):
    """One SDK model invocation over the configured native/OpenAI-compatible backend."""

    def __init__(self, backend, model: str, **settings):
        self.backend = backend
        self.model = model
        self.settings = settings
        self.response: LlmChatResponse | None = None

    def _call(self, instructions, input, output_schema):
        messages = (
            [{"role": "user", "content": input}]
            if isinstance(input, str)
            else [dict(item) for item in input]
        )
        if instructions:
            messages.insert(0, {"role": "system", "content": instructions})
        settings = dict(self.settings)
        if output_schema is not None and not settings.pop("unconstrained", False):
            settings["format"] = output_schema.json_schema()
        else:
            settings.pop("unconstrained", None)
        return dict(model=self.model, messages=messages, **settings)

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
        if tools or handoffs:
            raise ValueError(
                "BackendModel accepts model stages; use planned tools for execution"
            )
        call = self._call(system_instructions, input, output_schema)
        if call.pop("completion_mode", None) == "generate":
            messages = call.pop("messages")
            result = await self.backend.generate(
                **call, prompt=messages[0]["content"], stream=False
            )
            self.response = LlmChatResponse(
                model=self.model,
                message=LlmMessage(content=result.response),
                usage=getattr(result, "usage", None),
            )
        else:
            self.response = await self.backend.chat(**call, stream=False)
        if (
            output_schema is not None
            and not self.response["message"]["content"].strip()
        ):
            raise ModelBehaviorError("Empty structured model output")
        return ModelResponse(
            output=[output_message(self.response["message"]["content"])],
            usage=response_usage(self.response),
            response_id=None,
        )

    async def stream_response(
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
        if tools or handoffs:
            raise ValueError(
                "BackendModel accepts model stages; use planned tools for execution"
            )
        stream = await self.backend.chat(
            **self._call(system_instructions, input, output_schema), stream=True
        )
        text_parts = []
        item_id = f"msg_{uuid4().hex}"
        sequence = 0
        terminal = False
        usage = Usage()
        # All production adapters expose async generators; close them on cancellation.
        async with closing_stream(stream):
            async for part in stream:
                measured = response_usage(part)
                if measured.total_tokens:
                    usage = measured
                text = part.message.content or ""
                text_parts.append(text)
                terminal = (
                    terminal
                    or bool(getattr(part, "done", False))
                    or bool(getattr(part, "done_reason", None))
                )
                yield ResponseTextDeltaEvent(
                    type="response.output_text.delta",
                    delta=text,
                    content_index=0,
                    output_index=0,
                    item_id=item_id,
                    logprobs=[],
                    sequence_number=sequence,
                    gmart_chunk=part,
                )
                sequence += 1
        if not terminal:
            part = LlmChatResponse(
                model=self.model, done=True, done_reason="incomplete"
            )
            yield ResponseTextDeltaEvent(
                type="response.output_text.delta",
                delta="",
                content_index=0,
                output_index=0,
                item_id=item_id,
                logprobs=[],
                sequence_number=sequence,
                gmart_chunk=part,
            )
            sequence += 1
        self.response = LlmChatResponse(
            model=self.model,
            message=LlmMessage(content="".join(text_parts)),
            done=True,
            done_reason=getattr(part, "done_reason", None),
            usage={
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
        )
        # The SDK completion closes its run; the original provider termination
        # reason remains on every gMART chunk (length/incomplete are never masked).
        yield ResponseCompletedEvent(
            type="response.completed",
            sequence_number=sequence,
            response=Response(
                id=f"resp_{uuid4().hex}",
                created_at=time.time(),
                model=self.model,
                object="response",
                output=[output_message("".join(text_parts))],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
                status="completed",
                usage={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "input_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            ),
        )
