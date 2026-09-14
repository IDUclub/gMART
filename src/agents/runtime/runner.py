"""Named SDK agents, streaming and shared bounded structured-output repair.

Redis checkpoints, transport retries and domain validation are separate concerns.
There is no legacy execution switch and no remote tracing or implicit model default.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter

from agents import Agent, AgentOutputSchemaBase, ModelBehaviorError, RunConfig, Runner
from src.agents.model_clients.llm_base import LlmGenerateResponse
from src.agents.runtime.model import BackendModel


class IncompleteStructuredOutput(ValueError):
    """The provider did not finish the structured response."""


def run_config(name: str) -> RunConfig:
    return RunConfig(
        workflow_name=name, tracing_disabled=True, trace_include_sensitive_data=False
    )


def _agent(backend, model, name, settings, output_type=None):
    bridge = BackendModel(backend, model, **settings)
    return Agent(name=name, model=bridge, output_type=output_type), bridge


async def run_completion(
    backend,
    model: str,
    messages: list[dict] | None = None,
    *,
    agent_name: str,
    stream: bool = False,
    **settings,
):
    """Execute a named text/JSON stage; keep the service response/SSE contract."""
    agent, bridge = _agent(backend, model, agent_name, settings)
    if stream:
        return _stream(agent, messages or [])
    await Runner.run(
        agent, messages or [], max_turns=1, run_config=run_config(agent_name)
    )
    return bridge.response


async def _stream(agent, messages):
    result = Runner.run_streamed(
        agent, messages, max_turns=1, run_config=run_config(agent.name)
    )
    try:
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                chunk = getattr(event.data, "gmart_chunk", None)
                if chunk is not None:
                    yield chunk
    finally:
        if not result.is_complete:
            result.cancel()
            async for _ in result.stream_events():
                pass


async def run_title(backend, model: str, prompt: str, *, stream=False, **settings):
    response = await run_completion(
        backend,
        model,
        [{"role": "user", "content": prompt}],
        agent_name="chat.title",
        completion_mode="generate",
        **settings,
    )
    return LlmGenerateResponse(model=model, response=response.message.content)


def strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.removesuffix("```")
    return text.strip()


class StructuredOutput(AgentOutputSchemaBase):
    """SDK output schema with the existing non-strict server decoding contract.

    Pydantic and domain validators still reject invalid plans. Non-strict describes
    the provider schema dialect, not whether validation happens.
    """

    def __init__(self, output_type, *, schema=None, normalize=None, validate=None):
        self.adapter = TypeAdapter(output_type)
        self.schema = schema if schema is not None else self.adapter.json_schema()
        self.normalize = normalize
        self.validate = validate

    def is_plain_text(self):
        return False

    def name(self):
        return self.schema.get("title", "structured_response")

    def json_schema(self):
        return self.schema

    def is_strict_json_schema(self):
        return False

    def validate_json(self, json_str):
        try:
            payload = json.loads(strip_json_fence(json_str))
            if self.normalize:
                payload = self.normalize(payload)
            parsed = self.adapter.validate_python(payload)
            return self.validate(parsed) if self.validate else parsed
        except ValueError as exc:
            raise ModelBehaviorError(f"Invalid {self.name()}: {exc}") from exc


async def run_structured(
    backend,
    model: str,
    messages: list[dict],
    output_type,
    *,
    agent_name: str,
    retries: int = 2,
    schema=None,
    normalize=None,
    validate=None,
    error_message: str | None = None,
    repair_instruction: str = "",
    stop_on_empty_truncation: bool = False,
    attempt_settings: Callable[[int, list[dict]], dict] | None = None,
    **settings: Any,
):
    """Validate via the SDK; repair only model output, never replay tools on failure.

    An attempt policy can preserve a specialist's output/context budget and provider
    reasoning fallback. It receives the actual repair conversation for sizing.
    """
    conversation = list(messages)
    output = StructuredOutput(
        output_type, schema=schema, normalize=normalize, validate=validate
    )
    for attempt in range(retries + 1):
        current = dict(settings)
        if attempt_settings:
            current.update(attempt_settings(attempt, conversation))
        agent, bridge = _agent(backend, model, agent_name, current, output)
        try:
            result = await Runner.run(
                agent, conversation, max_turns=1, run_config=run_config(agent_name)
            )
            if bridge.response is not None and bridge.response.get("done_reason") in {
                "length",
                "max_tokens",
                "incomplete",
                "content_filter",
            }:
                raise ModelBehaviorError("Incomplete structured model output")
            return result.final_output
        except ModelBehaviorError as exc:
            response = bridge.response
            raw = response["message"]["content"] if response is not None else ""
            stop = (
                stop_on_empty_truncation
                and not raw.strip()
                and response is not None
                and response.get("done_reason") == "length"
            )
            if attempt == retries or stop:
                message = (
                    error_message
                    or f"Model returned invalid {output.name()} after retries"
                )
                if response is not None and response.get("done_reason") in {
                    "length",
                    "max_tokens",
                    "incomplete",
                    "content_filter",
                }:
                    raise IncompleteStructuredOutput(f"{message}: {exc}") from exc
                raise ValueError(f"{message}: {exc}") from exc
            conversation.extend(
                [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Твой предыдущий ответ содержит невалидный JSON, который нарушает схему: {exc}. Исправь указанные поля "
                            "и верни только валидный JSON целиком, без markdown и пояснений."
                            + repair_instruction
                        ),
                    },
                ]
            )
    raise AssertionError("unreachable")
