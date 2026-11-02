"""Typed LLM steps of the gMART pipelines, run as pydantic-ai agents.

A pipeline stays a deterministic program; the places where it asks the model
something are :class:`LlmStage` objects. A stage declares what it returns
(``output_type``), how it is called (stage key, ``think``, options) and how many
times pydantic-ai may send a non-validating answer back to the model with the
validation errors. Running it returns a :class:`StageResult` with the typed
output plus what the pipelines still inspect about the reply (the raw text, the
server's ``done_reason``, the reasoning trace).

Messages are given as the adapter's chat messages, so the prompts the services
build — and the requests the benchmarks freeze — are sent exactly as before.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior

from src.agents.llm_agents.pydantic_model import (
    GmartChatModel,
    GmartModelSettings,
    from_chat_messages,
)
from src.agents.model_clients.llm_base import BaseLlmAdapter

OutputT = TypeVar("OutputT")

_AUTO: Any = object()


class StageOutputError(ValueError):
    """The model's answer never validated against the stage's output type.

    Subclasses ``ValueError`` so the pipelines' existing "the model answered
    nonsense" fallbacks keep catching it. ``text`` is the last raw answer.
    """

    def __init__(
        self, stage: str, text: str, done_reason: str | None, cause: BaseException
    ) -> None:
        super().__init__(f"{stage}: model output did not validate: {cause}")
        self.stage = stage
        self.text = text
        self.done_reason = done_reason


@dataclass
class StageResult(Generic[OutputT]):
    """What a stage run produced."""

    output: OutputT
    text: str
    done_reason: str | None = None
    thinking: str | None = None
    model_calls: int = 1

    @property
    def truncated(self) -> bool:
        return self.done_reason in {"length", "repetition", "incomplete"}


@dataclass
class LlmStage(Generic[OutputT]):
    """One typed model call of a pipeline.

    Args:
        name: Stage key (``"orchestrator.plan"``…); the adapter's ``request_kind``.
        output_type: Pydantic model (or any type pydantic validates) the answer
            must satisfy, sent to the server as a JSON schema; ``None`` for text.
        retries: How many times a non-validating answer is sent back with the
            errors before :class:`StageOutputError`. Text stages never retry.
        think: ``think=`` for the adapter.
        options: Adapter options (``temperature``, ``num_predict``, ``num_ctx``).
        schema: JSON schema sent as ``format``; defaults to the output model's own
            ``model_json_schema()`` so the server sees what it saw before.
        extra: Further adapter keywords.
        validators: Extra checks on the parsed output; raise ``ModelRetry`` (or
            ``ValueError``) to send the answer back for another attempt.
    """

    name: str
    output_type: Any = None
    retries: int = 0
    think: bool | None = None
    options: dict[str, Any] | None = None
    schema: Any = _AUTO
    extra: dict[str, Any] | None = None
    validators: Sequence[Callable[[Any], Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        structured = self.output_type is not None
        if self.schema is _AUTO:
            self.schema = (
                self.output_type.model_json_schema()
                if isinstance(self.output_type, type)
                and issubclass(self.output_type, BaseModel)
                else None
            )
        self._agent: Agent[None, Any] = Agent(
            output_type=NativeOutput(self.output_type) if structured else str,
            retries={"output": self.retries if structured else 0},
            name=self.name,
        )
        for check in self.validators:
            self._agent.output_validator(_as_validator(check))

    # ------------------------------------------------------------------ #
    def _settings(self, overrides: dict[str, Any]) -> GmartModelSettings:
        options = overrides.pop("options", self.options)
        settings: GmartModelSettings = {
            "gmart_request_kind": overrides.pop("request_kind", self.name),
            "gmart_think": overrides.pop("think", self.think),
            "gmart_options": dict(options) if options is not None else None,
            "gmart_format": overrides.pop("schema", self.schema),
            "gmart_extra": {**(self.extra or {}), **overrides},
        }
        return settings

    async def run(
        self,
        llm_client: BaseLlmAdapter,
        model: str,
        messages: list[dict[str, Any]],
        **overrides: Any,
    ) -> StageResult[OutputT]:
        """Run the stage once (plus its validation retries).

        ``overrides`` replace the stage's ``options``/``think``/``schema``/
        ``request_kind`` for this run; any other keyword goes to the adapter.
        """

        history, prompt = from_chat_messages(messages)
        llm = GmartChatModel(llm_client, model)
        try:
            result = await self._agent.run(
                prompt,
                message_history=history,
                model=llm,
                model_settings=self._settings(dict(overrides)),
            )
        except UnexpectedModelBehavior as exc:
            text = llm.last_text
            if self.output_type is None:
                # An empty text answer is the pipeline's to handle, not an error.
                return StageResult(
                    output="",  # type: ignore[arg-type]
                    text=text,
                    done_reason=llm.last_done_reason,
                    thinking=llm.last_thinking,
                    model_calls=llm.calls,
                )
            raise StageOutputError(self.name, text, llm.last_done_reason, exc) from exc
        return StageResult(
            output=result.output,
            text=llm.last_text,
            done_reason=llm.last_done_reason,
            thinking=llm.last_thinking,
            model_calls=llm.calls,
        )

    async def stream(
        self,
        llm_client: BaseLlmAdapter,
        model: str,
        messages: list[dict[str, Any]],
        *,
        result: list[StageResult[str]] | None = None,
        **overrides: Any,
    ) -> AsyncIterator[str]:
        """Stream a text stage's answer delta by delta.

        When ``result`` is given, the finished :class:`StageResult` is appended to
        it once the stream ends (``done_reason`` included).
        """

        if self.output_type is not None:
            raise TypeError(f"{self.name}: only text stages stream")
        history, prompt = from_chat_messages(messages)
        llm = GmartChatModel(llm_client, model)
        chunks: list[str] = []
        try:
            async with self._agent.run_stream(
                prompt,
                message_history=history,
                model=llm,
                model_settings=self._settings(dict(overrides)),
            ) as run:
                async for delta in run.stream_text(delta=True, debounce_by=None):
                    if delta:
                        chunks.append(delta)
                        yield delta
        except UnexpectedModelBehavior:
            if chunks:
                raise
        if result is not None:
            result.append(
                StageResult(
                    output="".join(chunks),
                    text="".join(chunks),
                    done_reason=llm.last_done_reason,
                    thinking=llm.last_thinking,
                    model_calls=llm.calls,
                )
            )


def _as_validator(check: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def validate(output: Any) -> Any:
        try:
            checked = check(output)
        except ModelRetry:
            raise
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        return output if checked is None else checked

    validate.__name__ = getattr(check, "__name__", "validate")
    return validate
