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

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior

from src.agents.llm_agents.pydantic_model import (
    GmartChatModel,
    GmartModelSettings,
    from_chat_messages,
)
from src.agents.model_clients.llm_base import BaseLlmAdapter

OutputT = TypeVar("OutputT")

_AUTO: Any = object()

# What the pipelines have always told a model whose JSON did not parse.
FIX_JSON_PROMPT = (
    "Твой предыдущий ответ содержит невалидный JSON. "
    "Верни только валидный JSON нужной структуры без markdown и пояснений."
)


class StageOutputError(ValueError):
    """The model's answer never validated against the stage's output type.

    Subclasses ``ValueError`` so the pipelines' existing "the model answered
    nonsense" fallbacks keep catching it. ``text`` is the last raw answer.
    """

    def __init__(
        self,
        stage: str,
        text: str,
        done_reason: str | None,
        cause: BaseException,
        error: BaseException | None = None,
        thinking: str | None = None,
    ) -> None:
        super().__init__(f"{stage}: model output did not validate: {error or cause}")
        self.stage = stage
        self.text = text
        self.done_reason = done_reason
        # The validator's or parser's own exception (a ValidationError, a
        # ValueError…), when the failure came from one; pipelines quote it back.
        self.error = error
        self.thinking = thinking


@dataclass
class _RunState:
    """Per-run deps: remembers the last validation failure for StageOutputError."""

    error: BaseException | None = None


@dataclass
class StageResult(Generic[OutputT]):
    """What a stage run produced."""

    output: OutputT
    text: str
    done_reason: str | None = None
    thinking: str | None = None
    model_calls: int = 1
    # A streamed reply ended on the server's terminal chunk (always true unstreamed).
    terminal: bool = True

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
        retry_prompt: The user message that follows a non-validating answer;
            a validator's explanation is appended to it.
        parser: For answers that need more than validation against a type
            (restoring ids, checking against request data): the raw text goes to
            ``parser`` instead, whose return value is the output and whose
            ``ValueError`` counts as a failed validation. Exclusive with
            ``output_type``; send the JSON schema through ``schema``.
    """

    name: str
    output_type: Any = None
    retries: int = 0
    think: bool | None = None
    options: dict[str, Any] | None = None
    schema: Any = _AUTO
    extra: dict[str, Any] | None = None
    validators: Sequence[Callable[[Any], Any]] = field(default_factory=tuple)
    retry_prompt: str | None = None
    parser: Callable[[str], Any] | None = None

    def __post_init__(self) -> None:
        if self.parser is not None and self.output_type is not None:
            raise TypeError(f"{self.name}: give either output_type or parser")
        structured = self.output_type is not None or self.parser is not None
        if self.schema is _AUTO:
            self.schema = (
                self.output_type.model_json_schema()
                if isinstance(self.output_type, type)
                and issubclass(self.output_type, BaseModel)
                else None
            )
        self._structured = structured
        self._agent: Agent[_RunState, Any] = Agent(
            output_type=(
                NativeOutput(self.output_type) if self.output_type is not None else str
            ),
            deps_type=_RunState,
            retries={"output": self.retries if structured else 0},
            name=self.name,
        )
        if self.parser is not None:
            self._agent.output_validator(_as_validator(self.parser, transform=True))
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
            "gmart_retry_prompt": overrides.pop("retry_prompt", self.retry_prompt),
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
        state = _RunState()
        try:
            result = await self._agent.run(
                prompt,
                message_history=history,
                model=llm,
                deps=state,
                model_settings=self._settings(dict(overrides)),
            )
        except UnexpectedModelBehavior as exc:
            text = llm.last_text
            if not self._structured:
                # An empty text answer is the pipeline's to handle, not an error.
                return StageResult(
                    output="",  # type: ignore[arg-type]
                    text=text,
                    done_reason=llm.last_done_reason,
                    thinking=llm.last_thinking,
                    model_calls=llm.calls,
                )
            raise StageOutputError(
                self.name,
                text,
                llm.last_done_reason,
                exc,
                state.error,
                llm.last_thinking,
            ) from exc
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

        if self._structured:
            raise TypeError(f"{self.name}: only text stages stream")
        history, prompt = from_chat_messages(messages)
        llm = GmartChatModel(llm_client, model)
        settings = self._settings(dict(overrides))
        # The run lives in a task of its own. pydantic-ai's streamed run holds a
        # cancel scope and context variables that must be left in the task that
        # entered them; a consumer that stops early (a closed client stream, an
        # ``aclose`` from another task, a ``break``) would otherwise close them
        # elsewhere and cancel its own caller. The queue keeps backpressure.
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=1)

        async def produce() -> None:
            try:
                async with self._agent.run_stream(
                    prompt,
                    message_history=history,
                    model=llm,
                    deps=_RunState(),
                    model_settings=settings,
                ) as run:
                    async for delta in run.stream_text(delta=True, debounce_by=None):
                        if delta:
                            await queue.put((_DELTA, delta))
            except Exception as exc:  # re-raised in the consumer
                await queue.put((_ERROR, exc))
            else:
                await queue.put((_END, None))

        producer = asyncio.create_task(produce())
        chunks: list[str] = []
        try:
            while True:
                kind, value = await queue.get()
                if kind == _DELTA:
                    chunks.append(value)
                    yield value
                elif kind == _ERROR:
                    if isinstance(value, UnexpectedModelBehavior) and not chunks:
                        break  # an empty answer is the pipeline's to handle
                    raise value
                else:
                    break
        finally:
            await _stop(producer)
        if result is not None:
            result.append(
                StageResult(
                    output="".join(chunks),
                    text="".join(chunks),
                    done_reason=llm.last_done_reason,
                    thinking=llm.last_thinking,
                    model_calls=llm.calls,
                    terminal=llm.saw_done,
                )
            )


_DELTA, _ERROR, _END = "delta", "error", "end"


async def _stop(task: asyncio.Task[None]) -> None:
    """Cancel ``task`` if it still runs and wait for it to unwind.

    Only the task's own cancellation is absorbed: if the caller is cancelled
    while waiting, ``asyncio.wait`` raises and the cancellation propagates.
    """

    if not task.done():
        task.cancel()
    await asyncio.wait([task])
    if not task.cancelled():
        task.exception()  # retrieved: already delivered through the queue


def _as_validator(
    check: Callable[[Any], Any], *, transform: bool = False
) -> Callable[[RunContext[_RunState], Any], Any]:
    """Wrap a check as a pydantic-ai output validator.

    A parser (``transform``) always replaces the output with what it returns;
    a plain validator replaces it only when it returns something.
    """

    def validate(ctx: RunContext[_RunState], output: Any) -> Any:
        try:
            checked = check(output)
        except ModelRetry as exc:
            ctx.deps.error = exc
            raise
        except ValueError as exc:
            ctx.deps.error = exc
            raise ModelRetry(str(exc)) from exc
        return checked if transform or checked is not None else output

    validate.__name__ = getattr(check, "__name__", "validate")
    return validate
