"""How much slower than usual the LLM server answers right now, and deadlines that follow it.

One vLLM serves every agent (and other contours), so a pipeline that takes a minute
on an idle server can take ten when the server is busy. A fixed wall-clock deadline
then stops runs that were making normal progress, only slowly. Every completed call
is compared with the time it would take at the nominal speed:

    expected = LATENCY + prompt_tokens / PREFILL_SPEED + completion_tokens / nominal

and the recent calls give ``slowdown = Σ elapsed / Σ expected`` (never below 1). A call
still in flight longer than its whole ``max_tokens`` would take at the nominal speed
bounds the slowdown from below, so a server that is stuck on the very first call of
a pipeline is detected before that call returns.

A :class:`PacedDeadline` counts ``dt / slowdown`` instead of ``dt``: its limit is the
time the work would take on the usual server. ``LLM_DEADLINE_MAX_FACTOR`` caps the
stretch, so a run never lasts longer than ``limit × factor`` of wall time.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

NOMINAL_TOKENS_PER_SECOND_ENV = "LLM_NOMINAL_TOKENS_PER_SECOND"
DEADLINE_MAX_FACTOR_ENV = "LLM_DEADLINE_MAX_FACTOR"
#: Decode speed of gpt-oss-20b on the shared vLLM when it is not overloaded
#: (131 tok/s measured on 2026-09-25; 13 tok/s while it was).
NOMINAL_TOKENS_PER_SECOND = 100.0
DEADLINE_MAX_FACTOR = 4.0

#: Per-request overhead and prompt processing speed at the nominal pace.
LATENCY_SECONDS = 1.0
PREFILL_TOKENS_PER_SECOND = 5000.0
#: Allowance for an unknown prompt when bounding a call that is still running.
IN_FLIGHT_PROMPT_SECONDS = 10.0
#: Only recent calls describe the current load.
WINDOW_SECONDS = 10 * 60
MAX_SAMPLES = 200


def _env_float(name: str, default: float, minimum: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class _Sample:
    finished_at: float
    elapsed: float
    expected: float


class LlmPace:
    """Process-wide estimate of the LLM slowdown relative to its nominal speed."""

    def __init__(
        self,
        nominal_tokens_per_second: float = NOMINAL_TOKENS_PER_SECOND,
        max_factor: float = DEADLINE_MAX_FACTOR,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.nominal_tokens_per_second = nominal_tokens_per_second
        self.max_factor = max_factor
        self._clock = clock
        self._samples: deque[_Sample] = deque(maxlen=MAX_SAMPLES)
        self._in_flight: dict[int, tuple[float, int | None]] = {}
        self._ids = itertools.count()

    @classmethod
    def from_env(cls) -> LlmPace:
        return cls(
            _env_float(NOMINAL_TOKENS_PER_SECOND_ENV, NOMINAL_TOKENS_PER_SECOND, 1.0),
            _env_float(DEADLINE_MAX_FACTOR_ENV, DEADLINE_MAX_FACTOR, 1.0),
        )

    def started(self, max_tokens: int | None = None) -> int:
        call_id = next(self._ids)
        self._in_flight[call_id] = (self._clock(), max_tokens)
        return call_id

    def finished(
        self,
        call_id: int,
        *,
        completion_tokens: int | None,
        prompt_tokens: int | None = None,
    ) -> None:
        """Record a call; one without a token count only leaves the in-flight set."""

        started = self._in_flight.pop(call_id, None)
        if started is None or not isinstance(completion_tokens, int):
            return
        if not isinstance(prompt_tokens, int):
            prompt_tokens = 0
        now = self._clock()
        expected = (
            LATENCY_SECONDS
            + prompt_tokens / PREFILL_TOKENS_PER_SECOND
            + completion_tokens / self.nominal_tokens_per_second
        )
        self._samples.append(_Sample(now, now - started[0], expected))

    def slowdown(self) -> float:
        """How many times slower than nominal the server is, within ``[1, max_factor]``."""

        now = self._clock()
        while self._samples and now - self._samples[0].finished_at > WINDOW_SECONDS:
            self._samples.popleft()
        factor = 1.0
        if self._samples:
            elapsed = sum(sample.elapsed for sample in self._samples)
            expected = sum(sample.expected for sample in self._samples)
            factor = elapsed / expected
        for call_id, (started, max_tokens) in list(self._in_flight.items()):
            if now - started > WINDOW_SECONDS:
                # An abandoned stream that was never closed must not pin the estimate.
                del self._in_flight[call_id]
            elif isinstance(max_tokens, int) and max_tokens > 0:
                longest = (
                    LATENCY_SECONDS
                    + IN_FLIGHT_PROMPT_SECONDS
                    + max_tokens / self.nominal_tokens_per_second
                )
                factor = max(factor, (now - started) / longest)
        return min(max(factor, 1.0), self.max_factor)


llm_pace = LlmPace.from_env()


class PacedDeadline:
    """A time limit counted in seconds of work at the LLM's nominal speed.

    ``advance`` must be called regularly (the pipelines do it on every deadline
    check, about once a second): each interval is divided by the slowdown measured
    at that moment, so a slow spell is paid for only while it lasts.
    """

    def __init__(
        self,
        limit_seconds: float,
        *,
        pace: LlmPace | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit_seconds = limit_seconds
        self.pace = pace or llm_pace
        self._clock = clock
        self.started = clock()
        self._last = self.started
        self._paused = 0.0
        self.consumed = 0.0

    def advance(self, paused_seconds: float = 0.0) -> float:
        """Account the time since the last call; ``paused_seconds`` is cumulative."""

        now = self._clock()
        pause = max(0.0, paused_seconds - self._paused)
        self._paused = max(self._paused, paused_seconds)
        active = max(0.0, now - self._last - pause)
        self.consumed += active / self.pace.slowdown()
        self._last = now
        return self.consumed

    def exceeded(self, paused_seconds: float = 0.0) -> bool:
        return self.advance(paused_seconds) >= self.limit_seconds

    @property
    def wall_seconds(self) -> float:
        return self._clock() - self.started


class PipelineDeadlineExceeded(TimeoutError):
    """A pipeline ran out of its (load-adjusted) time budget."""

    def __init__(self, message: str, wall_seconds: float):
        super().__init__(message)
        self.wall_seconds = wall_seconds

    @property
    def user_message(self) -> str:
        minutes = max(1, round(self.wall_seconds / 60))
        return (
            f"Запрос не уложился в отведённое время (около {minutes} мин с учётом "
            "текущей загрузки языковой модели). Повторите запрос позже или сузьте его."
        )


async def paced_wait_for(
    awaitable: Awaitable[T], limit_seconds: float, *, poll_seconds: float = 1.0
) -> T:
    """``asyncio.wait_for`` whose timeout stretches while the LLM server is slow."""

    deadline = PacedDeadline(limit_seconds)
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=poll_seconds)
            if done:
                return task.result()
            if deadline.exceeded():
                raise PipelineDeadlineExceeded(
                    "LLM call deadline exceeded", deadline.wall_seconds
                )
    finally:
        if not task.done():
            task.cancel()
