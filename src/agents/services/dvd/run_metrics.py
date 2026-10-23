"""Stage latency, cost and pipeline decisions of one document-QA run.

One :class:`DvdRunMetrics` is kept per run in ``collected["metrics"]``. Stages
are timed with ``with metrics.stage(name):`` around the awaited work only (never
around a ``yield``), so the time a client takes to read events is not counted.
When the run ends, one ``DVD run metrics`` log line carries the whole record as
JSON; ``scripts/dvd_benchmark`` reads it back from ``/system/logs`` by
``request_id``. LLM calls and tokens come from :mod:`llm_usage`, which the model
adapters update for every completion made inside the run.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.agents.model_clients import llm_usage

LOG_EVENT = "DVD run metrics"


@dataclass
class DvdRunMetrics:
    request_id: str
    model: str | None = None
    started: float = field(default_factory=time.perf_counter)
    stages_ms: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    decisions: dict[str, Any] = field(default_factory=dict)
    ttft_ms: float | None = None
    total_ms: float | None = None
    usage: llm_usage.LlmUsage = field(default_factory=llm_usage.meter)

    def _elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.started) * 1000, 1)

    @contextmanager
    def stage(self, name: str):
        """Add the wall time of the enclosed block to stage ``name``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            spent = (time.perf_counter() - started) * 1000
            self.stages_ms[name] = round(self.stages_ms.get(name, 0.0) + spent, 1)

    def add(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def decide(self, **values: Any) -> None:
        self.decisions.update(values)

    def first_answer(self) -> None:
        """The first non-empty answer text reached the event journal."""
        if self.ttft_ms is None:
            self.ttft_ms = self._elapsed_ms()

    def snapshot(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model": self.model,
            "ttft_ms": self.ttft_ms,
            "total_latency_ms": self.total_ms,
            "stages_ms": dict(self.stages_ms),
            "counters": dict(self.counters),
            "decisions": dict(self.decisions),
            "llm": self.usage.snapshot(),
        }

    def finish(self) -> dict[str, Any]:
        if self.total_ms is None:
            self.total_ms = self._elapsed_ms()
        record = self.snapshot()
        logger.info(
            LOG_EVENT + " {}", json.dumps(record, ensure_ascii=False, sort_keys=True)
        )
        return record


class _NoMetrics(DvdRunMetrics):
    """Stand-in for code paths reached outside a pipeline run (e.g. unit tests)."""

    def __init__(self) -> None:
        super().__init__(request_id="", usage=llm_usage.LlmUsage())


def run_metrics(collected: dict[str, Any] | None) -> DvdRunMetrics:
    metrics = (collected or {}).get("metrics")
    return metrics if isinstance(metrics, DvdRunMetrics) else _NoMetrics()
