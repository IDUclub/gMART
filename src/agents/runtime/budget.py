"""Request-scoped admission control shared by nested agents and transport retries."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field


class BudgetExceeded(RuntimeError):
    def __init__(self, resource: str):
        self.resource = resource
        super().__init__(f"Analysis limit reached: {resource}")


@dataclass(frozen=True)
class BudgetLimits:
    total_tokens: int = 400_000
    model_calls: int = 60
    tool_calls: int = 80
    seconds: float = 600
    steps: int = 12
    context_tokens: int = 32_768
    output_tokens: int = 16_384
    final_reserve: int = 16_384


@dataclass
class RunBudget:
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    started: float = field(default_factory=time.monotonic)
    tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    estimated_calls: int = 0
    finalizing: bool = False
    exhausted: str | None = None
    reasoning_fallbacks: int = 0

    def fail(self, resource):
        self.exhausted = resource
        raise BudgetExceeded(resource)

    @property
    def remaining_seconds(self):
        return max(0.0, self.limits.seconds - (time.monotonic() - self.started))

    def check(self):
        if self.remaining_seconds <= 0:
            self.fail("time")

    def tool(self):
        self.check()
        if self.tool_calls >= self.limits.tool_calls:
            self.fail("tool_calls")
        self.tool_calls += 1

    def reserve(self, messages, schema, requested_output=None):
        self.check()
        # UTF-8 bytes are a conservative token estimate for byte-based model
        # tokenizers. Framing/schema have their own allowance. No silent truncation.
        input_tokens = token_bound(messages) + token_bound(schema) + 256
        reserve = 0 if self.finalizing else self.limits.final_reserve
        calls_reserve = 0 if self.finalizing else 1
        if self.model_calls >= self.limits.model_calls - calls_reserve:
            self.fail("model_calls")
        available = self.limits.total_tokens - self.tokens - reserve - input_tokens
        window = self.limits.context_tokens - input_tokens
        output = min(
            requested_output or self.limits.output_tokens,
            self.limits.output_tokens,
            available,
            window,
        )
        if output < 256:
            self.fail("context" if window < 256 else "tokens")
        amount = input_tokens + output
        self.tokens += amount
        self.model_calls += 1
        return Reservation(self, amount, output)

    def snapshot(self):
        return {
            "limits": asdict(self.limits),
            "charged_tokens": self.tokens,
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "estimated_calls": self.estimated_calls,
            "reasoning_fallbacks": self.reasoning_fallbacks,
            "remaining_seconds": round(self.remaining_seconds, 2),
        }


@dataclass
class Reservation:
    budget: RunBudget
    amount: int
    output: int
    settled: bool = False

    def settle(self, usage=None):
        if self.settled:
            return
        self.settled = True
        total = getattr(usage, "total_tokens", None)
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
        if isinstance(total, int) and total >= 0:
            self.budget.tokens += total - self.amount
        else:
            # Keep the entire reservation when the provider supplies no usage,
            # fails, or is cancelled; do not pretend unknown usage is free.
            self.budget.estimated_calls += 1


def token_bound(value):
    if value is None:
        return 0
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


current_budget: ContextVar[RunBudget | None] = ContextVar("gmart_budget", default=None)


@contextmanager
def budget_scope(budget):
    token = current_budget.set(budget)
    try:
        yield budget
    finally:
        current_budget.reset(token)
