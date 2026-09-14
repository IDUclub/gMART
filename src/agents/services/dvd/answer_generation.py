"""Per-draft token budgets and bounded continuation of interrupted answers.

Only visible text is carried into a continuation. Hidden reasoning cannot be
resumed through the chat API. Nothing is released until a complete draft exists;
the caller must still audit the assembled draft against the evidence.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

from loguru import logger

from .context_reducer import DvdContextReducer, cost

_CONTINUE = (
    "Продолжи незавершённый ответ. Начни с ТОЧНОГО повторения указанного ниже "
    "незавершённого фрагмента, затем допиши недостающее. Не повторяй более ранний текст "
    "и не добавляй вступление. Программа удалит проверенное перекрытие. "
    "Предыдущий ответ — непроверенный черновик, не источник фактов. "
    "Соблюдай исходный вопрос и правила, опирайся только на документальные источники."
)


def continuation_anchor(prefix: str) -> str:
    # Repeat the unfinished line. Bound overlap for very long paragraphs, keeping
    # a word boundary where available; the preserved prefix is never discarded.
    start = max(prefix.rfind("\n") + 1, len(prefix) - 512, 0)
    if start and prefix[start - 1] != "\n":
        boundary = prefix.find(" ", start)
        if boundary >= 0 and boundary + 1 < len(prefix):
            start = boundary + 1
    return prefix[start:]


class AnswerGenerationError(ValueError):
    """Stable failure code, without source text or generated content."""


def message_cost(messages: list[dict]) -> int:
    # Same conservative UTF-8 upper estimate as evidence reduction, plus framing.
    return 256 + sum(64 + cost(str(m.get("content", ""))) for m in messages)


def append_continuation(prefix: str, addition: str) -> str:
    if prefix and addition.startswith(prefix):
        return addition
    # Remove only substantial verbatim overlap; short matching suffixes can be
    # coincidental parts of words or numbers and must not be silently removed.
    for size in range(min(len(prefix), len(addition)), 31, -1):
        if prefix[-size:] == addition[:size]:
            return prefix + addition[size:]
    return prefix + addition


class DvdAnswerGenerator:
    def __init__(self, reducer: DvdContextReducer, *, llm_client=None):
        self.reducer = reducer
        self.llm_client = reducer.llm_client if llm_client is None else llm_client
        self.maximum = reducer.output_tokens
        self.minimum = min(
            int(os.getenv("DVD_ANSWER_MIN_TOKENS", "4096")), self.maximum
        )
        self.retries = int(os.getenv("DVD_ANSWER_RETRIES", "2"))
        if self.minimum < 128 or not 0 <= self.retries <= 4:
            raise ValueError("invalid DVD answer budget or retries")

    def initial_budget(self, context: str, question: str) -> int:
        # A heuristic, not a tokenizer count: larger evidence/questions allow
        # more reasoning and synthesis. Exact fit is checked on every request.
        extra = ((cost(context) + cost(question) + 4095) // 4096) * 512
        return min(self.maximum, self.minimum + extra)

    async def generate(
        self,
        model: str,
        question: str,
        context: str,
        temperature: float,
        build_messages: Callable[[str], list[dict]],
        *,
        iteration: int,
    ) -> str:
        desired = self.initial_budget(context, question)
        prefix = ""
        previous_budget = 0
        previous_progress = False
        for attempt in range(self.retries + 1):
            continuation = (
                [
                    {"role": "assistant", "content": prefix},
                    {
                        "role": "user",
                        "content": _CONTINUE
                        + "\nНачальный фрагмент (JSON-строка): "
                        + json.dumps(continuation_anchor(prefix), ensure_ascii=False),
                    },
                ]
                if prefix
                else []
            )
            fixed_cost = message_cost(build_messages("") + continuation)
            # Keep at least a small evidence allowance. Never truncate history,
            # instructions or the prefix to make a request appear to fit.
            budget = min(desired, self.reducer.window - fixed_cost - 512)
            if budget < 128:
                raise AnswerGenerationError("answer_generation_no_context_room")
            if attempt and budget <= previous_budget and not previous_progress:
                raise AnswerGenerationError(
                    "answer_generation_incomplete: no_budget_growth"
                )
            allowance = self.reducer.window - fixed_cost - budget
            evidence = context
            if cost(evidence) > allowance:
                try:
                    prepared = await self.reducer.prepare(
                        model, question, context, budget_limit=allowance
                    )
                except ValueError as exc:
                    raise AnswerGenerationError(
                        "answer_generation_context_reduction_failed"
                    ) from exc
                if prepared.failed_parts:
                    raise AnswerGenerationError("answer_generation_context_incomplete")
                evidence = prepared.text
            messages = build_messages(evidence) + continuation
            input_cost = message_cost(messages)
            if input_cost + budget > self.reducer.window:
                raise AnswerGenerationError("answer_generation_no_context_room")
            logger.info(
                "DVD answer model={} iteration={} attempt={} input_upper_estimate={} "
                "output_budget={} window={} prefix_bytes={}",
                model,
                iteration,
                attempt + 1,
                input_cost,
                budget,
                self.reducer.window,
                cost(prefix),
            )
            pieces = []
            reason = None
            terminal = False
            async for part in await self.llm_client.chat(
                model,
                messages,
                think=False,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": budget,
                    "num_ctx": self.reducer.window,
                },
            ):
                if part.message.content:
                    pieces.append(part.message.content)
                if getattr(part, "done_reason", None):
                    reason = part.done_reason
                terminal = terminal or bool(getattr(part, "done", False))
            addition = "".join(pieces)
            anchor = continuation_anchor(prefix)
            boundary_valid = (
                not prefix
                or not anchor
                or addition.startswith(anchor)
                or addition.startswith(prefix)
            )
            if prefix and addition.startswith(prefix):
                assembled = addition
            elif prefix and anchor and addition.startswith(anchor):
                assembled = prefix[: -len(anchor)] + addition
            else:
                assembled = append_continuation(prefix, addition)
            progress = len(assembled) > len(prefix)
            logger.info(
                "DVD answer model={} iteration={} attempt={} finish_reason={} "
                "visible_bytes={} progress={}",
                model,
                iteration,
                attempt + 1,
                reason,
                cost(addition),
                progress,
            )
            if reason == "content_filter":
                raise AnswerGenerationError("answer_generation_filtered")
            if reason == "incomplete":
                raise AnswerGenerationError(
                    "answer_generation_incomplete: missing_terminal"
                )
            if reason not in {None, "stop", "length", "max_tokens"}:
                raise AnswerGenerationError("answer_generation_unexpected_stop")
            if prefix and not boundary_valid:
                # Keep the known prefix and retry; guessing whitespace/word
                # boundaries can alter numbers or join unrelated statements.
                previous_budget, previous_progress = budget, False
                desired = min(self.maximum, max(desired, budget * 2))
                logger.warning(
                    "DVD answer continuation overlap mismatch iteration={} attempt={}",
                    iteration,
                    attempt + 1,
                )
                continue
            if reason in {"length", "max_tokens"}:
                prefix = assembled if assembled.strip() else ""
                previous_budget, previous_progress = budget, progress
                desired = min(self.maximum, max(desired, budget * 2))
                continue
            if not (reason == "stop" or terminal):
                raise AnswerGenerationError(
                    "answer_generation_incomplete: missing_terminal"
                )
            if not assembled.strip() or not progress:
                raise AnswerGenerationError(
                    "answer_generation_incomplete: empty_completion"
                )
            return assembled
        raise AnswerGenerationError("answer_generation_incomplete: retries_exhausted")
