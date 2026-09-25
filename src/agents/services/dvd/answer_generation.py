"""Per-draft token budgets and bounded continuation of interrupted answers.

Only visible text is carried into a continuation. Hidden reasoning cannot be
resumed through the chat API. Nothing is released until a complete draft exists;
the caller must still audit the assembled draft against the evidence.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from loguru import logger

from src.agents.model_clients.context_budget import (
    ANSWER_OUTPUT,
    output_budget,
    remaining_output_tokens,
)

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


_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def tables_to_lists(text: str) -> str:
    """Rewrite Markdown tables as bullet lists, one row per line.

    The draft prompt forbids tables, yet gpt-oss still writes them for overviews.
    A row keeps every cell with its column name, so labels [N] stay on the line
    the critic audits and the user reads.
    """

    lines = text.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        is_table = (
            index + 1 < len(lines)
            and _TABLE_ROW.match(lines[index])
            and _TABLE_RULE.match(lines[index + 1])
        )
        if not is_table:
            out.append(lines[index])
            index += 1
            continue
        header = _cells(lines[index])
        index += 2
        while index < len(lines) and _TABLE_ROW.match(lines[index]):
            pairs = []
            for name, cell in zip(header, _cells(lines[index])):
                cell = " ".join(cell.replace("<br>", " ").replace("•", "").split())
                # The row number column carries no content.
                if not cell or name in {"№", "#", "N"}:
                    continue
                pairs.append(f"{name}: {cell}" if name else cell)
            if pairs:
                out.append("- " + "; ".join(pairs))
            index += 1
    return "\n".join(out)


class StreamingTableRewriter:
    """``tables_to_lists`` for a streamed draft.

    Ordinary text passes through as it arrives. Lines that start with ``|`` are held
    until the block ends, then released rewritten, so the reader never sees a table
    that is later replaced. ``flush`` releases whatever is held at the end.
    """

    def __init__(self) -> None:
        self._pending = ""  # start of the current line, undecided (blank or a row)
        self._passing = False  # the current line is plain text, already released
        self._held: list[str] = []  # complete lines of a possible table

    def _release_held(self) -> str:
        text = "".join(line + "\n" for line in self._held)
        self._held = []
        return tables_to_lists(text[:-1]) + "\n" if text else ""

    def feed(self, text: str) -> str:
        out: list[str] = []
        segments = text.split("\n")
        for index, segment in enumerate(segments):
            terminated = index < len(segments) - 1
            if self._passing:
                out.append(segment + ("\n" if terminated else ""))
                self._passing = not terminated
                continue
            self._pending += segment
            stripped = self._pending.lstrip()
            if terminated:
                if stripped.startswith("|"):
                    self._held.append(self._pending)
                else:
                    out.append(self._release_held() + self._pending + "\n")
                self._pending = ""
            elif stripped and not stripped.startswith("|"):
                out.append(self._release_held() + self._pending)
                self._pending, self._passing = "", True
        return "".join(out)

    def flush(self) -> str:
        pending, self._pending, self._passing = self._pending, "", False
        if pending.lstrip().startswith("|"):
            self._held.append(pending)
            return self._release_held()[:-1]
        return self._release_held() + pending


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
        self.failed_parts: list[str] = []
        self.retries = int(os.getenv("DVD_ANSWER_RETRIES", "2"))
        if not 0 <= self.retries <= 4:
            raise ValueError("invalid DVD answer retries")

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
        prefix = ""
        self.failed_parts.clear()
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
            evidence = context
            messages = build_messages(evidence) + continuation
            # The draft grows with the evidence it covers; a truncated draft is
            # continued below, so the limit bounds each request, not the answer.
            room = await output_budget(
                self.llm_client,
                model,
                messages,
                self.reducer.window,
                output=ANSWER_OUTPUT,
            )
            if room.window_rest < 128:
                # Only evidence can be reduced. Preserve instructions, history
                # and every visible continuation prefix.
                fixed_available = await remaining_output_tokens(
                    self.llm_client,
                    model,
                    build_messages("") + continuation,
                    self.reducer.window,
                )
                allowance = fixed_available - self.reducer.window // 4
                if allowance < 512:
                    raise AnswerGenerationError("answer_generation_no_context_room")
                try:
                    prepared = await self.reducer.prepare(
                        model,
                        question,
                        context,
                        budget_limit=allowance,
                    )
                except ValueError as exc:
                    raise AnswerGenerationError(
                        "answer_generation_context_reduction_failed"
                    ) from exc
                if prepared.failed_parts:
                    if not prepared.processed_parts or not prepared.text.strip():
                        raise AnswerGenerationError(
                            "answer_generation_context_incomplete"
                        )
                    self.failed_parts.extend(prepared.failed_parts)
                context = prepared.text
                messages = build_messages(context) + continuation
                room = await output_budget(
                    self.llm_client,
                    model,
                    messages,
                    self.reducer.window,
                    output=ANSWER_OUTPUT,
                )
                if room.window_rest < 128:
                    raise AnswerGenerationError("answer_generation_no_context_room")
            budget = room.tokens
            input_cost = self.reducer.window - room.window_rest
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
                logger.warning(
                    "DVD answer continuation overlap mismatch iteration={} attempt={}",
                    iteration,
                    attempt + 1,
                )
                continue
            if reason in {"length", "max_tokens"}:
                prefix = assembled if assembled.strip() else ""
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
