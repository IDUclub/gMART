"""Sentence snapshots of an unverified draft, separate from answer chunks."""

import os
import re
import time

from loguru import logger

from . import knowledge_block, source_citations

_BOUNDARY = re.compile(r"\n|[.!?…]+(?=\s|$)")
_ABBREVIATION = re.compile(
    r"(?:\b(?:п|пп|ст|м|г|гг|ул|д|рис|табл|им|см|ср|др|пр|руб|коп)|"
    r"\bт\s*\.\s*[едп]|\bи\s+т\s*\.\s*д|\b[А-ЯЁA-Z])\.$",
    re.I,
)
_LIST_MARKER = re.compile(
    r"\s*(?:[-*]\s*)?(?:\d+(?:\.\d+)*[.)]|[а-яёa-z]\))(?=\s|$)", re.I
)


def complete_sentences(text: str, *, final: bool = False) -> str:
    """Keep complete lines or sentences; an EOF terminator needs final=True."""
    end = 0
    for match in _BOUNDARY.finditer(text):
        if match.group() == "\n":
            end = match.end()
            continue
        head = text[: match.end()]
        if match.group() == "." and (
            _ABBREVIATION.search(head)
            or _LIST_MARKER.fullmatch(text[end : match.end()])
        ):
            continue
        following = text[match.end() :].lstrip()
        if following:
            first = following[0]
            if not (
                first.isupper()
                or first.isdigit()
                or first in "-*["
                or _LIST_MARKER.match(following)
            ):
                continue
        elif not final:
            continue
        end = match.end()
    return text[:end].rstrip()


def min_interval() -> float:
    try:
        return (
            max(0, int(os.getenv("DVD_ANSWER_PREVIEW_MIN_INTERVAL_MS") or "700")) / 1000
        )
    except ValueError:
        logger.warning("Invalid DVD_ANSWER_PREVIEW_MIN_INTERVAL_MS; using 700")
        return 0.7


class AnswerPreview:
    def __init__(self, refs, iteration: int, *, last_sent: float | None = None):
        self.refs = refs
        self.iteration = iteration
        self.interval = min_interval()
        self.last_sent = last_sent
        self.sent = ""

    def event(self, text: str, *, final: bool = False) -> dict | None:
        now = time.monotonic()
        if (
            not final
            and self.last_sent is not None
            and now - self.last_sent < self.interval
        ):
            return None
        preview = complete_sentences(text, final=final)
        # Table conversion can rewrite earlier rows of an unfinished table.
        preview = source_citations.readable(knowledge_block.clean(preview), self.refs)
        if (
            not preview
            or len(preview) <= len(self.sent)
            or not preview.startswith(self.sent)
        ):
            return None
        self.sent, self.last_sent = preview, now
        return {
            "type": "status",
            "content": {
                "status": "answer_preview",
                "text": "Предварительный анализ (ещё не проверен):\n" + preview,
                "preview": preview,
                "iteration": self.iteration,
            },
        }
