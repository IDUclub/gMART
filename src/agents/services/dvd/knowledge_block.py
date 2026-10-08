"""Answer from the documents first, then from the model's own knowledge, marked.

The draft states what the retrieved fragments support, with source labels, and puts
everything else into one closing block that starts with ``KNOWLEDGE_HEADING``. The
critic audits only the grounded part and only once: a line it cannot confirm moves
into that block instead of costing a rewrite and a second audit. A line that
contradicts the sources, or that carries figures or clause numbers, is dropped:
such details from memory are exactly where a small model is wrong.

Nothing in the answer sends the user to external sources (official texts, legal
databases, websites, specialists); :func:`strip_external_referrals` removes such
advice from unlabelled lines.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from . import flags
from .document_reference import DESIGNATION
from .dvd_reasoning import _LIST_MARKER

KNOWLEDGE_HEADING = "Без опоры на загруженные документы"
KNOWLEDGE_TITLE = (
    f"**{KNOWLEDGE_HEADING}** (общие сведения модели, найденными фрагментами "
    "не подтверждены):"
)
NO_GROUNDED_ANSWER = "В найденных фрагментах документов прямого ответа на вопрос нет."

ANSWER_RULES = (
    "- Сначала ответь по фрагментам: каждое утверждение с меткой источника. Если "
    "во фрагментах нет нужных сведений (всех или части), коротко скажи, чего именно "
    "в них нет.\n"
    f"- Затем, только если без этого вопрос остаётся без ответа, в самом конце дай "
    f"отдельный блок, который начинается строкой «{KNOWLEDGE_HEADING}:», — ответ по "
    "твоим общим знаниям. В этом блоке нет меток источников и нет сведений из "
    "фрагментов. Не называй в нём числовые значения, размеры, расстояния, сроки, "
    "проценты, номера пунктов, статей, таблиц и редакций; можно объяснить смысл "
    "требований в общих чертах и назвать документы, которые обычно регулируют тему.\n"
)
NO_EXTERNAL_RULE = (
    "- Никогда не советуй обращаться к внешним источникам: официальным текстам, "
    "сайтам, справочно-правовым системам, специалистам, органам власти; не предлагай "
    "проверить актуальную редакцию. Отвечай тем, что есть.\n"
)

_HEADING = re.compile(
    rf"^\W*{re.escape(KNOWLEDGE_HEADING)}\W*?(?::|\s—|\s-)?\s*(?P<rest>.*)$", re.I
)
_LABEL = re.compile(r"\s*\[\d+\]")
_DIGIT = re.compile(r"\d")
_LETTER = re.compile(r"[А-Яа-яЁёA-Za-z]")

# Advice to look elsewhere: an imperative or a recommendation to check, ask or read
# somewhere, aimed at a target outside this application.
_TARGET = (
    r"(?:официальн|актуальн|первоисточник|оригинал|сайт|портал|интернет|орган\w*\s"
    r"|ведомств|министерств|минстро|росстандарт|администраци|специалист|юрист"
    r"|эксперт|консультант|гарант|техэксперт|справочн|правов\w+\s+систем"
    r"|баз\w*\s+данных|полн\w+\s+текст|источник)"
)
_ADVICE = re.compile(
    r"(?:\b(?:обратитесь|обращайтесь|уточните|проверьте|сверьтесь|сверьте|ознакомьтесь"
    r"|изучите|проконсультируйтесь|посмотрите|загляните|запросите|найдите)"
    r"|\b(?:рекоменду\w*|советую|советуем|стоит|следует|нужно|необходимо|лучше"
    r"|можно)\s+(?:\w+\s+){0,2}(?:обратиться|уточнить|проверить|свериться|сверить"
    r"|ознакомиться|изучить|проконсультироваться|посмотреть|запросить|найти))"
    rf"\b[^.!?\n]*?{_TARGET}",
    re.I,
)
_ALWAYS = re.compile(
    r"https?://|\bwww\.|консультант\s*плюс|консультантплюс|\bгарант\b|техэксперт"
    r"|\bcntd\b|pravo\.gov|\bвнешн\w+\s+(?:источник|ресурс)"
    r"|\bофициальн\w+\s+(?:текст|источник|сайт|публикаци|портал|ресурс|издани)",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def enabled() -> bool:
    return flags.enabled(flags.KNOWLEDGE_FALLBACK)


def split(body: str) -> tuple[str, list[str]]:
    """Separate the grounded part of a draft from its own-knowledge block."""
    lines = body.splitlines()
    for index, line in enumerate(lines):
        if match := _HEADING.match(line.strip()):
            # "**Без опоры …:** текст" keeps the text after the heading's markup.
            rest = match["rest"].lstrip("*_ :—-")
            rest = [rest] if _LETTER.search(rest) else []
            knowledge = [l for l in [*rest, *lines[index + 1 :]] if l.strip()]
            return "\n".join(lines[:index]).rstrip(), knowledge
    return body, []


def _referral(sentence: str) -> bool:
    return bool(_ALWAYS.search(sentence) or _ADVICE.search(sentence))


def strip_external_referrals(text: str) -> str:
    """Drop sentences that send the user elsewhere.

    Only lines without a source label are touched: a labelled line restates a
    retrieved norm, which may itself mention a body or an official publication.
    """
    kept = []
    for line in text.splitlines():
        if _LABEL.search(line) or not line.strip():
            kept.append(line)
            continue
        cleaned = " ".join(s for s in _SENTENCE.split(line) if not _referral(s))
        if _LETTER.search(_LIST_MARKER.sub("", cleaned)):
            kept.append(cleaned.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _has_figures(line: str) -> bool:
    """Whether a line states quantities or clause numbers, not just documents."""
    text = DESIGNATION.sub(" ", _LIST_MARKER.sub("", _LABEL.sub(" ", line)))
    return bool(_DIGIT.search(text))


def _knowledge_line(line: str) -> str | None:
    line = _LABEL.sub("", line).rstrip()
    if not _LIST_MARKER.sub("", line).strip() or _has_figures(line):
        return None
    return line


def _claim_key(line: str) -> str:
    return _LIST_MARKER.sub("", line).strip()


def sort_lines(grounded: str, claims: Iterable) -> tuple[str, list[str], dict]:
    """Keep audited-supported lines; move or drop the ones the audit rejected.

    ``claims`` are the critic's line audits (``text`` is a whole answer line,
    ``status`` supported | contradicted | insufficient). Lines the critic did not
    audit — headings, introductions — stay, unless no line was confirmed at all:
    then there is nothing grounded left for them to introduce. A contradicted line
    is dropped; an unconfirmed one moves to the own-knowledge block unless it
    carries figures.
    """
    status = {claim.text: claim.status for claim in claims}
    kept, moved, counts = [], [], {"supported": 0, "moved": 0, "dropped": 0}
    for line in grounded.splitlines():
        verdict = status.get(_claim_key(line))
        if verdict is None:
            kept.append(line)
        elif verdict == "supported":
            counts["supported"] += 1
            kept.append(line)
        elif verdict == "insufficient" and (text := _knowledge_line(line)):
            counts["moved"] += 1
            moved.append(text)
        else:
            counts["dropped"] += 1
    if not counts["supported"]:
        kept = []
    return "\n".join(kept).strip(), moved, counts


def render(
    grounded: str, knowledge: Iterable[str], quotation: str | None = None
) -> str:
    """Assemble the final answer: grounded part, quotation, then marked knowledge."""
    lines = list(dict.fromkeys(l for l in map(_knowledge_line, knowledge) if l))
    block = strip_external_referrals("\n".join(lines))
    grounded = strip_external_referrals(grounded)
    if not grounded and not quotation:
        grounded = NO_GROUNDED_ANSWER
    parts = [grounded, quotation]
    if block:
        parts.append(f"{KNOWLEDGE_TITLE}\n{block}")
    return "\n\n".join(p.strip() for p in parts if p and p.strip())
