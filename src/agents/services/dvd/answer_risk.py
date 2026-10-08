"""How much an answer's audit can go wrong, to size the critic's reasoning effort.

On gpt-oss-20b an audit takes ~3.5 s with ``reasoning_effort=low`` and ~12.4 s with
``medium``. ``low`` misses more semantic defects, so it is used only for answers
where little can be wrong: a short statement from one or two sources of one
document edition, with no quantity, obligation or condition. Anything that the
rules below do not recognise as low-risk stays high-risk: when the classifier is
unsure, the audit keeps its full effort. If quality drops, widen these rules
rather than send every answer back to ``medium``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .document_reference import ADDRESS, DESIGNATION

# The verbatim quotation is application output, not a model claim to audit.
_QUOTATION = "Полная цитата:"
_LABEL = re.compile(r"\[(\d+)\]")
_EDITION = re.compile(r"\bред(?:акци[яи])?\.?\s*(?:от\s*)?[\d.]+", re.I)
_DIGIT = re.compile(r"\d")
_CONSTRAINT = re.compile(
    r"\bне\s+(?:менее|более|выше|ниже|превыша\w*|допуска\w*|разреша\w*|следует|долж\w*)"
    r"|\b(?:долж[ен]\w*|обязан\w*|обязательн\w*|следует|необходимо|требуется"
    r"|запрещ\w*|допуска\w*|разреш\w*|недопустим\w*|минимальн\w*|максимальн\w*"
    r"|предельн\w*|наименьш\w*|наибольш\w*)\b",
    re.I,
)
_CONDITION = re.compile(
    r"\b(?:за\s+исключением|кроме|если|при\s+условии|в\s+случае|в\s+случаях"
    r"|не\s+распростран\w*|исключени\w*|только\s+(?:для|при|в)|применя\w*\s+к)\b",
    re.I,
)
# An answer that the evidence lacks something is a conclusion about everything
# retrieved: a false «insufficient» is as harmful as a false norm.
_ABSENCE = re.compile(
    r"\b(?:нет|отсутству\w*|не\s+(?:найден\w*|содерж\w*|привед\w*|указан\w*|установлен\w*)"
    r"|недостаточно)\b",
    re.I,
)
_MAX_LOW_RISK_LINES = 6
_MAX_LOW_RISK_CHARS = 1500
_MAX_LOW_RISK_SOURCES = 2


@dataclass(frozen=True)
class AnswerRisk:
    level: Literal["low", "high"]
    reasons: tuple[str, ...] = ()

    @property
    def low(self) -> bool:
        return self.level == "low"


HIGH_RISK = AnswerRisk("high", ("unclassified",))


def _explanation(answer: str) -> str:
    return answer.split(_QUOTATION, 1)[0]


def _without_identifiers(text: str) -> str:
    """Drop citation labels, document designations, clause addresses and editions.

    Their digits identify a source; they are not quantities of a requirement.
    """
    text = _LABEL.sub(" ", text)
    text = DESIGNATION.sub(" ", text)
    text = ADDRESS.sub(" ", text)
    return _EDITION.sub(" ", text)


def assess_risk(
    answer: str,
    source_documents: dict[str, object] | None = None,
    *,
    intent: str = "norm",
    rejected_before: bool = False,
    context_incomplete: bool = False,
) -> AnswerRisk:
    """Classify ``answer`` for its audit.

    ``source_documents`` maps a source label (``"[1]"``) to the identity of its
    document edition, so citations of two documents or editions are recognised.
    """
    explanation = _explanation(answer)
    reasons: list[str] = []
    if rejected_before:
        reasons.append("previous_rejection")
    if context_incomplete:
        reasons.append("partial_context")
    if intent == "document_list":
        reasons.append("document_list")
    if _DIGIT.search(_without_identifiers(explanation)):
        reasons.append("numbers")
    if _CONSTRAINT.search(explanation):
        reasons.append("obligation")
    if _CONDITION.search(explanation):
        reasons.append("conditions")
    if _ABSENCE.search(explanation):
        reasons.append("absence_claim")
    lines = [line for line in explanation.splitlines() if line.strip()]
    if len(lines) > _MAX_LOW_RISK_LINES or len(explanation) > _MAX_LOW_RISK_CHARS:
        reasons.append("long_answer")
    cited = {f"[{n}]" for n in _LABEL.findall(explanation)}
    if not cited:
        # A claim that cannot be mapped to a source needs the full audit.
        reasons.append("no_citations")
    if len(cited) > _MAX_LOW_RISK_SOURCES:
        reasons.append("many_sources")
    if any(len(set(_LABEL.findall(line))) > 1 for line in lines):
        reasons.append("multi_source_claim")
    if source_documents:
        editions = {
            source_documents[label] for label in cited if label in source_documents
        }
        names = {
            edition[1] if isinstance(edition, tuple) and len(edition) > 1 else edition
            for edition in editions
        }
        if len(names) > 1:
            reasons.append("multiple_documents")
        elif len(editions) > 1:
            reasons.append("multiple_editions")
    return AnswerRisk("high" if reasons else "low", tuple(reasons))
