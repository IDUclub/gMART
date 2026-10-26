"""The critic's view of the evidence an answer was drafted from.

The answer is drafted over the prepared context of a retrieval. Reducing that
context again for the audit (``prepare(question + draft, context)``) costs two LLM
calls per part and is paid on every review round. When the prepared context does
not fit the audit's budget, the audit reads only the sources the answer cites: each
claim line carries the labels of its sources, and the application, not a model,
copies those sources verbatim.

The full evidence is reopened (reduced as before) whenever a claim cannot be mapped
to a source: a line without a label, a label that is not a source, or a statement
that the evidence lacks something, which is a claim about every source.
``DVD_CRITIC_CONTEXT`` selects the policy: ``auto`` (default) as above, ``cited``
to audit cited sources even when everything fits, ``full`` for the previous
behaviour.
"""

from __future__ import annotations

import os
import re

from .answer_risk import _ABSENCE
from .dvd_context import SOURCE_SEPARATOR, source_records

_QUOTATION = "Полная цитата:"
_LABEL = re.compile(r"\[\d+\]")
POLICIES = {"auto", "cited", "full"}


def critic_context_policy() -> str:
    policy = (os.getenv("DVD_CRITIC_CONTEXT") or "auto").strip().lower()
    return policy if policy in POLICIES else "auto"


def _claim_lines(explanation: str) -> list[str]:
    # The same lines the critic audits; layout lines and headings assert nothing.
    from .dvd_reasoning import AnswerCritic

    return AnswerCritic._claim_texts(explanation)


def cited_context(context: str, answer: str) -> str | None:
    """The sources ``answer`` cites, verbatim and in context order.

    ``None`` when the audit needs every source: a claim without a source label, a
    label that names no source, or a statement that something is absent.
    """
    explanation = answer.split(_QUOTATION, 1)[0]
    if _ABSENCE.search(explanation):
        return None
    lines = _claim_lines(explanation)
    if not lines or any(not _LABEL.search(line) for line in lines):
        return None
    records = source_records(context)
    cited = {label for line in lines for label in _LABEL.findall(line)}
    if not cited <= set(records) - {"unlabelled"}:
        return None
    if cited == set(records):
        return None  # Nothing to leave out.
    return "".join(
        header + "\n" + body + SOURCE_SEPARATOR
        for label, (header, body) in records.items()
        if label in cited
    )
