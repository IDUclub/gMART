"""Deterministic line audit for when the critic cannot finish within its limit.

It proves much less than the critic, so it errs towards rejection: a line passes
only when it cites known sources, every number it states occurs in them, and most
of its content words do. Anything else counts as unconfirmed and is sorted out
of the grounded answer. An invented norm shares neither its figures nor its
wording with the fragment it cites, so it does not pass.
"""

from __future__ import annotations

import re

from src.agents.services.service_entities.dvd_plan import AuditedClaim

from .document_reference import DESIGNATION
from .dvd_context import source_records

_LABEL = re.compile(r"\[\d+\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)*")
_WORD = re.compile(r"[а-яa-z]{5,}")
_STEM = 5
_MIN_SHARED_WORDS = 0.6


def _normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return re.sub(r"(\d),(\d)", r"\1.\2", text)


def _stems(text: str) -> set[str]:
    return {word[:_STEM] for word in _WORD.findall(text)}


def audit_line(line: str, records: dict[str, tuple[str, str]]) -> str:
    labels = set(_LABEL.findall(line))
    if not labels or not labels <= set(records):
        return "insufficient"
    source = _normalize(" ".join(" ".join(records[label]) for label in labels))
    claim = _normalize(DESIGNATION.sub(" ", _LABEL.sub(" ", line)))
    if not set(_NUMBER.findall(claim)) <= set(_NUMBER.findall(source)):
        return "insufficient"
    words = _stems(claim)
    if not words or len(words & _stems(source)) / len(words) < _MIN_SHARED_WORDS:
        return "insufficient"
    return "supported"


def audit(lines: list[str], context: str) -> list[AuditedClaim]:
    """Audit answer ``lines`` (as the critic selects them) against ``context``."""
    records = {k: v for k, v in source_records(context).items() if k != "unlabelled"}
    return [AuditedClaim(text=line, status=audit_line(line, records)) for line in lines]
