"""Human-readable normative references for compliance layers and answers."""

from __future__ import annotations

import re
from typing import Any


def source_reference(source: dict[str, Any]) -> str:
    name = " ".join((source.get("document_name") or "").split())
    # Keep the designation when the graph supplies a full SP document title.
    code = re.search(r"\bСП\s*(\d+(?:\.\d+)+)\b", name, re.IGNORECASE)
    label = f"СП {code.group(1)}" if code else name or "Источник не указан"
    clause = " ".join((source.get("clause_number") or "").split())
    if clause:
        label += f", п. {clause}"
    return label


def source_references(source: dict[str, Any]) -> list[str]:
    """Retain citations for equivalent checks without repeating the same clause."""
    sources = [source, *(source.get("equivalent_sources") or [])]
    return list(dict.fromkeys(source_reference(item) for item in sources))
