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


def merged_sources(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Norms merged into this check, excluding the executed one.

    Norms are distinct by ``restriction_id``: two clauses of one document without a
    clause number share a label, yet both were merged and must both be counted.
    """
    own = source.get("restriction_id")
    seen = {own or source_reference(source)}
    merged = []
    for item in source.get("equivalent_sources") or []:
        key = item.get("restriction_id") or source_reference(item)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


def grouped_references(sources: list[dict[str, Any]]) -> str:
    """``СП 2.4.3648 (2 нормы); СП 42.13330, п. 7.1`` — one label per citation."""
    counts: dict[str, int] = {}
    for item in sources:
        label = source_reference(item)
        counts[label] = counts.get(label, 0) + 1
    return "; ".join(
        label if count == 1 else f"{label} ({count} {norms_word(count)})"
        for label, count in counts.items()
    )


def norms_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "норма"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "нормы"
    return "норм"
