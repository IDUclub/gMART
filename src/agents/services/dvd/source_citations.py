"""Replace source labels with the document and clause they stand for.

The model writes ``[N]`` after a statement: the labels are local to one retrieval
and let the critic check each line against its fragment. The user reads the
document instead: ``[2]`` becomes «(СП 55.13330.2016, п. 9.18)». Labels inside
quoted source text (``>`` lines) are the document's own bibliography references
and stay as they are.
"""

from __future__ import annotations

import re
from typing import Any

from . import flags
from .dvd_context import DvdContextBuilder

_KIND = {
    "article": "статья",
    "section": "раздел",
    "chapter": "глава",
    "appendix": "приложение",
    "table": "таблица",
}
_RUN = re.compile(r"[ \t]*\[\d+\](?:\s*(?:,|;|и)?\s*\[\d+\])*")
_LABEL = re.compile(r"\[(\d+)\]")
_QUOTE_HEADER = re.compile(r"^\[\d+\]\s+")
_QUOTATION = "Полная цитата:"


def enabled() -> bool:
    return flags.enabled(flags.READABLE_CITATIONS)


def _document(hit: dict[str, Any]) -> str:
    name = " ".join(str(hit.get("name") or "").split())
    version = str(hit.get("version") or "").strip()
    if version and version not in name:
        name += f", {version}" if version.startswith("ред") else f", ред. {version}"
    return name


def _address(hit: dict[str, Any]) -> str:
    if number := str(hit.get("numbering") or "").strip():
        return f"{_KIND.get(hit.get('type'), 'п.')} {number}"
    path = [p for p in hit.get("structure_path") or [] if str(p).strip()]
    if not path:
        path = [p for p in (hit.get("breadcrumb") or "").split(" / ") if p.strip()]
    title = hit.get("fragment_name") or (path[-1] if path else "")
    title = " ".join(str(title).split())
    return f"раздел «{title}»" if title else ""


def citations(hits: list[dict[str, Any]]) -> dict[str, list[str]]:
    """``{"[N]": [document, address]}`` for the labels of ``build_context(hits)``."""
    return {
        f"[{index}]": [_document(hit), _address(hit)]
        for index, hit in enumerate(DvdContextBuilder.ordered_hits(hits), 1)
    }


def _render(labels: list[str], refs: dict[str, list[str]]) -> str:
    groups: dict[str, list[str]] = {}
    for label in dict.fromkeys(labels):
        if label not in refs:
            continue
        document, address = refs[label]
        addresses = groups.setdefault(document, [])
        if address and address not in addresses:
            addresses.append(address)
    parts = [
        ", ".join(filter(None, [doc, *addresses])) for doc, addresses in groups.items()
    ]
    parts = [p for p in parts if p]
    return f" ({'; '.join(parts)})" if parts else ""


def readable(text: str, refs: dict[str, list[str]] | None) -> str:
    """Write each run of labels as one parenthesised reference to its sources.

    A label without a known source is dropped: it points at nothing the user
    can find. Without ``refs`` the text is returned unchanged.
    """
    if not refs or not text or not enabled():
        return text
    lines, quoting = [], False
    for line in text.splitlines():
        if line.strip() == _QUOTATION:
            quoting = True
        if line.lstrip().startswith(">"):
            lines.append(line)
        elif quoting and _QUOTE_HEADER.match(line):
            # The quotation header already names the document and the clause.
            lines.append(_QUOTE_HEADER.sub("", line))
        else:
            lines.append(
                _RUN.sub(
                    lambda m: _render(
                        [f"[{n}]" for n in _LABEL.findall(m.group())], refs
                    ),
                    line,
                )
            )
    return "\n".join(lines)
