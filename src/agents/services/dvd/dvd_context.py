from __future__ import annotations

from typing import Any


class DvdContextBuilder:
    """
    Formats IDU_DVD search hits into a numbered, citable context string for the LLM.

    Each hit becomes a block headed by ``[N] <document>, ред. <version>, п. <numbering> — <breadcrumb>``
    so the answering model can ground its response and cite sources by number and clause.
    """

    def build_context(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return ""
        return "\n\n".join(
            self._format_hit(index, hit) for index, hit in enumerate(hits, start=1)
        )

    def _format_hit(self, index: int, hit: dict[str, Any]) -> str:
        name = hit.get("name") or "Документ без названия"
        header_bits = [f"[{index}] {name}"]
        if version := hit.get("version"):
            header_bits.append(f"ред. {version}")
        if numbering := hit.get("numbering"):
            header_bits.append(f"п. {numbering}")
        if fragment_name := hit.get("fragment_name"):
            header_bits.append(fragment_name)
        if node_id := hit.get("id"):
            header_bits.append(f"node_id={node_id}")
        header = ", ".join(header_bits)
        if breadcrumb := hit.get("breadcrumb"):
            header += f" — {breadcrumb}"

        body = (
            hit.get("table_html") or hit.get("context") or hit.get("text") or ""
        ).strip()
        # Never truncate a target or its descendants. Oversized contexts are processed
        # by DvdContextReducer in bounded parallel requests, retaining source labels.
        target = (hit.get("text") or "").strip()
        if target and target not in body and not hit.get("table_html"):
            body = target + "\n" + body
        return f"{header}\n{body}" if body else header
