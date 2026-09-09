from __future__ import annotations

from typing import Any


class DvdContextBuilder:
    """
    Formats IDU_DVD search hits into a numbered, citable context string for the LLM.

    Each hit becomes a block headed by ``[N] <document>, ред. <version>, п. <numbering> — <breadcrumb>``
    so the answering model can ground its response and cite sources by number and clause.
    """

    MAX_FRAGMENT_CHARS = 1500

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
        header = ", ".join(header_bits)
        if breadcrumb := hit.get("breadcrumb"):
            header += f" — {breadcrumb}"

        body = (
            hit.get("table_html") or hit.get("context") or hit.get("text") or ""
        ).strip()
        if len(body) > self.MAX_FRAGMENT_CHARS:
            body = body[: self.MAX_FRAGMENT_CHARS].rstrip() + " […]"
        return f"{header}\n{body}" if body else header
