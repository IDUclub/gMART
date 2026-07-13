from __future__ import annotations

from typing import Any


class NormGraphContextBuilder:
    """
    Formats NormGraph restriction hits (+ optional neighbours/fallback/conflicts) into a
    numbered, citable context string for the LLM.

    Each restriction becomes a block headed by
    ``[N] <document>, ред. <version>, п. <numbering> (id: <restriction_id>) — <breadcrumb>``
    followed by the triple (``subject → object | kind | value``) and the verbatim clause
    excerpt, so the answering model can cite by number and by ``restriction_id``. A separate
    "Обнаруженные противоречия" block lists any ``list_conflicts`` results so the critic can
    require them to be surfaced in the answer.
    """

    MAX_EXCERPT_CHARS = 1200

    def build_context(
        self,
        hits: list[dict[str, Any]],
        neighbors: list[dict[str, Any]] | None = None,
        dvd_fallback: list[dict[str, Any]] | None = None,
        conflicts: list[dict[str, Any]] | None = None,
    ) -> str:
        blocks: list[str] = []
        index = 1

        for hit in hits or []:
            blocks.append(self._format_restriction(index, hit))
            index += 1

        for neighbor in neighbors or []:
            restriction = neighbor.get("restriction") or {}
            relation = neighbor.get("relation") or "связано"
            blocks.append(
                self._format_restriction(index, restriction, relation=relation)
            )
            index += 1

        for hit in dvd_fallback or []:
            blocks.append(self._format_dvd_fallback(index, hit))
            index += 1

        context = "\n\n".join(blocks)

        if conflicts:
            conflicts_block = self._format_conflicts(conflicts)
            context = f"{context}\n\n{conflicts_block}" if context else conflicts_block

        return context

    def _format_restriction(
        self, index: int, hit: dict[str, Any], relation: str | None = None
    ) -> str:
        provenance = hit.get("provenance") or {}
        name = provenance.get("name") or "Документ без названия"
        header_bits = [f"[{index}] {name}"]
        if version := provenance.get("version"):
            header_bits.append(f"ред. {version}")
        if numbering := provenance.get("numbering"):
            header_bits.append(f"п. {numbering}")
        restriction_id = hit.get("id")
        header = ", ".join(header_bits)
        if restriction_id:
            header += f" (id: {restriction_id})"
        if relation:
            header += f" [{relation}]"
        if breadcrumb := provenance.get("breadcrumb"):
            header += f" — {breadcrumb}"

        triple = self._format_triple(hit)
        excerpt = (hit.get("extraction_text") or "").strip()
        if len(excerpt) > self.MAX_EXCERPT_CHARS:
            excerpt = excerpt[: self.MAX_EXCERPT_CHARS].rstrip() + " […]"

        body_lines = [line for line in (triple, excerpt) if line]
        body = "\n".join(body_lines)
        return f"{header}\n{body}" if body else header

    @staticmethod
    def _format_triple(hit: dict[str, Any]) -> str:
        subject = hit.get("subject") or "?"
        obj = hit.get("object") or "?"
        kind = hit.get("kind") or "?"
        triple = f"{subject} → {obj} | {kind}"
        value = hit.get("value") or {}
        value_bits = [
            str(value[key])
            for key in ("operator", "number", "unit")
            if value.get(key) is not None
        ]
        if value_bits:
            triple += " | " + " ".join(value_bits)
        if condition := value.get("condition"):
            triple += f" (условие: {condition})"
        return triple

    def _format_dvd_fallback(self, index: int, hit: dict[str, Any]) -> str:
        name = hit.get("name") or "Документ без названия"
        header_bits = [f"[{index}] {name}"]
        if numbering := hit.get("numbering"):
            header_bits.append(f"п. {numbering}")
        header = ", ".join(header_bits) + " (не структурировано в графе)"
        text = (hit.get("text") or "").strip()
        if len(text) > self.MAX_EXCERPT_CHARS:
            text = text[: self.MAX_EXCERPT_CHARS].rstrip() + " […]"
        return f"{header}\n{text}" if text else header

    @staticmethod
    def _format_conflicts(conflicts: list[dict[str, Any]]) -> str:
        lines = ["Обнаруженные противоречия:"]
        for conflict in conflicts:
            restriction = conflict.get("restriction") or {}
            other = conflict.get("other") or {}
            severity = conflict.get("severity") or "possible"
            reason = conflict.get("reason") or ""
            lines.append(
                f"- {NormGraphContextBuilder._short_ref(restriction)} vs "
                f"{NormGraphContextBuilder._short_ref(other)} "
                f"[{severity}]: {reason}"
            )
        return "\n".join(lines)

    @staticmethod
    def _short_ref(restriction: dict[str, Any]) -> str:
        provenance = restriction.get("provenance") or {}
        name = provenance.get("name") or "?"
        numbering = provenance.get("numbering") or "?"
        restriction_id = restriction.get("id") or "?"
        return f"{name} п.{numbering} (id: {restriction_id})"
