class ProvisionContextBuilder:
    _PIVOT_LABELS = {
        "sum_absolute_total": "Суммарное абсолютное покрытие (всего)",
        "average_absolute_total": "Среднее абсолютное покрытие (всего)",
        "median_absolute_total": "Медианное абсолютное покрытие (всего)",
        "average_index_total": "Средний индекс покрытия (всего)",
        "median_index_total": "Медианный индекс покрытия (всего)",
        "sum_absolute_within": "Суммарное абсолютное покрытие (в пределах проекта)",
        "average_absolute_within": "Среднее абсолютное покрытие (в пределах проекта)",
        "median_absolute_within": "Медианное абсолютное покрытие (в пределах проекта)",
    }

    # Known layer names returned by CalculateObjectEffects
    _LAYER_LABELS: dict[str, str] = {
        "buildings": "здания",
        "services": "объекты сервиса",
        "links": "связи",
    }

    def build_context(self, effects_result: dict, service_name: str) -> str:
        """
        Format the effects result as an LLM context string.

        If the MCP server included a pre-formatted ``text_pivot`` (generated
        server-side via ``form_llm_context``), it is used directly to avoid
        duplicating expensive computation.  Otherwise the context is assembled
        from the structured ``pivot`` dict and layer counts.

        Args:
            effects_result (dict): Raw result from CalculateObjectEffects.
                Expected keys: before_prove_data, after_prove_data, effects,
                pivot, text_pivot (optional).
            service_name (str): Human-readable service name for context framing.
        Returns:
            str: Human-readable context for LLM analysis.
        """
        header = f"Анализ эффектов обеспеченности услугой «{service_name}»:"

        # Fast path: server already produced an LLM-ready context
        if text_pivot := effects_result.get("text_pivot"):
            return f"{header}\n\n{text_pivot}"

        # Slow path: build from structured data
        lines = [header]
        pivot = effects_result.get("pivot") or {}
        if pivot:
            lines.append("\nСводные показатели:")
            for key, label in self._PIVOT_LABELS.items():
                if key in pivot and pivot[key] is not None:
                    lines.append(f"  - {label}: {pivot[key]}")
        else:
            lines.append("\nСводные показатели недоступны.")

        for section_label, key in (
            ("До реализации проекта", "before_prove_data"),
            ("После реализации проекта", "after_prove_data"),
        ):
            summary = self._summarize_layers(
                section_label, effects_result.get(key) or {}
            )
            if summary:
                lines.append(summary)

        effects_fc = effects_result.get("effects")
        if isinstance(effects_fc, dict):
            count = len(effects_fc.get("features") or [])
            lines.append(f"\nСлой эффектов содержит {count} объектов.")

        return "\n".join(lines)

    def _summarize_layers(self, label: str, group: dict) -> str:
        if not group:
            return ""
        parts = [f"\n{label}:"]
        for name, fc in group.items():
            if isinstance(fc, dict):
                count = len(fc.get("features") or [])
                readable = self._LAYER_LABELS.get(name, name)
                parts.append(f"  - {readable}: {count} объектов")
        return "\n".join(parts)
