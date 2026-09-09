import json


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
        "sum_absolute_scenario_project": "Суммарный абсолютный эффект (объекты проекта)",
        "average_absolute_scenario_project": "Средний абсолютный эффект (объекты проекта)",
        "median_absolute_scenario_project": "Медианный абсолютный эффект (объекты проекта)",
        "average_index_scenario_project": "Средний индексный эффект (объекты проекта)",
        "median_index_scenario_project": "Медианный индексный эффект (объекты проекта)",
    }

    # Known layer names returned by CalculateObjectEffects
    _LAYER_LABELS: dict[str, str] = {
        "buildings": "здания",
        "services": "объекты сервиса",
        "links": "связи",
    }

    # Strict column contract for the services provision summary table.
    # Keys and labels are fixed in code so the LLM can never rename them.
    SUMMARY_TABLE_COLUMNS: list[dict[str, str]] = [
        {"key": "service", "label": "Сервис"},
        {"key": "capacity", "label": "Вместимость (чел)"},
        {"key": "demand", "label": "Спрос (чел)"},
        {"key": "deficit", "label": "Дефицит (чел)"},
        {"key": "surplus", "label": "Профицит (чел)"},
        {"key": "balance", "label": "Баланс (чел)"},
    ]

    METRIC_TABLE_COLUMNS: list[dict[str, str]] = [
        {"key": "metric", "label": "Показатель"},
        {"key": "value", "label": "Значение"},
    ]

    # Strict row labels for the single-service provision metrics table.
    _PROVISION_METRIC_LABELS: list[tuple[str, str]] = [
        ("services_count", "Количество объектов сервиса"),
        ("total_capacity", "Вместимость (чел)"),
        ("total_demand", "Спрос (чел)"),
        (
            "satisfied_demand_within",
            "Удовлетворённый спрос в нормативной доступности (чел)",
        ),
        (
            "satisfied_demand_without",
            "Удовлетворённый спрос вне нормативной доступности (чел)",
        ),
        ("unsatisfied_demand", "Неудовлетворённый спрос (чел)"),
        ("deficit", "Дефицит (чел)"),
        ("surplus", "Профицит (чел)"),
        ("balance", "Баланс (чел)"),
        ("average_provision_value", "Средняя обеспеченность (0–1)"),
        ("median_provision_value", "Медианная обеспеченность (0–1)"),
    ]

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

    # ------------------------------------------------------------------
    # Strict tables (fixed columns rendered by code, not by the LLM)
    # ------------------------------------------------------------------

    def build_effects_pivot_table(
        self, effects_result: dict, service_name: str
    ) -> dict | None:
        """
        Build a strict metric/value table from the CalculateObjectEffects pivot.
        Returns:
            dict | None: Table event content, or None when pivot is missing.
        """
        pivot = effects_result.get("pivot") or {}
        rows = [
            {"metric": label, "value": self._round(pivot[key])}
            for key, label in self._PIVOT_LABELS.items()
            if key in pivot and pivot[key] is not None
        ]
        if not rows:
            return None
        return {
            "name": "effects_pivot",
            "title": f"Эффекты обеспеченности: {service_name}",
            "columns": self.METRIC_TABLE_COLUMNS,
            "rows": rows,
        }

    def build_provision_metrics_table(self, summary: dict, service_name: str) -> dict:
        """
        Build a strict metric/value table for a single-service provision result.
        Args:
            summary (dict): ProvisionSummarySchema payload from CalculateServicesProvision.
            service_name (str): Human-readable service name.
        Returns:
            dict: Table event content.
        """
        rows = [
            {"metric": label, "value": self._round(summary[key])}
            for key, label in self._PROVISION_METRIC_LABELS
            if key in summary and summary[key] is not None
        ]
        return {
            "name": "provision_metrics",
            "title": f"Обеспеченность сервисом «{service_name}»",
            "columns": self.METRIC_TABLE_COLUMNS,
            "rows": rows,
        }

    def build_summary_table(self, services_result: dict) -> dict:
        """
        Build the strict deficit/surplus summary table over several services.
        Args:
            services_result (dict): CalculateServicesProvision result:
                {"services": {service_type_id: {name, summary, layers, error}}}.
        Returns:
            dict: Table event content with rows sorted by deficit (worst first).
        """
        rows = []
        for service in (services_result.get("services") or {}).values():
            summary = service.get("summary")
            if not summary:
                continue
            rows.append(
                {
                    "service": service.get("name", ""),
                    "capacity": summary.get("total_capacity"),
                    "demand": summary.get("total_demand"),
                    "deficit": summary.get("deficit"),
                    "surplus": summary.get("surplus"),
                    "balance": summary.get("balance"),
                }
            )
        rows.sort(key=lambda row: row.get("deficit") or 0, reverse=True)
        return {
            "name": "provision_summary",
            "title": "Сводка обеспеченности сервисами",
            "columns": self.SUMMARY_TABLE_COLUMNS,
            "rows": rows,
        }

    # ------------------------------------------------------------------
    # LLM contexts for the new provision modes
    # ------------------------------------------------------------------

    def build_provision_context(self, summary: dict, service_name: str) -> str:
        """Format a single-service provision summary as an LLM context string."""
        lines = [f"Текущая обеспеченность сервисом «{service_name}»:"]
        for key, label in self._PROVISION_METRIC_LABELS:
            if key in summary and summary[key] is not None:
                lines.append(f"  - {label}: {self._round(summary[key])}")
        return "\n".join(lines)

    def build_summary_context(self, services_result: dict) -> str:
        """Format a multi-service provision result as an LLM context string."""
        lines = ["Сводка обеспеченности сервисами (текущее состояние сценария):"]
        failed: list[str] = []
        for service in (services_result.get("services") or {}).values():
            name = service.get("name", "")
            summary = service.get("summary")
            if not summary:
                failed.append(name)
                continue
            values = {
                label: self._round(summary[key])
                for key, label in self._PROVISION_METRIC_LABELS
                if key in summary and summary[key] is not None
            }
            lines.append(f"\n{name}: {json.dumps(values, ensure_ascii=False)}")
        if failed:
            lines.append(
                "\nНе удалось рассчитать обеспеченность для сервисов: "
                + ", ".join(failed)
            )
        return "\n".join(lines)

    @staticmethod
    def _round(value):
        if isinstance(value, float):
            return round(value, 3)
        return value
