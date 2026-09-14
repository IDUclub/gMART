"""Small dependency-free process metrics for executable compliance checks."""

from __future__ import annotations

from collections import Counter, defaultdict
from threading import Lock
from typing import Any

from src.agents.services.service_entities.compliance import ComplianceResult


class ComplianceMetrics:
    """Collect aggregate, non-identifying counters for operational monitoring."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._templates: Counter[str] = Counter()
        self._verification: Counter[str] = Counter()
        self._planner: Counter[str] = Counter()
        self._downstream_errors: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._fill_rates: dict[str, list[float]] = defaultdict(list)

    def observe(
        self,
        result: ComplianceResult,
        *,
        planner_status: str,
        timings_ms: dict[str, float],
    ) -> None:
        with self._lock:
            self._templates[f"{result.template}@{result.template_version}"] += 1
            self._verification[result.verification_status] += 1
            self._planner[planner_status] += 1
            for stage, value in timings_ms.items():
                self._durations[stage].append(round(float(value), 3))
            for requirement in result.resolved_requirements:
                if (
                    requirement.requirement_type == "attribute"
                    and requirement.fill_rate is not None
                ):
                    self._fill_rates[requirement.role].append(requirement.fill_rate)

    def observe_downstream_error(self, source: str) -> None:
        if source not in {"urban_api", "idu_mcp"}:
            raise ValueError(f"unknown downstream source: {source}")
        with self._lock:
            self._downstream_errors[source] += 1

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
        total = sum(values)
        return {
            "count": len(values),
            "sum": round(total, 3),
            "avg": round(total / len(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scope": "process",
                "norms_by_template": dict(sorted(self._templates.items())),
                "verification_status": dict(sorted(self._verification.items())),
                "planner_status": dict(sorted(self._planner.items())),
                "duration_ms": {
                    key: self._distribution(values)
                    for key, values in sorted(self._durations.items())
                },
                "fill_rate_by_role": {
                    key: self._distribution(values)
                    for key, values in sorted(self._fill_rates.items())
                },
                "downstream_errors": {
                    "urban_api": self._downstream_errors["urban_api"],
                    "idu_mcp": self._downstream_errors["idu_mcp"],
                },
            }


COMPLIANCE_METRICS = ComplianceMetrics()
