"""Grounded task and step state for the scenario-data execution loop."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    PlanStep,
)


class AttemptStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Keep persisted execution state useful without letting tool payloads dominate it."""

    if depth >= 4:
        return "…"
    if isinstance(value, dict):
        return {
            str(key): _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:30]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, depth=depth + 1) for item in value[:40]]
    if isinstance(value, str):
        return value[:1000]
    return value


class StepAttemptContext(BaseModel):
    attempt_id: str
    revision: int
    step_id: str
    purpose: str
    kind: str
    group: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    satisfies: list[str] = Field(default_factory=list)
    status: AttemptStatus = AttemptStatus.RUNNING
    error_type: str | None = None
    error: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class ScenarioExecutionContext(BaseModel):
    """Observable execution memory, without private model reasoning."""

    objective: str
    user_query: str
    scenario_id: int | None = None
    project_id: int | None = None
    requirements: list[dict[str, Any]] = Field(default_factory=list)
    required_output: dict[str, Any] = Field(default_factory=dict)
    verified_mappings: list[dict[str, Any]] = Field(default_factory=list)
    attempts: list[StepAttemptContext] = Field(default_factory=list)

    @classmethod
    def from_acquisition(
        cls,
        acquisition: AcquisitionPlan,
        *,
        user_query: str,
        scenario_id: int | None,
        project_id: int | None,
    ) -> "ScenarioExecutionContext":
        return cls(
            objective=acquisition.objective,
            user_query=user_query,
            scenario_id=scenario_id,
            project_id=project_id,
            requirements=[
                requirement.model_dump(mode="json")
                for requirement in acquisition.requirements
            ],
            required_output=acquisition.required_output.model_dump(mode="json"),
        )

    def update_acquisition(self, acquisition: AcquisitionPlan) -> None:
        self.objective = acquisition.objective
        self.requirements = [
            requirement.model_dump(mode="json")
            for requirement in acquisition.requirements
        ]
        self.required_output = acquisition.required_output.model_dump(mode="json")

    def update_mappings(self, mappings: list[dict[str, Any]]) -> None:
        self.verified_mappings = [
            {
                "domain": mapping.get("domain"),
                "direction": mapping.get("direction"),
                "source_tool": mapping.get("source_tool"),
                "matches": [
                    {"id": match.get("id"), "name": match.get("name")}
                    for match in (mapping.get("matches") or [])[:25]
                    if isinstance(match, dict)
                ],
            }
            for mapping in mappings[-8:]
            if isinstance(mapping, dict)
        ]

    def start_step(
        self,
        revision: int,
        step: PlanStep,
        arguments: dict[str, Any],
    ) -> StepAttemptContext:
        return self.start_attempt(
            revision=revision,
            step_id=step.step_id,
            purpose=step.purpose,
            kind=step.kind.value,
            group=step.group,
            tool_name=step.tool_name,
            arguments=arguments,
            satisfies=step.satisfies,
        )

    def start_attempt(
        self,
        *,
        revision: int,
        step_id: str,
        purpose: str,
        kind: str,
        group: str | None,
        tool_name: str,
        arguments: dict[str, Any],
        satisfies: list[str],
    ) -> StepAttemptContext:
        attempt = StepAttemptContext(
            attempt_id=f"r{revision}:{step_id}:{len(self.attempts) + 1}",
            revision=revision,
            step_id=step_id,
            purpose=purpose,
            kind=kind,
            group=group,
            tool_name=tool_name,
            arguments=_bounded_value(dict(arguments)),
            satisfies=list(satisfies),
        )
        self.attempts.append(attempt)
        return attempt

    @staticmethod
    def complete_attempt(
        attempt: StepAttemptContext, observation: dict[str, Any]
    ) -> None:
        attempt.status = AttemptStatus.COMPLETED
        attempt.evidence = _bounded_value(
            {
                key: observation.get(key)
                for key in ("summary", "layer_count", "table_count", "aggregate")
                if observation.get(key) is not None
            }
        )

    @staticmethod
    def fail_attempt(attempt: StepAttemptContext, error: Exception) -> None:
        attempt.status = AttemptStatus.FAILED
        attempt.error_type = type(error).__name__
        attempt.error = str(error)[:1000]

    @staticmethod
    def skip_attempt(attempt: StepAttemptContext, error: Exception) -> None:
        attempt.status = AttemptStatus.SKIPPED
        attempt.error_type = type(error).__name__
        attempt.error = str(error)[:1000]

    def attempt_observation(self, attempt: StepAttemptContext) -> dict[str, Any]:
        arguments = json.dumps(
            attempt.arguments, ensure_ascii=False, sort_keys=True, default=str
        )[:2000]
        return {
            "context": "Контекст исполняемого шага",
            "summary": (
                f"{attempt.status.value}: {attempt.group}.{attempt.tool_name}; "
                f"цель шага: {attempt.purpose}; аргументы: {arguments}; "
                f"ошибка: {attempt.error or 'нет'}"
            )[:3500],
            "step_context": attempt.model_dump(mode="json"),
        }

    def planner_snapshot(
        self,
        *,
        urban_calls: int,
        workspace_calls: int,
        replans: int,
    ) -> dict[str, Any]:
        covered = {
            requirement
            for attempt in self.attempts
            if attempt.status == AttemptStatus.COMPLETED
            for requirement in attempt.satisfies
        }
        requirement_ids = [
            str(requirement.get("requirement_id"))
            for requirement in self.requirements
            if requirement.get("requirement_id")
        ]
        return {
            "task": {
                "objective": self.objective,
                "user_query": self.user_query,
                "scenario_id": self.scenario_id,
                "project_id": self.project_id,
                "requirements": self.requirements,
                "required_output": self.required_output,
            },
            "verified_mappings": self.verified_mappings,
            "attempts": [
                attempt.model_dump(mode="json") for attempt in self.attempts[-12:]
            ],
            "open_requirements": [
                requirement
                for requirement in requirement_ids
                if requirement not in covered
            ],
            "budgets_used": {
                "urban_calls": urban_calls,
                "workspace_calls": workspace_calls,
                "replans": replans,
            },
        }

    def failure_note(self, reasons: list[str]) -> str:
        attempted = [
            f"{attempt.group}.{attempt.tool_name}"
            + (f" — {attempt.error}" if attempt.error else "")
            for attempt in self.attempts
            if attempt.status in {AttemptStatus.FAILED, AttemptStatus.SKIPPED}
        ]
        completed = {
            requirement
            for attempt in self.attempts
            if attempt.status == AttemptStatus.COMPLETED
            for requirement in attempt.satisfies
        }
        all_requirements = {
            str(requirement.get("requirement_id"))
            for requirement in self.requirements
            if requirement.get("requirement_id")
        }
        open_requirements = sorted(all_requirements - completed)
        chunks = ["Не удалось получить результат полностью."]
        if attempted:
            chunks.append("Проверенные пути: " + "; ".join(attempted[-6:]) + ".")
        if reasons:
            chunks.append("Нерешённые проблемы: " + " ".join(reasons))
        if open_requirements:
            chunks.append(
                "Не закрыты требования: " + ", ".join(open_requirements) + "."
            )
        return " ".join(chunks)
