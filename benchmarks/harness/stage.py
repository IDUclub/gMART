"""Stage registry: one benchmarked LLM step of a gMART agent.

A stage takes a case's ``input`` and a :class:`StageContext` and runs gMART's own
code for that step up to its model call; the export records that call's request.

``kind="auto"`` stages answer with JSON scored by checks alone; ``kind="manual"``
stages produce free text that also needs a human look in the runner's ``review.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

StageKind = Literal["auto", "manual"]


@dataclass
class StageContext:
    llm: Any
    #: The served model id, passed to gMART code exactly as the agents would.
    model: str
    temperature: float


StageFn = Callable[[dict[str, Any], StageContext], Awaitable[Any]]


@dataclass(frozen=True)
class Stage:
    id: str
    kind: StageKind
    description: str
    fn: StageFn
    #: What the stage measures, for the report: which gMART code runs and what is
    #: left out (deterministic post-processing that would mask a model mistake).
    scope: str = ""


STAGES: dict[str, Stage] = {}


def stage(
    stage_id: str, *, kind: StageKind = "auto", description: str, scope: str = ""
) -> Callable[[StageFn], StageFn]:
    def register(fn: StageFn) -> StageFn:
        if stage_id in STAGES:
            raise ValueError(f"stage {stage_id!r} registered twice")
        STAGES[stage_id] = Stage(
            id=stage_id, kind=kind, description=description, fn=fn, scope=scope
        )
        return fn

    return register


def load_stages() -> dict[str, Stage]:
    # Importing the modules registers their stages.
    from benchmarks.stages import (  # noqa: F401
        answers,
        critics,
        mappings,
        planners,
    )

    return STAGES
