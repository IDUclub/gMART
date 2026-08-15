from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ScenarioDataActionKind(StrEnum):
    CALL_TOOL = "call_tool"
    FINAL_ANSWER = "final_answer"


class ScenarioDataAction(BaseModel):
    action: ScenarioDataActionKind
    group: str | None = None
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    layer_name: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> "ScenarioDataAction":
        if self.action == ScenarioDataActionKind.CALL_TOOL:
            if not self.group or not self.tool_name:
                raise ValueError("call_tool action requires group and tool_name")
        return self
