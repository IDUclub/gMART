"""The unchanged A2A executor serves the ADK workflow with the same frames.

``RestrictionAgentExecutor`` (python-a2a, protocol 0.3.0, Synapse card) is
wired once to the current service and once to the ADK one; the streamed A2A
frames and the stored task must match, apart from generated ids and times.
"""

from __future__ import annotations

from typing import Any

import pytest

from research.adk_pilot.tests.fakes import (
    SCENARIO_ID,
    Scenario,
    adk_service,
    legacy_service,
    plan_json,
)
from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore

VOLATILE = {"messageId", "timestamp"}


def _stable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def _params(query: str) -> dict:
    return {
        "id": "task-1",
        "contextId": "ctx-1",
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": query}],
            "metadata": {"scenario_id": SCENARIO_ID},
        },
        "metadata": {"model": "test-model", "temperature": 0.7},
    }


class _LegacyEntry:
    """Calls the inner pipeline the way the public wrapper does after auth."""

    def __init__(self, service):
        self.service = service

    def run_restriction_execution_pipline(self, *, token, **kwargs):
        return self.service._run_restriction_execution_pipline(
            token_ref=[token], **kwargs
        )


@pytest.mark.parametrize(
    "scenario",
    [
        Scenario(
            "Какие жилые дома попадают в зону 100 м от школ? Выведи объекты.",
            [plan_json("restrictions")],
        ),
        Scenario("Покажи зону", [plan_json("needs_clarification")]),
    ],
    ids=["restrictions", "clarification"],
)
async def test_a2a_frames_match(scenario):
    async def stream(service) -> tuple[list[dict], dict]:
        store = A2ATaskStore()
        executor = RestrictionAgentExecutor(service, store)
        frames = [
            frame
            async for frame in executor.stream(
                _params(scenario.query), object(), "token"
            )
        ]
        return frames, store.get_task("task-1")

    legacy_frames, legacy_task = await stream(
        _LegacyEntry(legacy_service(scenario.llm()))
    )
    adk_frames, adk_task = await stream(adk_service(scenario.llm()))

    assert _stable(adk_frames) == _stable(legacy_frames)
    assert _stable(adk_task) == _stable(legacy_task)
    assert adk_task["status"]["state"] == "completed"
    texts = [
        part.get("text")
        for frame in adk_frames
        if frame.get("kind") == "artifact-update"
        for part in frame["artifact"]["parts"]
    ]
    if scenario.plans[0] == plan_json("needs_clarification"):
        # Today's executor drops ``clarification`` items: an A2A client gets a
        # completed task without the question. Kept as is — it is the contract.
        assert texts == []
    else:
        assert "".join(t for t in texts if t).startswith("Объяснение плана.")
