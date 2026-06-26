"""Integration: validate the agents' A2A responses against the *official* a2a-sdk strict v0.3
models — the exact ``Task.model_validate`` path that rejected the deployed agent for the
integrators (missing ``messageId`` on the echoed user message, timezone-less ``timestamp``).

Self-skips when ``a2a-sdk`` is not installed (mirrors the tiered approach of the other
integration tests). Install it to run this check::

    uv pip install a2a-sdk

The restriction pipeline is faked, so this exercises the real router/service/task-store
serialization without hitting Ollama / MCP / Urban API or creating real spatial objects.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

# The official SDK is an optional, integrator-side dependency — skip cleanly without it.
# The strict Pydantic v0.3 models (with model_validate) live under a2a.compat.v0_3.types;
# a2a.types itself exposes the protobuf messages in a2a-sdk 1.x.
a2a_types = pytest.importorskip("a2a.compat.v0_3.types", reason="a2a-sdk not installed")

from src.agents.a2a.task_store import A2ATaskStore  # noqa: E402
from src.agents.services.a2a_service import A2AService  # noqa: E402

FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
            "properties": {"name": "school"},
        }
    ],
}


class FakeRestrictionService:
    """Yields a representative restriction pipeline: status, text chunk, GeoJSON layer."""

    async def run_restriction_execution_pipline(self, **kwargs):
        yield {"type": "status", "content": {"text": "working"}}
        yield {"type": "chunk", "content": {"text": "Зона ограничения построена."}}
        yield {
            "type": "feature_collection",
            "content": {"name": "schools", "feature_collection": FEATURE_COLLECTION},
        }


async def _run_send(params: dict) -> dict:
    service = A2AService(FakeRestrictionService(), task_store=A2ATaskStore())
    response = await service.handle_json_rpc(
        {"jsonrpc": "2.0", "id": "it-1", "method": "message/send", "params": params},
        object(),
    )
    assert "result" in response, response
    return response["result"]


async def test_message_send_task_passes_official_v030_validation():
    result = await _run_send(
        {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "data", "data": {"scenario_id": 772}},
                    {"kind": "text", "text": "построй зону вокруг школ 200 м"},
                ],
            }
        }
    )

    # The exact strict parse that failed for the integrators must now succeed.
    task = a2a_types.Task.model_validate(result)

    assert task.kind == "task"
    assert task.status.state.value in {"completed", "working", "failed"}
    # Every Message in history validated — including the echoed user message (history[0]).
    assert task.history, "history must be present"
    for message in task.history:
        assert message.message_id, "messageId required on every Message"


async def test_each_history_message_validates_as_official_message_model():
    result = await _run_send(
        {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "scenario_id=1 x"}],
            }
        }
    )
    for raw_message in result["history"]:
        message = a2a_types.Message.model_validate(raw_message)
        assert message.message_id


async def test_status_timestamp_parses_as_rfc3339():
    result = await _run_send(
        {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "scenario_id=1 x"}],
            }
        }
    )
    timestamp = result["status"]["timestamp"]
    # protobuf Timestamp.FromJsonString-compatible: an explicit offset must be present.
    assert timestamp.endswith("Z") or "+" in timestamp
    from datetime import datetime

    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
