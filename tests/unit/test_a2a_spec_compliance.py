"""Unit tests for A2A 0.3 spec-compliance fixes shared by the restriction / provision /
document-QA agents.

Covers the integrator feedback: structured ``scenario_id`` (DataPart / metadata) with a text
fallback, ``-32602`` for invalid params, ``messageId`` on every Message (including the echoed
user message), RFC3339 timestamps, the ``type`` → ``kind`` part discriminator, the required
``scenario-context`` AgentCard extension, ``historyLength`` trimming, and a terminal ``failed``
event on the streaming error path.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.agents.a2a.a2a_format import (
    SCENARIO_CONTEXT_EXTENSION_URI,
    apply_history_length,
    normalize_response,
    utc_now_rfc3339,
)
from src.agents.a2a.agent import RestrictionA2AAgent
from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.provision_agent import ProvisionA2AAgent
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.services.a2a_service import A2AService

FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
            "properties": {},
        }
    ],
}


class FakeRestrictionService:
    """Minimal stand-in for RestrictionParserService yielding fixed pipeline events."""

    def __init__(self, events: list[dict] | None = None) -> None:
        self._events = events or []
        self.calls: list[dict] = []

    async def run_restriction_execution_pipline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self._events:
            yield event


def _executor(events=None) -> RestrictionAgentExecutor:
    return RestrictionAgentExecutor(FakeRestrictionService(events), A2ATaskStore())


def _service(events=None) -> tuple[A2AService, FakeRestrictionService]:
    svc = FakeRestrictionService(events)
    return A2AService(svc), svc


def _message(text: str, **extra) -> dict:
    msg = {"role": "user", "parts": [{"type": "text", "text": text}]}
    msg.update(extra)
    return msg


# ---------------------------------------------------------------------------
# a2a_format helpers
# ---------------------------------------------------------------------------
def test_utc_now_rfc3339_has_zulu_offset():
    ts = utc_now_rfc3339()
    assert ts.endswith("Z")
    # Parses back as an aware datetime (protobuf Timestamp.FromJsonString-compatible).
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_normalize_response_renames_part_type_to_kind():
    obj = {"parts": [{"type": "text", "text": "hi"}]}
    normalize_response(obj)
    assert obj["parts"][0] == {"kind": "text", "text": "hi"}


def test_normalize_response_preserves_geojson_type():
    obj = {"parts": [{"type": "data", "data": dict(FEATURE_COLLECTION)}]}
    normalize_response(obj)
    part = obj["parts"][0]
    assert part["kind"] == "data"
    # GeoJSON 'type' nested under data must NOT be rewritten.
    assert part["data"]["type"] == "FeatureCollection"
    assert part["data"]["features"][0]["type"] == "Feature"


def test_normalize_response_does_not_clobber_existing_kind():
    obj = {"parts": [{"kind": "text", "type": "legacy", "text": "x"}]}
    normalize_response(obj)
    assert obj["parts"][0]["kind"] == "text"


def test_apply_history_length_trims_and_zero_clears():
    task = {"history": [1, 2, 3, 4]}
    assert apply_history_length(dict(task), 2)["history"] == [3, 4]
    assert apply_history_length(dict(task), 0)["history"] == []
    # Absent / invalid limits leave history untouched.
    assert apply_history_length(dict(task), None)["history"] == [1, 2, 3, 4]
    assert apply_history_length(dict(task), "nope")["history"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# scenario_id resolution (structured channels + text fallback)
# ---------------------------------------------------------------------------
def test_scenario_id_from_data_part():
    out = _executor()._prepare_execution(
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
    assert out["scenario_id"] == 772
    assert "школ" in out["user_query"]


def test_scenario_id_from_message_metadata():
    out = _executor()._prepare_execution(
        {"message": _message("build zone", metadata={"scenario_id": 5})}
    )
    assert out["scenario_id"] == 5


def test_scenario_id_from_params_metadata():
    out = _executor()._prepare_execution(
        {"message": _message("build zone"), "metadata": {"scenario_id": 9}}
    )
    assert out["scenario_id"] == 9


def test_scenario_id_text_fallback_still_supported():
    out = _executor()._prepare_execution(
        {"message": _message("scenario_id=772 построй зону вокруг школ")}
    )
    assert out["scenario_id"] == 772
    # The inline id is stripped from the forwarded query.
    assert "scenario_id" not in out["user_query"]


def test_scenario_id_missing_raises_invalid_params():
    with pytest.raises(A2AInvalidParamsError) as err:
        _executor()._prepare_execution({"message": _message("build zone")})
    assert err.value.code == -32602
    assert "scenario_id" in err.value.message


def test_scenario_id_non_integer_raises_invalid_params():
    with pytest.raises(A2AInvalidParamsError) as err:
        _executor()._prepare_execution(
            {"message": _message("x", metadata={"scenario_id": "abc"})}
        )
    assert err.value.code == -32602


def test_data_part_scenario_id_wins_over_text():
    out = _executor()._prepare_execution(
        {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "data", "data": {"scenario_id": 100}},
                    {"kind": "text", "text": "scenario_id=200 build"},
                ],
            }
        }
    )
    assert out["scenario_id"] == 100


# ---------------------------------------------------------------------------
# messageId on every Message (including the echoed user message)
# ---------------------------------------------------------------------------
async def test_user_echo_has_message_id():
    ex = _executor([{"type": "chunk", "content": {"text": "ok"}}])
    task = await ex.execute(
        {"message": _message("scenario_id=1 build")}, mcp_client=object()
    )
    assert task["history"], "history must not be empty"
    for msg in task["history"]:
        assert msg.get("messageId"), f"missing messageId on {msg}"
    assert task["message"]["messageId"]


async def test_user_echo_preserves_incoming_message_id():
    ex = _executor([{"type": "chunk", "content": {"text": "ok"}}])
    task = await ex.execute(
        {"message": _message("scenario_id=1 build", messageId="client-123")},
        mcp_client=object(),
    )
    assert task["history"][0]["messageId"] == "client-123"


# ---------------------------------------------------------------------------
# RFC3339 timestamps
# ---------------------------------------------------------------------------
def test_task_store_status_timestamp_is_rfc3339():
    store = A2ATaskStore()
    task = store.create_task("t1", "c1", {"role": "user", "parts": []}, {})
    assert task["status"]["timestamp"].endswith("Z")
    from python_a2a.models.task import TaskState

    status = store.set_status("t1", TaskState.COMPLETED)
    assert status["timestamp"].endswith("Z")


# ---------------------------------------------------------------------------
# type -> kind at the service boundary
# ---------------------------------------------------------------------------
async def test_service_response_uses_kind_discriminator():
    service, _ = _service(
        [
            {"type": "chunk", "content": {"text": "текст ответа"}},
            {
                "type": "feature_collection",
                "content": {
                    "name": "schools",
                    "feature_collection": FEATURE_COLLECTION,
                },
            },
        ]
    )
    response = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {"message": _message("scenario_id=772 build")},
        },
        object(),
    )
    task = response["result"]
    # Every part across history / status.message / artifacts uses "kind".
    for msg in task["history"]:
        for part in msg["parts"]:
            assert "kind" in part and "type" not in part
    geojson_parts = [
        part
        for artifact in task["artifacts"]
        for part in artifact["parts"]
        if part.get("kind") == "data"
    ]
    assert geojson_parts, "expected a GeoJSON data part"
    # GeoJSON payload under data keeps its own 'type'.
    assert geojson_parts[0]["data"]["type"] == "FeatureCollection"


# ---------------------------------------------------------------------------
# JSON-RPC error code through the service
# ---------------------------------------------------------------------------
async def test_missing_scenario_id_returns_minus_32602():
    service, _ = _service()
    response = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {"message": _message("build zone")},
        },
        object(),
    )
    assert "error" in response
    assert response["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# historyLength
# ---------------------------------------------------------------------------
async def test_history_length_zero_drops_history():
    service, _ = _service([{"type": "chunk", "content": {"text": "ok"}}])
    response = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": _message("scenario_id=1 build"),
                "configuration": {"historyLength": 0},
            },
        },
        object(),
    )
    assert response["result"]["history"] == []


# ---------------------------------------------------------------------------
# streaming error path emits a terminal failed event (not an empty stream)
# ---------------------------------------------------------------------------
async def test_stream_invalid_params_emits_terminal_failed():
    service, _ = _service()
    events = [
        event
        async for event in service.stream_json_rpc(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/stream",
                "params": {"message": _message("no scenario here")},
            },
            object(),
        )
    ]
    assert events, "stream must not be empty on error"
    terminal = [
        e
        for e in events
        if e.get("result", {}).get("statusUpdate", {}).get("final")
        and e["result"]["statusUpdate"]["status"]["state"] == "failed"
    ]
    assert terminal, "expected a terminal failed status-update event"
    errors = [e for e in events if "error" in e]
    assert errors and errors[0]["error"]["code"] == -32602


async def test_stream_happy_path_emits_completed_terminal():
    service, _ = _service([{"type": "chunk", "content": {"text": "ok"}}])
    events = [
        event
        async for event in service.stream_json_rpc(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/stream",
                "params": {"message": _message("scenario_id=1 build")},
            },
            object(),
        )
    ]
    assert "task" in events[0]["result"]
    assert any(
        e.get("result", {}).get("statusUpdate", {}).get("final")
        and e["result"]["statusUpdate"]["status"]["state"] == "completed"
        for e in events
    )


# ---------------------------------------------------------------------------
# AgentCard: required scenario-context extension + discoverable JSON input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("agent_cls", [RestrictionA2AAgent, ProvisionA2AAgent])
def test_agent_card_declares_required_scenario_extension(agent_cls):
    card = agent_cls().get_agent_card("http://host:80")
    extensions = card["capabilities"]["extensions"]
    ext = next(e for e in extensions if e["uri"] == SCENARIO_CONTEXT_EXTENSION_URI)
    assert ext["required"] is True
    assert "scenario_id" in ext["params"]["properties"]
    assert "application/json" in card["defaultInputModes"]
    assert "application/json" in card["skills"][0]["inputModes"]


def test_agent_card_protocol_version():
    card = RestrictionA2AAgent().get_agent_card("")
    assert card["protocolVersion"] == "0.3.0"
