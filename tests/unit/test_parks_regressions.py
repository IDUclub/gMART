"""Regression coverage for cursor-only A2A replay and bounded analysis JSON."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.services.parks.parks_a2a_service import ParksA2AService
from src.agents.services.parks.parks_service import ParksService


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("awaiting_confirmation", "input-required"),
        ("submission_unknown", "failed"),
        ("failed", "failed"),
        ("done", "completed"),
    ],
)
@pytest.mark.parametrize(
    "remaining", [[], [{"type": "chunk", "content": {"text": "", "done": True}}]]
)
async def test_a2a_final_status_uses_checkpoint_not_replayed_tail(
    status, expected, remaining
):
    async def pipeline(**kwargs):
        assert kwargs["after"] == 9
        for event in remaining:
            yield event

    service = SimpleNamespace(
        get_run=AsyncMock(
            return_value={"status": status, "conversation": "a2a:context"}
        ),
        run_parks_pipeline=pipeline,
    )
    responses = [
        item
        async for item in ParksA2AService(service).stream_json_rpc(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/stream",
                "params": {"metadata": {"request_id": "existing", "after": 9}},
            },
            "test-token",
        )
    ]
    final = responses[-1]["result"]
    assert final["taskId"] == "existing"
    assert final["final"] is True
    assert final["status"]["state"] == expected
    service.get_run.assert_awaited_with("existing", "test-token")


def state(features):
    return {
        "query": "Парк для отдыха",
        "parameters": {"tags": ["REST"], "park_geojson": {}},
        "economy": {"economyParams": [{"name": "total_area", "primaryValue": 100}]},
        "result": {"features": features},
    }


def test_analysis_keeps_complete_records_and_reports_omission():
    features = [
        {"id": i, "geometry": {"coordinates": [[30, 60]] * 10000}} for i in range(3)
    ]
    features.append(
        {"id": "small", "geometry": {"type": "Point", "coordinates": [30, 60]}}
    )
    original = state(features)
    text = ParksService._analysis_context(original)
    data = json.loads(text)
    assert len(text) <= 60000
    assert data["features"] == [features[-1]]
    assert data["included_features"] == 1
    assert data["omitted_features"] == 3
    assert data["context_truncated"] is True
    assert data["economy"] == original["economy"]
    assert original["result"]["features"] == features


def test_analysis_small_payload_is_not_truncated():
    original = state([{"id": 1}])
    data = json.loads(ParksService._analysis_context(original))
    assert data["features"] == original["result"]["features"]
    assert data["context_truncated"] is False
    assert data["omitted_features"] == 0


def test_analysis_counts_repeated_tags_and_marks_provider_mismatch():
    original = state([])
    original["parameters"]["tags"] = ["CHILDREN", "CHILDREN", "REST", "SPORTS"]
    original["economy"]["economyParams"] = [
        {"name": "CHILDREN", "secondaryValue": 3},
        {"name": "REST", "secondaryValue": 1},
        {"name": "SPORTS", "secondaryValue": None},
    ]
    counts = json.loads(ParksService._analysis_context(original))["count_comparison"]
    assert counts["CHILDREN"] == {"requested": 2, "actual": 3, "status": "mismatch"}
    assert counts["REST"] == {"requested": 1, "actual": 1, "status": "match"}
    assert counts["SPORTS"] == {"requested": 1, "actual": None, "status": "unknown"}


def test_analysis_feature_count_limit_is_explicit():
    data = json.loads(
        ParksService._analysis_context(state([{"id": i} for i in range(160)]))
    )
    assert data["included_features"] == 150
    assert data["omitted_features"] == 10
    assert data["context_truncated"] is True


def test_analysis_oversized_metadata_is_omitted_as_a_whole():
    original = state([{"id": 1}])
    original["parameters"]["location_requests"] = [{"text": "я" * 70000}]
    text = ParksService._analysis_context(original)
    data = json.loads(text)
    assert len(text) <= 60000
    assert data["parameters"] is None
    assert data["omitted_fields"] == ["parameters"]
    assert data["context_truncated"] is True
    assert data["economy"] == original["economy"]
