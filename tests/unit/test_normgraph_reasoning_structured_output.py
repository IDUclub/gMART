from __future__ import annotations

import json

import pytest

from src.agents.services.normgraph_reasoning import NormGraphRetrievalPlanner


class RecordingLlmClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "message": {
                "content": json.dumps(
                    {
                        "primary_tool": "search",
                        "search_query": "противопожарное расстояние",
                        "limit": 10,
                        "neighbors_depth": 0,
                        "check_conflicts": False,
                    }
                )
            }
        }


@pytest.mark.asyncio
async def test_planner_disables_thinking_and_requests_json_schema():
    client = RecordingLlmClient()

    plan = await NormGraphRetrievalPlanner(client).build_plan(
        "gpt-oss:20b", "Какие противопожарные расстояния действуют?"
    )

    assert plan.search_query == "противопожарное расстояние"
    assert client.calls[0]["think"] is False
    assert client.calls[0]["format"]["title"] == "NormGraphPlan"
    assert client.calls[0]["options"]["num_predict"] == 1024
