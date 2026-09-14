"""Optional live-model acceptance; all scenario data and tool outputs are synthetic."""

from types import SimpleNamespace
from unittest.mock import Mock

import fakeredis.aioredis
import pytest

from src.agents.schema.orchestrator_response import OrchestratorResponse
from src.agents.services.orchestrator.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore


@pytest.mark.integration
async def test_live_model_comparison_artifacts_and_free_replay(
    require_openai_backend, monkeypatch
):
    url, model = require_openai_backend
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    db = fakeredis.aioredis.FakeRedis(decode_responses=True)
    calls = []

    async def pipeline(**kwargs):
        calls.append(kwargs["user_query"])
        value = 12 if len(calls) == 1 else 8
        yield {
            "type": "table",
            "content": {
                "name": f"synthetic_count_{len(calls)}",
                "title": "Школы" if len(calls) == 1 else "Детские сады",
                "columns": [{"key": "count", "label": "Количество, шт."}],
                "rows": [{"count": value}],
                "total_rows": 1,
                "complete": True,
            },
        }
        yield {
            "type": "feature_collection",
            "content": {
                "name": f"Синтетический слой {len(calls)}",
                "feature_collection": {"type": "FeatureCollection", "features": []},
            },
        }

    specialist = Mock()
    specialist.run_scenario_data_pipeline = pipeline
    config = SimpleNamespace(
        DVD_MCP_URL=None,
        NORM_GRAPH_MCP_URL=None,
        URBAN_MCP_URL="http://synthetic.invalid",
    )
    service = OrchestratorService(
        "http://localhost:11434",
        None,
        Mock(),
        PipelineStateStore(db),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        config,
        specialist,
    )
    args = dict(
        idu_mcp_client=Mock(),
        effects_mcp_client=Mock(),
        dvd_mcp_client=None,
        normgraph_mcp_client=None,
        urban_mcp_client=Mock(),
        token="synthetic-only",
        model=model,
        temperature=0,
        user_query="Сравни количество школ и детских садов в сценарии 123. Это синтетический тест, все результаты инструментов вымышлены. Верни таблицу сравнения и объясни ограничения.",
        scenario_id=123,
        persist_history=False,
    )
    try:
        events = [event async for event in service.run_orchestration_pipeline(**args)]
        for event in events:
            OrchestratorResponse.model_validate(event)
        result = events[-1]["content"]
        assert result["status"] == "completed", result["answer"]
        assert len(calls) == 2
        assert sum(a["kind"] == "feature_collection" for a in result["artifacts"]) == 2
        assert sum(a["kind"] == "table" for a in result["artifacts"]) >= 3
        assert all(a["confirmed"] for a in result["artifacts"])
        assert result["evidence_ids"]
        replay = [
            event
            async for event in service.run_orchestration_pipeline(
                **dict(args, request_id=result["continue_from"])
            )
        ]
        assert replay == events and len(calls) == 2
    finally:
        await service.llm_client.client.close()
        await db.aclose()
