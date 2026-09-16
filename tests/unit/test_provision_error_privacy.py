"""Exercise downstream calculation failures through the real provision pipeline."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from src.agents.a2a.provision_executor import ProvisionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.services.provision.provision_context import ProvisionContextBuilder
from src.agents.services.provision.provision_tool_executor import ProvisionToolExecutor
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.service_entities.provision_plan import ProvisionPlan


@pytest.mark.parametrize("mode", ["provision", "summary", "partial_summary"])
async def test_downstream_error_is_logged_and_not_sent_to_client(state_store, mode):
    svc = object.__new__(ProvisionService)
    svc.state_store = state_store
    svc.resolve_model = AsyncMock(return_value="m")
    svc.context_builder = ProvisionContextBuilder()
    svc.tool_executor = ProvisionToolExecutor()
    svc._resolve_service_plan = AsyncMock(
        return_value=(
            ProvisionPlan(
                mode="provision" if mode == "provision" else "summary",
                service_name="Школа",
            ),
            {"Школа": 22, "Детский сад": 21},
        )
    )
    results = {"22": {"name": "Школа", "error": "PRIVATE_CALCULATION_ERROR"}}
    if mode == "partial_summary":
        results["21"] = {
            "name": "Детский сад",
            "summary": {"total_demand": 100, "deficit": 20},
        }
    effects = SimpleNamespace(
        calculate_services_provision=AsyncMock(return_value={"services": results})
    )
    logs = []
    sink = logger.add(lambda message: logs.append(str(message)))
    try:
        events = [
            e
            async for e in svc.run_provision_pipeline(
                idu_mcp_client=object(),
                effects_mcp_client=effects,
                token="t",
                model="m",
                temperature=0,
                user_query="Обеспеченность школами",
                scenario_id=772,
                persist_history=False,
            )
        ]
    finally:
        logger.remove(sink)
    request_id = events[0]["content"]["request_id"]
    assert "PRIVATE_CALCULATION_ERROR" not in json.dumps(events)
    assert "PRIVATE_CALCULATION_ERROR" not in json.dumps(
        await state_store.get_buffered_events(request_id)
    )
    assert any(
        request_id in line
        and "PRIVATE_CALCULATION_ERROR" in line
        and "scenario_id=772" in line
        for line in logs
    )
    state = await state_store.get_state(request_id)
    if mode == "partial_summary":
        assert state["status"] == "done"
        assert any(e["type"] == "table" for e in events)
        assert "расчёт не выполнен" in "".join(
            e.get("content", {}).get("text", "") for e in events
        )
    else:
        assert state["status"] == "failed"
        assert events[-1]["type"] == "error"
        assert not any(e["type"] == "chunk" and e["content"]["done"] for e in events)


async def test_a2a_exception_is_logged_and_not_sent_to_client():
    class BrokenService:
        async def run_provision_pipeline(self, **kwargs):
            raise RuntimeError("PRIVATE_TRANSPORT_ERROR")
            yield  # pragma: no cover

    executor = ProvisionAgentExecutor(BrokenService(), A2ATaskStore())
    logs = []
    sink = logger.add(lambda message: logs.append(str(message)))
    try:
        events = [
            e
            async for e in executor.stream(
                {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": "Обеспеченность школами"}],
                        "metadata": {"scenario_id": 772},
                    }
                },
                idu_mcp_client=object(),
                effects_mcp_client=object(),
                token="t",
            )
        ]
    finally:
        logger.remove(sink)
    assert "PRIVATE_TRANSPORT_ERROR" not in json.dumps(events)
    assert any(
        "PRIVATE_TRANSPORT_ERROR" in line and "task_id=" in line for line in logs
    )
    assert any(
        e.get("kind") == "status-update"
        and e.get("final")
        and e["status"]["state"] == "failed"
        for e in events
    )
