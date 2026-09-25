from __future__ import annotations

import pytest
from python_a2a.models.task import TaskState

from src.agents.a2a.compliance_agent import ComplianceA2AAgent
from src.agents.a2a.compliance_executor import (
    ComplianceAgentExecutor,
    ComplianceMcpClients,
)
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.services.compilance.compliance_a2a_service import (
    ComplianceA2AService,
)

CLIENTS = ComplianceMcpClients(idu=object(), normgraph=object())


class FakeRestrictionService:
    def __init__(self, events: list[dict]) -> None:
        self.events = events
        self.calls: list[dict] = []

    async def run_compliance_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


def _params(text: str, **data) -> dict:
    return {
        "contextId": "ctx-1",
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": text},
                {"type": "data", "data": {"scenario_id": 772, **data}},
            ],
        },
    }


def test_agent_card_points_at_the_compliance_endpoint():
    card = ComplianceA2AAgent().get_agent_card("http://host:80")

    assert card["name"] == "compliance-agent"
    assert card["url"] == "http://host:80/compliance/a2a"


def test_inline_scenario_id_is_accepted_and_hidden_from_the_query():
    executor = ComplianceAgentExecutor(FakeRestrictionService([]), A2ATaskStore())

    execution = executor._prepare_execution(
        {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": "scenario_id=845 проверь нормы"}],
            }
        }
    )

    assert execution["scenario_id"] == 845
    assert execution["user_query"] == "проверь нормы"


def test_scenario_id_is_required():
    executor = ComplianceAgentExecutor(FakeRestrictionService([]), A2ATaskStore())

    with pytest.raises(A2AInvalidParamsError):
        executor._prepare_execution(
            {"message": {"role": "user", "parts": [{"type": "text", "text": "x"}]}}
        )


async def test_document_choice_ends_the_task_in_input_required():
    options = [{"number": 1, "label": "СП 42 — 4", "value": "СП 42"}]
    service = FakeRestrictionService(
        [
            {"type": "status", "content": {"status": "compliance_scope", "text": "…"}},
            {
                "type": "clarification",
                "content": {"question": "Выберите документ", "options": options},
            },
        ]
    )
    executor = ComplianceAgentExecutor(service, A2ATaskStore())

    events = [
        event
        async for event in executor.stream(
            _params("Проверь нормы по СП"), CLIENTS, "token"
        )
    ]

    final = events[-1]
    assert final["final"] is True
    assert final["status"]["state"] == TaskState.INPUT_REQUIRED.value
    assert final["status"]["message"]["parts"][1]["data"]["options"] == options
    call = service.calls[0]
    assert call["conversation_key"] == "a2a-compliance:ctx-1"
    assert call["persist_history"] is False
    assert call["scenario_id"] == 772


async def test_layers_and_summary_become_separate_artifacts():
    layer = {"type": "FeatureCollection", "features": []}
    service = FakeRestrictionService(
        [
            {
                "type": "feature_collection",
                "content": {"name": "Нарушение нормы — А", "feature_collection": layer},
            },
            {
                "type": "feature_collection",
                "content": {"name": "Нарушение нормы — Б", "feature_collection": layer},
            },
            {"type": "compliance_summary", "content": {"total_norms": 2}},
            {"type": "chunk", "content": {"text": "Готово", "done": True}},
        ]
    )
    store = A2ATaskStore()
    executor = ComplianceAgentExecutor(service, store)

    task = await executor.execute(_params("Проверь нормы"), CLIENTS, "token")

    ids = [artifact["artifactId"] for artifact in task["artifacts"]]
    assert ids[:3] == [
        "compliance-layer-1",
        "compliance-layer-2",
        "compliance-summary",
    ]
    assert task["status"]["state"] == TaskState.COMPLETED.value


async def test_json_rpc_send_runs_the_executor():
    service = FakeRestrictionService(
        [{"type": "chunk", "content": {"text": "Готово", "done": True}}]
    )
    a2a = ComplianceA2AService(service)

    response = await a2a.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": _params("Проверь нормы"),
        },
        CLIENTS,
        "token",
    )

    assert response["result"]["status"]["state"] == TaskState.COMPLETED.value
    assert service.calls[0]["normgraph_mcp_client"] is CLIENTS.normgraph
