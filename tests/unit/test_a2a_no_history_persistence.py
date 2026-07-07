"""A2A runs must leave no trace in ChatStorage: every A2A executor calls its
pipeline with persist_history=False (the DVD executor is covered in test_dvd_a2a.py)."""

from __future__ import annotations

from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.provision_executor import ProvisionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore

_PARAMS = {
    "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "запрос"}],
        "metadata": {"scenario_id": 772},
    }
}

_DONE_CHUNK = {"type": "chunk", "content": {"text": "ответ", "done": True}}


class FakeRestrictionService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_restriction_execution_pipline(self, **kwargs):
        self.calls.append(kwargs)
        yield _DONE_CHUNK


class FakeProvisionService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run_provision_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        yield _DONE_CHUNK


async def test_restriction_executor_disables_history_persistence():
    service = FakeRestrictionService()
    ex = RestrictionAgentExecutor(service, A2ATaskStore())

    async for _ in ex.stream(_PARAMS, mcp_client=object()):
        pass

    (call,) = service.calls
    assert call["persist_history"] is False


async def test_provision_executor_disables_history_persistence():
    service = FakeProvisionService()
    ex = ProvisionAgentExecutor(service, A2ATaskStore())

    async for _ in ex.stream(
        _PARAMS, idu_mcp_client=object(), effects_mcp_client=object()
    ):
        pass

    (call,) = service.calls
    assert call["persist_history"] is False
