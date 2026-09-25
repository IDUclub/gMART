"""Regressions for the production failures observed on 2026-09-22."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
from src.agents.services.dvd.context_reducer import PreparedContext
from src.agents.services.provision.provision_context import ProvisionContextBuilder
from src.agents.services.provision.provision_tool_executor import ProvisionToolExecutor
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.service_entities.provision_plan import ProvisionPlan
from tests.helpers import answer_text, plan_json, verdict_json
from tests.unit.test_app_config_dvd import make_config
from tests.unit.test_dvd_rag_service import _run
from tests.unit.test_urban_mcp_client import FakeTransport


@pytest.mark.parametrize(
    "url", [" https://urban.example/urban_mcp \n", "https://urban.example/urban_mcp/\t"]
)
def test_urban_mcp_strips_deployment_whitespace_before_building_transports(url):
    endpoints = []

    def factory(url, **kwargs):
        endpoints.append(url)
        return FakeTransport([])

    config = make_config(urban_mcp_url=url)
    assert config.URBAN_MCP_URL == "https://urban.example/urban_mcp"
    UrbanMcpClient(url, None, client_factory=factory)
    assert endpoints
    assert all(
        endpoint.startswith("https://urban.example/urban_mcp/mcp/")
        for endpoint in endpoints
    )


@pytest.mark.parametrize("stage", ["preparation", "answer_generation", "review"])
async def test_partial_document_context_is_reviewed_warned_persisted_and_replayed(
    service, fake_llm, fake_mcp, stage
):
    fact = fake_mcp.default_hits[0]["text"]
    context = "[1] Source\n" + fact
    partial = PreparedContext(
        context,
        processed_parts=1,
        failed_parts=["round-1/part-5: [5] (output_truncated)"],
    )
    contexts = [
        partial if stage == "preparation" else PreparedContext(context),
        partial if stage == "review" else PreparedContext(context),
    ]
    if stage == "answer_generation":
        contexts = [
            PreparedContext(context + " " * 40000),
            partial,
            PreparedContext(context),
        ]
    service.context_reducer.prepare = AsyncMock(side_effect=contexts)
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = [fact + " [1]"]
    events = await _run(service, fake_mcp)
    answer = answer_text(events)
    assert fact in answer
    assert "частичный ответ" in answer.lower()
    assert "не удалось обработать" in answer
    assert "output_truncated" not in json.dumps(events)
    assert not any(event["type"] == "error" for event in events)
    assert any(
        event["type"] == "status" and event["content"].get("status") == "self_review"
        for event in events
    )
    request_id = events[0]["content"]["request_id"]
    assert (await service.state_store.get_state(request_id))["status"] == "done"
    assert (await service.state_store.get_checkpoint(request_id))["qa_progress"][
        "final_answer"
    ] == answer
    assert service._schedule_persist_answer.call_args.args[2]["final_answer"] == answer
    assert await _run(service, fake_mcp, request_id=request_id) == events


async def test_real_reducer_failure_keeps_other_sources_in_pipeline(service, fake_llm):
    from src.agents.services.dvd.context_reducer import DvdContextReducer
    from tests.helpers import FakeDvdMcpClient
    from tests.unit.test_dvd_context_reducer import Summarizer

    service.context_reducer = DvdContextReducer(
        Summarizer(fail=True), window_tokens=8192, retries=0
    )
    client = FakeDvdMcpClient(
        hits_per_call=[
            [
                {"name": "Good source", "text": "padding " * 400 + "FACT1"},
                {"name": "Failed source", "text": "FAIL_SOURCE " * 400},
            ]
        ]
    )
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["FACT1 [1]"]
    events = await _run(service, client)
    assert "FACT1" in answer_text(events)
    assert "частичный ответ" in answer_text(events)
    assert not any(event["type"] == "error" for event in events)


async def test_failed_and_irrelevant_sources_do_not_allow_an_ungrounded_answer(
    service, fake_llm
):
    from src.agents.services.dvd.context_reducer import DvdContextReducer
    from tests.helpers import FakeDvdMcpClient
    from tests.unit.test_dvd_context_reducer import Summarizer

    service.context_reducer = DvdContextReducer(
        Summarizer(fail=True), window_tokens=8192, retries=0
    )
    client = FakeDvdMcpClient(
        hits_per_call=[
            [
                {"name": "Irrelevant source", "text": "padding " * 400},
                {"name": "Failed source", "text": "FAIL_SOURCE " * 400},
            ]
        ]
    )
    fake_llm.json_responses = [plan_json()]
    events = await _run(service, client)
    assert not answer_text(events)
    assert events[-1]["type"] == "error"
    assert not any(call.stream for call in fake_llm.chat_calls)


@pytest.mark.parametrize(
    "query", ["Привет!", "Здравствуйте", "Добрый день.", "hello", "Hi!"]
)
async def test_standalone_greeting_does_not_search_or_require_document_citations(
    service, fake_llm, fake_mcp, query
):
    events = await _run(service, fake_mcp, user_query=query)
    assert answer_text(events)
    assert not fake_mcp.search_calls
    assert not fake_llm.chat_calls
    assert not any(event["type"] == "error" for event in events)
    request_id = events[0]["content"]["request_id"]
    assert await _run(service, fake_mcp, request_id=request_id) == events


async def test_greeting_with_document_question_still_searches(
    service, fake_llm, fake_mcp
):
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Ответ [1]."]
    await _run(service, fake_mcp, user_query="Привет! Какие требования к школам?")
    assert fake_mcp.search_calls


async def test_summary_keeps_missing_service_without_zero_metrics(state_store):
    svc = object.__new__(ProvisionService)
    svc.state_store = state_store
    svc.resolve_model = AsyncMock(return_value="m")
    svc.context_builder = ProvisionContextBuilder()
    svc.tool_executor = ProvisionToolExecutor()
    svc._resolve_service_plan = AsyncMock(
        return_value=(
            ProvisionPlan(mode="summary"),
            {"Школа": 22, "Спортивный центр": 68},
        )
    )
    effects = SimpleNamespace(
        calculate_services_provision=AsyncMock(
            return_value={
                "services": {
                    "22": {
                        "name": "Школа",
                        "summary": {
                            "total_capacity": 80,
                            "total_demand": 100,
                            "deficit": 20,
                        },
                    },
                    "68": {
                        "name": "Спортивный центр",
                        "error": "HTTPException: 400: Service type id not found in urban_db for provided territory/context ids. PRIVATE_DETAIL",
                    },
                }
            }
        )
    )
    events = [
        event
        async for event in svc.run_provision_pipeline(
            idu_mcp_client=object(),
            effects_mcp_client=effects,
            token="t",
            model="m",
            temperature=0,
            user_query="Сводка обеспеченности",
            scenario_id=845,
            persist_history=False,
        )
    ]
    table = next(event["content"] for event in events if event["type"] == "table")
    rows = {row["service"]: row for row in table["rows"]}
    assert rows["Школа"]["deficit"] == 20
    assert rows["Спортивный центр"]["deficit"] is None
    assert rows["Спортивный центр"]["status"] == "Нет данных"
    assert "нет данных" in answer_text(events).lower()
    assert "PRIVATE_DETAIL" not in json.dumps(events)
    assert (await state_store.get_state(events[0]["content"]["request_id"]))[
        "status"
    ] == "done"


async def test_quote_mismatch_then_reasoning_exhaustion_retries_smaller_source_parts(
    monkeypatch,
):
    from src.agents.services.dvd.context_reducer import DvdContextReducer
    from tests.unit.test_llm_adapters import _adapter_with, _Choice, _Completion, _Delta

    adapter, _ = _adapter_with(None)

    async def tokenize(endpoint, *, body, **_):
        messages = body["messages"]
        if len(messages) == 1 and messages[0]["content"].startswith("[5]"):
            # The evidence itself: large enough to need reduction.
            return {"count": len(messages[0]["content"].encode("utf-8"))}
        return {"count": 1000}

    adapter.client.post = tokenize
    calls = []

    async def create(**request):
        calls.append(request)
        payload = json.loads(request["messages"][1]["content"])
        if (
            "selections"
            not in request["response_format"]["json_schema"]["schema"]["properties"]
        ):
            content = {
                "evidence": [{"source_id": "[5]", "quotes": ["Invented rule"]}],
                "complete": True,
            }
        else:
            sources = payload["sources"]
            if (
                sum(
                    len(span) for source in sources for span in source["spans"].values()
                )
                > 700
            ):
                return _Completion(
                    [
                        _Choice(
                            message=_Delta("", reasoning_content="private reasoning"),
                            finish_reason="length",
                        )
                    ]
                )
            content = {
                "selections": {
                    source["source_id"]: [
                        int(i) for i, text in source["spans"].items() if "FACT" in text
                    ]
                    for source in sources
                },
                "complete": True,
            }
        return _Completion(
            [_Choice(message=_Delta(json.dumps(content)), finish_reason="stop")]
        )

    adapter.client.chat.completions.create = create
    try:
        result = await DvdContextReducer(
            adapter, window_tokens=8192, concurrency=1
        ).prepare(
            "gpt-oss-20b",
            "q",
            "[5] Standard\nFACT1.\n" + "padding. " * 700 + "\nFACT2.",
        )
        assert not result.failed_parts
        assert "FACT1" in result.text and "FACT2" in result.text
        assert "Invented" not in result.text
        assert any(
            "output_truncated" in call["messages"][0]["content"] for call in calls
        )
        assert all(call["max_tokens"] <= 8192 - 1000 - 256 for call in calls)
    finally:
        await adapter.client.close()
