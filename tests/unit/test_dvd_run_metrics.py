"""Stage 0 instrumentation: per-run stage latency, LLM usage and decisions."""

from __future__ import annotations

import json

from loguru import logger

from tests.helpers import plan_json, verdict_json


def _metrics_records(logs: list[str]) -> list[dict]:
    from src.agents.services.dvd.run_metrics import LOG_EVENT

    return [
        json.loads(line.split(LOG_EVENT, 1)[1].strip())
        for line in logs
        if LOG_EVENT in line
    ]


async def test_run_logs_one_metrics_record(service, fake_llm, fake_mcp):
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Не менее 6 м² на человека [1]."]
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="INFO")
    try:
        events = [
            event
            async for event in service.run_document_qa_pipeline(
                dvd_mcp_client=fake_mcp,
                token="tok",
                model="m",
                temperature=0.0,
                user_query="нормы озеленения",
                chat_id="chat-1",
            )
        ]
    finally:
        logger.remove(sink)

    (record,) = _metrics_records(logs)
    request_id = events[0]["content"]["request_id"]
    assert record["request_id"] == request_id
    for stage in ("planner", "retrieval", "context_prepare", "critic"):
        assert stage in record["stages_ms"]
    assert record["counters"]["critic_calls"] == 1
    assert record["counters"]["drafts"] == 1
    assert record["counters"]["retrieved_hits"] == 1
    assert record["decisions"]["first_draft_accepted"] is True
    assert record["decisions"]["query_type"] == "semantic"
    assert record["decisions"]["rag_iterations"] == 1
    assert record["ttft_ms"] is not None
    assert record["total_latency_ms"] >= record["ttft_ms"]
    assert service._active == {}


async def test_failed_run_still_logs_metrics(service, fake_llm, fake_mcp):
    async def broken(*args, **kwargs):
        raise RuntimeError("planner down")

    fake_llm.chat = broken
    logs: list[str] = []
    sink = logger.add(lambda message: logs.append(str(message)), level="INFO")
    try:
        try:
            async for _ in service.run_document_qa_pipeline(
                dvd_mcp_client=fake_mcp,
                token="tok",
                model="m",
                temperature=0.0,
                user_query="нормы озеленения",
                chat_id="chat-1",
            ):
                pass
        except RuntimeError:
            pass
    finally:
        logger.remove(sink)
    (record,) = _metrics_records(logs)
    assert record["ttft_ms"] is None
    assert "planner" in record["stages_ms"]
    assert service._active == {}


def test_llm_usage_counts_finished_calls_of_the_metered_context():
    import contextvars

    from src.agents.model_clients import llm_usage
    from src.agents.model_clients.llm_pace import LlmPace

    pace = LlmPace()

    def run():
        usage = llm_usage.meter()
        pace.finished(pace.started(100), completion_tokens=20, prompt_tokens=300)
        pace.finished(pace.started(100), completion_tokens=None)
        llm_usage.record_tokenize()
        llm_usage.record_tokenize(cached=True)
        return usage

    usage = contextvars.copy_context().run(run)
    assert usage.snapshot() == {
        "calls": 2,
        "input_tokens": 300,
        "output_tokens": 20,
        "unmeasured_calls": 1,
        "tokenize_calls": 1,
        "tokenize_cache_hits": 1,
    }
    # Outside a metered context nothing is recorded (and nothing fails).
    pace.finished(pace.started(100), completion_tokens=5, prompt_tokens=5)
