import pytest
from loguru import logger

from src.agents.model_clients.request_priority import (
    DEFAULT_PRIORITIES,
    _warn_once,
    request_priority,
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in (
        "OPENAI_REQUEST_PRIORITY",
        "LLM_REQUEST_PRIORITIES",
        "LLM_REQUEST_PRIORITY_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    _warn_once.cache_clear()


def test_default_table():
    urgent = {
        "dvd.planner",
        "dvd.conversation",
        "dvd.context_reducer",
        "dvd.answer",
        "dvd.critic",
        "dvd.revision",
        "norms.planner",
        "norms.answer",
        "norms.review",
        "pzz.answer",
        "provision.answer",
        "scenario.answer",
        "scenario.judge",
        "orchestrator.plan",
        "simple_llm",
    }
    parsing = {
        "restriction.plan",
        "restriction.explain",
        "restriction.answer",
        "pzz.inputs",
        "pzz.columns",
        "provision.plan",
        "scenario.plan",
        "scenario.mapping",
        "chat_title",
        "compliance.scope",
        "compliance.mapping",
        "project.inputs",
    }
    expected = {
        **dict.fromkeys(urgent, -100),
        **dict.fromkeys(parsing, -60),
        "context_summary": 20,
    }
    assert DEFAULT_PRIORITIES == expected
    assert {key: request_priority(key) for key in expected} == expected


def test_live_overrides_and_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_REQUEST_PRIORITY", "-7")
    assert request_priority() == -7
    assert request_priority("dvd.critic") == -100
    monkeypatch.setenv("LLM_REQUEST_PRIORITIES", " dvd.critic = -60,chat_title=0")
    assert request_priority("dvd.critic") == -60
    assert request_priority("chat_title") is None
    monkeypatch.setenv("LLM_REQUEST_PRIORITIES", "dvd.critic=4,dvd.critic=-8")
    assert request_priority("dvd.critic") == -8
    monkeypatch.setenv("OPENAI_REQUEST_PRIORITY", "0")
    assert request_priority() is None


@pytest.mark.parametrize("entry", ["dvd.critic=bad", "unknown=5", "dvd.critic"])
def test_bad_entries_warn_once_and_leave_default(monkeypatch, entry):
    monkeypatch.setenv("LLM_REQUEST_PRIORITIES", entry + ",chat_title=-3")
    warnings = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        for _ in range(2):
            assert request_priority("dvd.critic") == -100
            assert request_priority("chat_title") == -3
    finally:
        logger.remove(sink)
    assert len(warnings) == 1


@pytest.mark.parametrize("value", ["false", "0", "no", "off", " FALSE "])
def test_master_switch(monkeypatch, value):
    monkeypatch.setenv("OPENAI_REQUEST_PRIORITY", "-7")
    monkeypatch.setenv("LLM_REQUEST_PRIORITIES", "dvd.critic=-3")
    monkeypatch.setenv("LLM_REQUEST_PRIORITY_ENABLED", value)
    assert request_priority("dvd.critic") is None
    assert request_priority() is None


@pytest.mark.parametrize("method", ["_select_evidence", "_summarize"])
async def test_context_reducer_passes_key(fake_llm, method):
    from src.agents.services.dvd.context_reducer import DvdContextReducer, SummaryError

    reducer = DvdContextReducer(fake_llm)
    with pytest.raises(SummaryError):
        await getattr(reducer, method)("m", "q", "[1] Source\nSome evidence.", 2000)
    assert fake_llm.chat_calls[-1].request_kind == "dvd.context_reducer"
