"""The retrieval planner reasons briefly and never waits on a runaway plan."""

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.dvd.dvd_reasoning import (
    RetrievalPlanner,
    planner_output_cap,
    planner_reasoning_effort,
)
from tests.unit.test_dvd_critic_limit import LENGTH, CappedLlm

GPT_OSS = "gpt-oss-20b"


def test_planner_reasons_at_low_effort_on_gpt_oss(monkeypatch):
    adapter = OpenAiCompatAdapter("http://llm.test")
    assert planner_reasoning_effort(adapter, GPT_OSS) == "low"
    monkeypatch.setenv("DVD_PLANNER_REASONING_EFFORT", "")
    # Empty leaves the server default.
    assert planner_reasoning_effort(adapter, GPT_OSS) is None
    assert planner_reasoning_effort(adapter, "qwen3") is None


def test_planner_cap_is_configurable(monkeypatch):
    assert planner_output_cap() == 4096
    monkeypatch.setenv("DVD_PLANNER_MAX_OUTPUT_TOKENS", "2048")
    assert planner_output_cap() == 2048


async def test_capped_plan_falls_back_to_the_question():
    llm = CappedLlm()
    llm.json_responses = [LENGTH]
    plan = await RetrievalPlanner(llm).build_plan(GPT_OSS, "Что такое денник?")
    assert plan.search_query.startswith("Что такое денник")
    assert plan.retrieval_mode == "semantic"
    # One call: a capped plan is not retried with a wider limit.
    assert len(llm.chat_calls) == 1
    assert llm.chat_calls[0].options["num_predict"] <= planner_output_cap()
