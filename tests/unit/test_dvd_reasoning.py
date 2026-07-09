"""Unit tests for the RAG reasoners — RetrievalPlanner and AnswerCritic (structured output)."""

from __future__ import annotations

from src.agents.services.dvd_reasoning import AnswerCritic, RetrievalPlanner
from src.agents.services.service_entities.dvd_plan import SearchKind
from tests.helpers import plan_json, verdict_json


async def test_planner_parses_and_clamps_bounds(fake_llm):
    fake_llm.json_responses = [
        plan_json(
            search_query="озеленение дворов", kind="table", limit=99, context_height=99
        )
    ]
    plan = await RetrievalPlanner(fake_llm).build_plan("m", "вопрос", history=[])
    assert plan.search_query == "озеленение дворов"
    assert plan.kind == SearchKind.TABLE
    assert plan.limit == 20  # clamped to 1..20
    assert plan.context_height == 5  # clamped to 0..5


async def test_planner_keeps_valid_filters(fake_llm):
    fake_llm.json_responses = [
        plan_json(
            document_names=["СП 42.13330", "  ", ""],
            block="Amendment",
            types=["Clause", "TABLE"],
        )
    ]
    plan = await RetrievalPlanner(fake_llm).build_plan("m", "вопрос", history=[])
    assert plan.document_names == ["СП 42.13330"]  # empties dropped
    assert plan.block == "amendment"  # lowercased
    assert plan.types == ["clause", "table"]  # lowercased


async def test_planner_drops_invalid_block_and_empty_filters(fake_llm):
    fake_llm.json_responses = [plan_json(document_names=[], block="bogus", types=None)]
    plan = await RetrievalPlanner(fake_llm).build_plan("m", "вопрос", history=[])
    assert plan.document_names is None
    assert plan.block is None  # unknown block → dropped
    assert plan.types is None


async def test_planner_falls_back_to_user_query_when_blank(fake_llm):
    fake_llm.json_responses = [plan_json(search_query="   ")]
    plan = await RetrievalPlanner(fake_llm).build_plan(
        "m", "исходный вопрос", history=[]
    )
    assert plan.search_query == "исходный вопрос"


async def test_planner_injects_critique_and_prev_query_on_revision(fake_llm):
    fake_llm.json_responses = [plan_json()]
    await RetrievalPlanner(fake_llm).build_plan(
        "m", "q", history=[], prev_critique="мало деталей", prev_query="старый запрос"
    )
    system_prompt = fake_llm.chat_calls[-1].messages[0]["content"]
    assert "мало деталей" in system_prompt
    assert "старый запрос" in system_prompt


async def test_planner_retries_on_invalid_json(fake_llm):
    fake_llm.json_responses = ["not json at all", plan_json(search_query="ок")]
    plan = await RetrievalPlanner(fake_llm).build_plan("m", "q", history=[])
    assert plan.search_query == "ок"
    assert len(fake_llm.chat_calls) == 2


async def test_critic_parses_rejection(fake_llm):
    fake_llm.json_responses = [
        verdict_json(
            satisfied=False, critique="нет источников", refined_search_query="новый"
        )
    ]
    verdict = await AnswerCritic(fake_llm).review("m", "q", "ctx", "answer")
    assert verdict.satisfied is False
    assert verdict.critique == "нет источников"
    assert verdict.refined_search_query == "новый"


async def test_critic_fails_open_on_invalid_json(fake_llm):
    # 3 attempts (retries=2) all invalid → critic returns satisfied to avoid an infinite loop
    fake_llm.json_responses = ["garbage", "garbage", "garbage"]
    verdict = await AnswerCritic(fake_llm).review("m", "q", "ctx", "answer")
    assert verdict.satisfied is True
    assert len(fake_llm.chat_calls) == 3
