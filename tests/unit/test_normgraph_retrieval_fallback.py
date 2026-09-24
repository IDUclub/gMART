"""Norms-QA retries must change the query, and a failed plan must not fail the step."""

import pytest

from src.agents.model_clients.llm_base import LlmResponseError
from src.agents.services.normgraph.normgraph_rag_service import NormGraphRagService
from src.agents.services.normgraph.normgraph_reasoning import PLACEMENT_KINDS
from src.agents.services.service_entities.normgraph_plan import (
    NormGraphCriticVerdict,
    NormGraphPlan,
    PrimaryTool,
)

HIT = {"id": "r1", "document_name": "СП 42.13330.2016", "clause_number": "11.34"}


class Mcp:
    def __init__(self):
        self.calls = []

    async def restrictions_applicable(self, **arguments):
        self.calls.append(("restrictions_applicable", arguments))
        return {"hits": []}

    async def search_restrictions(self, **arguments):
        self.calls.append(("search_restrictions", arguments))
        return {"hits": [HIT]}


class Planner:
    def __init__(self, *plans):
        self.plans = list(plans)

    async def build_plan(self, *args):
        plan = self.plans.pop(0)
        if isinstance(plan, Exception):
            raise plan
        return plan


class Critic:
    async def review(self, *args):
        return NormGraphCriticVerdict(satisfied=True)


@pytest.fixture
def norms(monkeypatch, fake_llm, fake_urban, state_store):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    service = NormGraphRagService("http://llm", None, fake_urban, state_store)
    service.critic = Critic()

    async def answer(*args):
        yield service._chunk("Не менее 10 м [1].", done=False, iteration=args[5])

    service._generate_answer = answer
    return service


async def run(service, mcp):
    collected = {"final_answer": "", "tool_calls": [], "newly_completed": False}
    events = [
        event
        async for event in service._run_qa_loop(
            mcp, "model", 0, "Расстояние от окон до автостоянок?", [], collected, "rid"
        )
    ]
    return events, collected


def applicable():
    return NormGraphPlan(
        primary_tool=PrimaryTool.APPLICABLE,
        search_query="расстояние от окон до автостоянок",
        object="окна жилых домов",
        subject="открытые автостоянки",
    )


async def test_empty_object_lookup_is_retried_as_text_search(norms):
    norms.planner = Planner(applicable(), applicable())
    mcp = Mcp()

    _, collected = await run(norms, mcp)

    assert [name for name, _ in mcp.calls] == [
        "restrictions_applicable",
        "search_restrictions",
    ]
    arguments = mcp.calls[1][1]
    assert arguments["query"] == "расстояние от окон до автостоянок"
    assert "object" not in arguments and "subject" not in arguments
    assert collected["final_answer"] == "Не менее 10 м [1]."


async def test_planner_failure_falls_back_to_text_search(norms):
    norms.planner = Planner(
        applicable(), LlmResponseError("output budget exhausted", 502)
    )
    mcp = Mcp()

    _, collected = await run(norms, mcp)

    assert mcp.calls[1] == (
        "search_restrictions",
        {
            "query": "расстояние от окон до автостоянок",
            "kinds": list(PLACEMENT_KINDS),
            "limit": 10,
            "neighbors_depth": 0,
        },
    )
    assert collected["final_answer"] == "Не менее 10 м [1]."


class EmptyMcp(Mcp):
    async def search_restrictions(self, **arguments):
        self.calls.append(("search_restrictions", arguments))
        return {"hits": [HIT] if "kinds" not in arguments else []}


def search(query="расстояние от окон до автостоянок"):
    return NormGraphPlan(search_query=query)


async def test_placement_question_filters_by_placement_kinds_until_they_find_nothing(
    norms,
):
    norms.planner = Planner(search(), search())
    mcp = EmptyMcp()

    _, collected = await run(norms, mcp)

    assert [arguments.get("kinds") for _, arguments in mcp.calls] == [
        list(PLACEMENT_KINDS),
        None,
    ]
    assert collected["final_answer"] == "Не менее 10 м [1]."


async def test_other_questions_keep_every_kind(norms):
    norms.planner = Planner(search("что такое красная линия"))
    mcp = Mcp()
    collected = {"final_answer": "", "tool_calls": [], "newly_completed": False}

    async for _ in norms._run_qa_loop(
        mcp, "model", 0, "Что такое красная линия?", [], collected, "rid"
    ):
        pass

    assert "kinds" not in mcp.calls[0][1]
