"""Stage 5: a narrow first retrieval, widened only when evidence is missing."""

from __future__ import annotations

from tests.helpers import FakeDvdMcpClient, plan_json, verdict_json


async def _run(service, mcp):
    return [
        event
        async for event in service.run_document_qa_pipeline(
            dvd_mcp_client=mcp,
            token="tok",
            model="m",
            temperature=0.0,
            user_query="нормы озеленения",
            chat_id="chat-1",
        )
    ]


async def test_first_pass_is_narrow(service, fake_llm, fake_mcp):
    fake_llm.json_responses = [
        plan_json(limit=12, context_height=3),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["Ответ [1]."]
    await _run(service, fake_mcp)
    (call,) = fake_mcp.search_calls
    assert (call.limit, call.context_height) == (6, 1)


async def test_missing_evidence_widens_to_planner_then_broad(service, fake_llm):
    hit = {"id": "a", "name": "СП 55", "numbering": "9.1", "text": "Текст."}
    other = {"id": "b", "name": "СП 55", "numbering": "9.2", "text": "Ещё."}
    third = {"id": "c", "name": "СП 55", "numbering": "9.3", "text": "Третий."}
    mcp = FakeDvdMcpClient(hits_per_call=[[hit], [hit, other], [hit, other, third]])
    fake_llm.json_responses = [
        plan_json(limit=12, context_height=3),
        verdict_json(satisfied=False, critique="нет нормы"),
        verdict_json(satisfied=False, critique="нет нормы"),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["Ответ [1].", "Ответ [1] [2].", "Ответ [1] [2] [3]."]

    def missing_evidence(verdict):
        verdict.needs_evidence = True
        return verdict

    review = service.critic.review

    async def critic(*args, **kwargs):
        verdict = await review(*args, **kwargs)
        return verdict if verdict.satisfied else missing_evidence(verdict)

    service.critic.review = critic
    await _run(service, mcp)
    sizes = [(c.limit, c.context_height) for c in mcp.search_calls]
    # Narrow first pass, then the planner's own sizes, then the broadest search.
    assert sizes == [(6, 1), (12, 3), (20, 3)]


async def test_small_plan_and_document_lists_keep_their_sizes(
    service, fake_llm, fake_mcp
):
    from src.agents.services.dvd.dvd_rag_service import DvdRagService
    from src.agents.services.service_entities.dvd_plan import validate_retrieval_plan

    small = validate_retrieval_plan(
        {"retrieval_mode": "semantic", "limit": 4, "context_height": 0}
    )
    assert DvdRagService._first_pass(small) is None
    listing = validate_retrieval_plan(
        {"retrieval_mode": "semantic", "limit": 18, "intent": "document_list"}
    )
    assert DvdRagService._first_pass(listing) is None
    exact = validate_retrieval_plan(
        {"retrieval_mode": "structure", "pattern": "9.1", "limit": 18}
    )
    assert DvdRagService._first_pass(exact) is None


async def test_switch_keeps_planner_sizes(service, fake_llm, fake_mcp, monkeypatch):
    monkeypatch.setenv("DVD_SMALL_FIRST_RETRIEVAL", "false")
    fake_llm.json_responses = [
        plan_json(limit=12, context_height=3),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["Ответ [1]."]
    await _run(service, fake_mcp)
    (call,) = fake_mcp.search_calls
    assert (call.limit, call.context_height) == (12, 3)
