"""Stage 4: the critic reuses the answer's evidence instead of reducing it again."""

from __future__ import annotations

from unittest.mock import AsyncMock

from src.agents.services.dvd.dvd_context import SOURCE_SEPARATOR, source_records
from src.agents.services.dvd.evidence_set import cited_context
from tests.helpers import plan_json, verdict_json

CONTEXT = "".join(
    f"[{n}] СП 55, п. 9.{n}\nТекст пункта 9.{n}.{SOURCE_SEPARATOR}" for n in (1, 2, 3)
)


def test_only_cited_sources_are_kept_verbatim():
    answer = "- Требование первое [1].\n- Требование третье [3]."
    cited = cited_context(CONTEXT, answer)
    records = source_records(cited)
    assert list(records) == ["[1]", "[3]"]
    assert records["[3]"] == source_records(CONTEXT)["[3]"]


def test_quotation_labels_do_not_widen_the_selection():
    answer = (
        "Пункт устанавливает требование [2].\n\nПолная цитата:\n\n"
        "[1] СП 55, п. 9.1\n\n> Текст пункта 9.1."
    )
    assert list(source_records(cited_context(CONTEXT, answer))) == ["[2]"]


def test_full_evidence_is_reopened_when_a_claim_is_not_mapped():
    # A claim line without a source label.
    assert cited_context(CONTEXT, "Требование первое [1].\nЕщё вывод.") is None
    # A label that is not a source.
    assert cited_context(CONTEXT, "Требование [7].") is None
    # A statement that something is absent concerns every source.
    assert cited_context(CONTEXT, "Во фрагментах нет сведений о школах [1].") is None
    # Every source cited: nothing to leave out.
    assert cited_context(CONTEXT, "Первое [1].\nВторое [2].\nТретье [3].") is None


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


def _three_hits(fake_mcp):
    fake_mcp.default_hits = [
        {"id": str(n), "name": "СП 55", "numbering": f"9.{n}", "text": f"Текст {n}."}
        for n in (1, 2, 3)
    ]


async def test_oversized_review_audits_cited_sources_without_reduction(
    service, fake_llm, fake_mcp
):
    _three_hits(fake_mcp)
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Требование второе [2]."]
    service.context_reducer.fits = AsyncMock(return_value=False)
    seen = {}
    review = service.critic.review

    async def spy(model, query, context, answer, **kwargs):
        seen.update(context=context, literal=kwargs.get("literal_context"))
        return await review(model, query, context, answer, **kwargs)

    service.critic.review = spy
    prepare = service.context_reducer.prepare
    service.context_reducer.prepare = AsyncMock(wraps=prepare)
    await _run(service, fake_mcp)

    assert list(source_records(seen["context"])) == ["[2]"]
    # Literal label/table checks still see every source of the draft.
    assert set(source_records(seen["literal"])) == {"[1]", "[2]", "[3]"}
    review_call = service.context_reducer.prepare.await_args_list[-1]
    assert list(source_records(review_call.args[2])) == ["[2]"]
    # The cited sources fit: no evidence-reduction call reached the LLM, only the
    # planner and the critic (structured) and the draft (streamed).
    assert len([c for c in fake_llm.chat_calls if not c.stream]) == 2


async def test_fitting_review_keeps_the_whole_evidence(service, fake_llm, fake_mcp):
    _three_hits(fake_mcp)
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Требование второе [2]."]
    seen = {}
    review = service.critic.review

    async def spy(model, query, context, answer, **kwargs):
        seen["context"] = context
        return await review(model, query, context, answer, **kwargs)

    service.critic.review = spy
    await _run(service, fake_mcp)
    assert set(source_records(seen["context"])) == {"[1]", "[2]", "[3]"}


async def test_full_policy_reduces_the_review_as_before(
    service, fake_llm, fake_mcp, monkeypatch
):
    monkeypatch.setenv("DVD_CRITIC_CONTEXT", "full")
    _three_hits(fake_mcp)
    fake_llm.json_responses = [plan_json(), verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Требование второе [2]."]
    prepare = service.context_reducer.prepare
    service.context_reducer.prepare = AsyncMock(wraps=prepare)
    await _run(service, fake_mcp)
    # Sources once, then the review over the whole evidence.
    assert service.context_reducer.prepare.await_count == 2
    review_call = service.context_reducer.prepare.await_args_list[1]
    assert set(source_records(review_call.args[2])) == {"[1]", "[2]", "[3]"}
