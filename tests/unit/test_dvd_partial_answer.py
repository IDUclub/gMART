"""Rejected drafts can contribute only explicitly verified facts to a partial answer."""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.services.dvd.runs import stream_document_run
from tests.helpers import FakeDvdMcpClient, answer_text, plan_json
from tests.unit.test_dvd_rag_service import _run


async def test_live_critic_schema_keeps_draft_wording_and_conditions(fake_llm):
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic
    from src.agents.services.dvd.partial_answer import PartialAnswerEvidence

    source = "Учебные помещения размещаются не выше третьего этажа, если иное не определено Правилами."
    claim = "Помещения допускаются до 3-го этажа включительно, если Правила не предусматривают исключение. [1]"
    draft = f"1. {claim}\n- Все школы должны иметь дирижабль."
    fake_llm.json_responses = [audit(claim, source)]
    fake_llm.chat = AsyncMock(wraps=fake_llm.chat)
    critic = AnswerCritic(fake_llm)

    verdict = await critic.review(
        "test", "Где размещать помещения?", f"[1] Правила\n{source}", draft
    )
    call = fake_llm.chat.call_args.kwargs
    choices = call["format"]["$defs"]["AuditedClaim"]["properties"]["text"]["enum"]
    assert choices == [claim, "Все школы должны иметь дирижабль."]
    assert source not in choices
    assert claim in call["messages"][-1]["content"]
    evidence = PartialAnswerEvidence()
    evidence.add(
        verdict.claims, draft, f"[1] Правила\n{source}", critic._literal_defects
    )
    assert len(evidence.candidates()) == 1
    assert source in evidence.render([0])
    assert claim.split(" [1]")[0] not in evidence.render([0])


async def test_partial_output_preserves_source_scope_despite_false_model_approval(
    service, fake_llm
):
    source = "Учебные помещения для младшего школьного возраста размещаются не выше третьего этажа, если иное не определено Правилами."
    unsafe = "Любая школа должна быть не выше трёх этажей."
    client = FakeDvdMcpClient(default_hits=[{"name": "Правила", "text": source}])
    # Regression from the live gpt-oss audit: both model stages can mistakenly
    # approve the broader assertion even though the quoted evidence is correct.
    fake_llm.json_responses = (
        [plan_json()]
        + [audit(unsafe, source)] * 2
        + [json.dumps({"approved_ids": [0]})]
    )
    fake_llm.answer_texts = [unsafe] * 2
    events = await _run(service, client)
    answer = answer_text(events)
    assert unsafe not in answer
    assert source in answer
    assert "Дословные выдержки" in answer
    assert events[-1]["content"]["done"]


def audit(text, quote, status="supported", label="[1]", refined=None):
    return json.dumps(
        {
            "satisfied": False,
            "critique": "PRIVATE: остальная часть не подтверждена",
            "refined_search_query": refined,
            "claims": [
                {
                    "text": text,
                    "status": status,
                    "evidence": [{"source_id": label, "quote": quote}],
                }
            ],
        },
        ensure_ascii=False,
    )


async def test_partial_answer_uses_verified_claims_from_all_rounds(service, fake_llm):
    facts = [
        "Требование А действует.",
        "Требование Б действует.",
        "Требование В действует.",
    ]
    client = FakeDvdMcpClient(
        hits_per_call=[
            [{"name": f"Документ {i}", "text": fact, "id": str(i)}]
            for i, fact in enumerate(facts, 1)
        ]
    )
    # Each rejection proposes a new query, so every round reads new sources.
    fake_llm.json_responses = [plan_json(search_query="q0")]
    for i, fact in enumerate(facts):
        fake_llm.json_responses.append(audit(fact + " [1]", fact, refined=f"q{i + 1}"))
    fake_llm.json_responses += [json.dumps({"approved_ids": [0, 1, 2]})]
    fake_llm.answer_texts = [fact + " [1]\nНЕПРОВЕРЕННЫЙ текст." for fact in facts]
    events = await _run(service, client)
    answer = answer_text(events)
    assert "удалось подтвердить" in answer
    assert all(fact in answer for fact in facts)
    assert all(f"Документ {i}" in answer for i in (1, 2, 3))
    assert all(f"[{i}]" in answer for i in (1, 2, 3))
    assert "НЕПРОВЕРЕННЫЙ" not in answer and "PRIVATE" not in json.dumps(events)
    assert events[-1]["content"]["done"]
    rid = events[0]["content"]["request_id"]
    assert (await service.state_store.get_state(rid))["status"] == "done"
    replay = await service.state_store.get_buffered_events(rid)
    assert answer_text(replay) == answer
    assert len([c for c in fake_llm.chat_calls if c.stream]) == 3
    # Final validation selects IDs; it does not generate prose.
    assert "approved_ids" in fake_llm.chat_calls[-1].messages[0]["content"]


@pytest.mark.parametrize(
    "kind",
    [
        "insufficient",
        "contradicted",
        "invented_quote",
        "unknown_source",
        "not_in_draft",
    ],
)
async def test_no_verified_claims_returns_honest_empty_answer(
    service, fake_llm, fake_mcp, kind
):
    fact = fake_mcp.default_hits[0]["text"]
    review = audit(
        fact + " [1]",
        fact,
        status=kind if kind in ("insufficient", "contradicted") else "supported",
        label="[99]" if kind == "unknown_source" else "[1]",
    )
    if kind == "invented_quote":
        review = audit(fact + " [1]", "Цитата, которой нет.")
    repairable = kind in ("insufficient", "contradicted")
    # A rejected line gets a local correction; a repair that changes nothing
    # falls back to a new draft, and to the partial answer once drafts run out.
    no_edit = json.dumps({"edits": []})
    fake_llm.json_responses = (
        [plan_json(), review, no_edit, review, no_edit]
        if repairable
        else [plan_json(), review, review]
    )
    fake_llm.answer_texts = [
        "Другой черновик [1]" if kind == "not_in_draft" else fact + " [1]"
    ] * 2
    events = await _run(service, fake_mcp)
    assert "Не удалось подтвердить" in answer_text(events)
    assert fact not in answer_text(events)
    assert not any(e["type"] == "error" for e in events)
    # plan + 2 × (draft, review) [+ 2 repairs]; no final request without candidates
    assert len(fake_llm.chat_calls) == (7 if repairable else 5)


async def test_final_selection_cannot_include_contradicted_claim(
    service, fake_llm, fake_mcp
):
    fact = fake_mcp.default_hits[0]["text"]
    fake_llm.json_responses = [
        plan_json(),
        audit(fact + " [1]", fact, refined="r2"),
        audit(fact + " [1]", fact, "contradicted", refined="r3"),
        audit(fact + " [1]", fact, "insufficient"),
    ]
    fake_llm.answer_texts = [fact + " [1]"] * 3
    events = await _run(service, fake_mcp)
    assert "Не удалось подтвердить" in answer_text(events)
    assert fact not in answer_text(events)


async def test_final_selector_can_remove_conflicting_supported_claims(
    service, fake_llm, fake_mcp
):
    fact = fake_mcp.default_hits[0]["text"]
    fake_llm.json_responses = (
        [plan_json()]
        + [audit(fact + " [1]", fact)] * 2
        + [json.dumps({"approved_ids": []})]
    )
    fake_llm.answer_texts = [fact + " [1]"] * 2
    events = await _run(service, fake_mcp)
    assert "Не удалось подтвердить" in answer_text(events)


async def test_fabricated_selection_is_technical_error_not_partial_success(
    service, fake_llm, fake_mcp
):
    fact = fake_mcp.default_hits[0]["text"]
    fake_llm.json_responses = (
        [plan_json()]
        + [audit(fact + " [1]", fact)] * 2
        + [json.dumps({"approved_ids": [999]})]
    )
    fake_llm.answer_texts = [fact + " [1]"] * 2
    events = [
        e
        async for e in stream_document_run(
            service,
            model="m",
            dvd_mcp_client=fake_mcp,
            token=None,
            user_query="вопрос",
            temperature=0,
            persist_history=False,
        )
    ]
    assert events[-1]["type"] == "error"
    assert not any(e["type"] == "chunk" for e in events)
    assert events[-1]["content"]["traceback"] == ""


async def test_resume_restores_verified_evidence_without_local_label_collision(
    service, fake_llm, fake_mcp
):
    from src.agents.services.dvd.dvd_reasoning import AnswerCritic
    from src.agents.services.dvd.partial_answer import PartialAnswerEvidence
    from src.agents.services.service_entities.dvd_plan import AuditedClaim

    old = "Первый подтверждённый факт."
    ledger = PartialAnswerEvidence()
    ledger.add(
        [
            AuditedClaim.model_validate(
                json.loads(audit(old + " [1]", old))["claims"][0]
            )
        ],
        old + " [1]",
        "[1] Старый документ\n" + old,
        AnswerCritic._literal_defects,
    )
    rid = service.state_store.new_request_id()
    await service.state_store.create(
        rid, chat_id=None, user_query="q", scenario_id=None, model="m", temperature=0
    )
    await service.state_store.save_checkpoint(
        rid,
        "qa_progress",
        {
            "completed_iterations": 2,
            "accepted": False,
            "partial_evidence": ledger.records,
        },
    )
    fact = fake_mcp.default_hits[0]["text"]
    fake_llm.json_responses = [
        plan_json(),
        audit(fact + " [1]", fact),
        json.dumps({"approved_ids": [0, 1]}),
    ]
    fake_llm.answer_texts = [fact + " [1]"]
    events = await _run(
        service, fake_mcp, request_id=rid, chat_id=None, persist_history=False
    )
    answer = answer_text(events)
    assert old in answer and fact in answer
    assert "Старый документ" in answer and "СП 42.13330.2016" in answer
    assert "[1]" in answer and "[2]" in answer
    checkpoint = await service.state_store.get_checkpoint(rid)
    assert checkpoint["qa_progress"]["final_answer"] == answer
    assert len(checkpoint["qa_progress"]["partial_evidence"]) == 2


async def test_final_empty_search_can_use_earlier_verified_evidence(service, fake_llm):
    fact = "Подтверждённый факт."
    client = FakeDvdMcpClient(
        hits_per_call=[[{"name": "Источник", "text": fact}], [], []]
    )
    # The critic's query finds nothing; only then does the planner run again.
    fake_llm.json_responses = [
        plan_json(search_query="q1"),
        audit(fact + " [1]", fact, refined="q2"),
        plan_json(search_query="q3"),
        json.dumps({"approved_ids": [0]}),
    ]
    fake_llm.answer_texts = [fact + " [1]"]
    events = await _run(service, client)
    assert fact in answer_text(events)
    assert "частичный ответ" in answer_text(events)
    assert not any(e["type"] == "error" for e in events)


async def test_conflicting_rounds_reach_final_audit_with_source_provenance(
    service, fake_llm
):
    facts = [
        "Значение должно быть не более 10.",
        "Значение должно быть не менее 20.",
        "Проверка обязательна.",
    ]
    client = FakeDvdMcpClient(
        hits_per_call=[
            [{"name": f"Источник {i}", "version": "2026", "text": fact}]
            for i, fact in enumerate(facts)
        ]
    )
    fake_llm.json_responses = [plan_json(search_query="q0")]
    for i, fact in enumerate(facts):
        fake_llm.json_responses.append(audit(fact + " [1]", fact, refined=f"q{i + 1}"))
    fake_llm.json_responses += [json.dumps({"approved_ids": [2]})]
    fake_llm.answer_texts = [fact + " [1]" for fact in facts]
    events = await _run(service, client)
    assert facts[2] in answer_text(events)
    assert all(fact not in answer_text(events) for fact in facts[:2])
    payload = json.loads(fake_llm.chat_calls[-1].messages[1]["content"])
    assert payload["candidate_ids"] == [0, 1, 2]
    assert [r["text"] for r in payload["records"]] == facts
    assert all(
        r["evidence"][0]["quote"] == facts[i] for i, r in enumerate(payload["records"])
    )
    assert all(
        f"Источник {i}" in r["evidence"][0]["source"]
        for i, r in enumerate(payload["records"])
    )


async def test_partial_selection_transport_failure_still_emits_generic_error(
    service, fake_llm, fake_mcp
):
    fact = fake_mcp.default_hits[0]["text"]
    fake_llm.json_responses = [plan_json(), audit(fact + " [1]", fact)] * 3
    fake_llm.answer_texts = [fact + " [1]"] * 3
    original = fake_llm.chat

    async def failing_chat(*args, **kwargs):
        if len(fake_llm.chat_calls) == 9:
            raise RuntimeError("PRIVATE_PARTIAL_REVIEW_FAILURE")
        return await original(*args, **kwargs)

    fake_llm.chat = failing_chat
    events = [
        e
        async for e in stream_document_run(
            service,
            model="m",
            dvd_mcp_client=fake_mcp,
            token=None,
            user_query="q",
            temperature=0,
            persist_history=False,
        )
    ]
    assert events[-1]["type"] == "error"
    assert "PRIVATE_PARTIAL" not in json.dumps(events)
    assert not any(e["type"] == "chunk" for e in events)
