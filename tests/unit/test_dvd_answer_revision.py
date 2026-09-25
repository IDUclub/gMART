"""A rejected draft is repaired at the lines the critic named, not rewritten."""

import json

from src.agents.services.dvd.answer_revision import (
    AnswerRevision,
    LineAddition,
    LineEdit,
    apply_revision,
)
from src.agents.services.dvd.dvd_reasoning import AnswerCritic
from src.agents.services.service_entities.dvd_plan import AuditedClaim, ClaimEvidence
from tests.helpers import FakeDvdMcpClient, answer_text, plan_json
from tests.unit.test_dvd_rag_service import _run

SOURCE = (
    "6.1.11 Кровлю гостиниц проектируют с учетом СП 17.13330. "
    "В гостиницах должен быть обеспечен доступ для МГН."
)
ROOF = "Кровлю проектируют по пункту 6.1.14 [1]."
FIXED_ROOF = "Кровлю проектируют по пункту 6.1.11 [1]."
ACCESS = "В гостиницах обеспечивается доступ для МГН [1]."


def audit(claims, *, satisfied=False, corrections=(), refined=None, missing=()):
    return json.dumps(
        {
            "unsupported_claims": [],
            "missing_requirements": list(missing),
            "corrections": list(corrections),
            "satisfied": satisfied,
            "critique": "" if satisfied else "неверный пункт",
            "refined_search_query": refined,
            "claims": [
                {
                    "text": text,
                    "status": status,
                    "evidence": [{"source_id": "[1]", "quote": quote}] if quote else [],
                }
                for text, status, quote in claims
            ],
        },
        ensure_ascii=False,
    )


ROOF_FIX = {
    "target": ROOF,
    "instruction": "замени пункт 6.1.14 на 6.1.11",
    "source_id": "[1]",
    "quote": "6.1.11 Кровлю гостиниц проектируют с учетом СП 17.13330.",
}


def test_revision_changes_only_named_lines_and_keeps_list_markers():
    draft = f"Требования:\n- {ROOF}\n- {ACCESS}\n1. Лишнее утверждение [1]."
    revised = apply_revision(
        draft,
        AnswerRevision(
            edits=[
                LineEdit(target=ROOF, replacement=FIXED_ROOF),
                LineEdit(target="Лишнее утверждение [1].", replacement=""),
            ],
            additions=[
                LineAddition(after=ACCESS, text="Число номеров для МГН [1]."),
                LineAddition(after="", text="Итог [1]."),
            ],
        ),
    )
    assert revised == (
        f"Требования:\n- {FIXED_ROOF}\n- {ACCESS}\n- Число номеров для МГН [1].\n"
        "- Итог [1]."
    )


def test_fixed_line_keeps_its_source_labels():
    revised = apply_revision(
        f"- {ROOF}",
        AnswerRevision(
            edits=[LineEdit(target=ROOF, replacement="Кровлю проектируют по 6.1.11.")]
        ),
    )
    assert revised == "- Кровлю проектируют по 6.1.11. [1]"


async def test_local_corrections_do_not_request_new_evidence(fake_llm):
    fake_llm.json_responses = [
        audit(
            [(ROOF, "contradicted", ""), (ACCESS, "supported", SOURCE[-50:])],
            corrections=[ROOF_FIX],
        )
    ]
    verdict = await AnswerCritic(fake_llm).review(
        "m",
        "Требования к гостиницам?",
        f"[1] СП 257\n{SOURCE}",
        f"- {ROOF}\n- {ACCESS}",
    )
    assert not verdict.satisfied and not verdict.needs_evidence
    assert [c.target for c in verdict.corrections] == [ROOF]
    schema = fake_llm.chat_calls[0].format["$defs"]["Correction"]
    assert schema["properties"]["target"]["enum"] == [ROOF, ACCESS, ""]
    assert schema["properties"]["source_id"]["enum"] == ["[1]", ""]


async def test_suggested_search_still_asks_for_new_evidence(fake_llm):
    fake_llm.json_responses = [
        audit(
            [(ROOF, "insufficient", "")],
            corrections=[ROOF_FIX],
            refined="кровли гостиниц",
        )
    ]
    verdict = await AnswerCritic(fake_llm).review(
        "m", "q", f"[1] СП 257\n{SOURCE}", ROOF
    )
    assert verdict.needs_evidence


async def test_minor_remarks_of_an_accepting_critic_are_not_corrections(fake_llm):
    fake_llm.json_responses = [
        audit(
            [(ACCESS, "supported", SOURCE[-50:])],
            satisfied=True,
            corrections=[{"target": ACCESS, "instruction": "можно подробнее"}],
        )
    ]
    verdict = await AnswerCritic(fake_llm).review(
        "m", "q", f"[1] СП 257\n{SOURCE}", ACCESS
    )
    assert verdict.satisfied and verdict.corrections == []


async def test_verified_lines_are_not_audited_again(fake_llm):
    fake_llm.json_responses = [
        audit([(FIXED_ROOF, "supported", SOURCE[:50])], satisfied=True)
    ]
    kept = AuditedClaim(
        text=ACCESS,
        status="supported",
        evidence=[ClaimEvidence(source_id="[1]", quote=SOURCE[-50:])],
    )
    verdict = await AnswerCritic(fake_llm).review(
        "m",
        "q",
        f"[1] СП 257\n{SOURCE}",
        f"- {FIXED_ROOF}\n- {ACCESS}",
        verified=[kept],
    )
    call = fake_llm.chat_calls[0]
    choices = call.format["$defs"]["AuditedClaim"]["properties"]["text"]
    assert choices["enum"] == [FIXED_ROOF]
    assert "already_verified_lines" in call.messages[-1]["content"]
    assert verdict.satisfied and kept in verdict.claims


async def test_literal_defects_become_corrections(fake_llm):
    verdict = await AnswerCritic(fake_llm).review(
        "m", "q", f"[1] СП 257\n{SOURCE}", "Кровля по правилам [N]."
    )
    assert not verdict.satisfied and verdict.corrections
    assert "[N]" in verdict.corrections[0].instruction
    assert fake_llm.chat_calls == []


async def test_rejected_draft_is_repaired_without_redrafting(service, fake_llm):
    client = FakeDvdMcpClient(hits_per_call=[[{"name": "СП 257", "text": SOURCE}]])
    fake_llm.json_responses = [
        plan_json(),
        audit(
            [(ROOF, "contradicted", ""), (ACCESS, "supported", SOURCE[-50:])],
            corrections=[ROOF_FIX],
        ),
        json.dumps({"edits": [{"target": ROOF, "replacement": FIXED_ROOF}]}),
        audit([(FIXED_ROOF, "supported", SOURCE[:50])], satisfied=True),
    ]
    fake_llm.answer_texts = [f"- {ROOF}\n- {ACCESS}"]
    events = await _run(service, client)

    assert answer_text(events) == f"- {FIXED_ROOF}\n- {ACCESS}"
    # One drafted answer and one retrieval: the repair reused both.
    assert len([c for c in fake_llm.chat_calls if c.stream]) == 1
    assert len(client.search_calls) == 1
    revision, review = [c for c in fake_llm.chat_calls if not c.stream][-2:]
    assert "замени пункт 6.1.14 на 6.1.11" in revision.messages[-1]["content"]
    audited = review.format["$defs"]["AuditedClaim"]["properties"]["text"]
    assert audited["enum"] == [FIXED_ROOF]
    statuses = [e["content"].get("text") or "" for e in events if e["type"] == "status"]
    assert any("Исправляю отмеченные места" in m for m in statuses)


async def test_repair_that_changes_nothing_falls_back_to_a_new_draft(service, fake_llm):
    client = FakeDvdMcpClient(hits_per_call=[[{"name": "СП 257", "text": SOURCE}]])
    fake_llm.json_responses = [
        plan_json(),
        audit([(ROOF, "contradicted", "")], corrections=[ROOF_FIX]),
        json.dumps({"edits": []}),
        audit([(FIXED_ROOF, "supported", SOURCE[:50])], satisfied=True),
    ]
    fake_llm.answer_texts = [ROOF, FIXED_ROOF]
    events = await _run(service, client)
    assert answer_text(events) == FIXED_ROOF
    assert len([c for c in fake_llm.chat_calls if c.stream]) == 2


async def test_re_review_checks_the_edits_and_raises_no_new_omissions(fake_llm):
    from src.agents.services.service_entities.dvd_plan import Correction

    fake_llm.json_responses = [audit([(FIXED_ROOF, "supported", SOURCE[:50])])]
    await AnswerCritic(fake_llm).review(
        "m",
        "q",
        f"[1] СП 257\n{SOURCE}",
        FIXED_ROOF,
        verified=[],
        previous=[Correction(**ROOF_FIX)],
        removed=[ACCESS],
    )
    call = fake_llm.chat_calls[0]
    assert call.format["properties"]["missing_requirements"]["maxItems"] == 0
    assert "RE-REVIEW" in call.messages[0]["content"]
    payload = call.messages[-1]["content"]
    assert "замени пункт 6.1.14 на 6.1.11" in payload and ACCESS in payload
    assert "removed_lines" in payload


async def test_rejected_line_without_a_correction_gets_one(fake_llm):
    fake_llm.json_responses = [
        audit([(ROOF, "insufficient", ""), (ACCESS, "supported", SOURCE[-50:])])
    ]
    verdict = await AnswerCritic(fake_llm).review(
        "m", "q", f"[1] СП 257\n{SOURCE}", f"- {ROOF}\n- {ACCESS}"
    )
    # Repaired in place instead of a new search or a whole new draft.
    assert [c.target for c in verdict.corrections] == [ROOF]
    assert "удали утверждение" in verdict.corrections[0].instruction
    assert not verdict.needs_evidence
