"""Compact line audits preserve verdicts while sharing model decode capacity."""

import asyncio
import json
from collections import Counter

import pytest

from src.agents.services.dvd import dvd_reasoning
from src.agents.services.dvd.answer_risk import AnswerRisk
from src.agents.services.dvd.dvd_reasoning import AnswerCritic, critic_output_cap
from src.agents.services.dvd.retry_policy import CriticBudgetError, CriticResponseError
from src.agents.services.dvd.run_metrics import DvdRunMetrics
from src.agents.services.service_entities.dvd_plan import AuditedClaim

CONTEXT = "[1] Правила\nУсловие применяется при наличии доступа."
LINES = [f"Условие {i} применяется при наличии доступа [1]." for i in range(1, 8)]


class AuditLlm:
    def __init__(self, response=None):
        self.response = response
        self.calls = []
        self.active = 0
        self.peak = 0
        self.attempts = Counter()

    async def model_input_tokens(self, *args, **kwargs):
        return 2000

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["format"]
        ids = schema["$defs"]["LineClaim"]["properties"]["line"].get("enum", [])
        self.attempts[tuple(ids)] += 1
        attempt = self.attempts[tuple(ids)]
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            data = {
                "claims": [
                    {
                        "line": line,
                        "status": "supported",
                        "evidence": [
                            {"source_id": "[1]", "quote": "при наличии доступа"}
                        ],
                    }
                    for line in reversed(ids)
                ],
                "unsupported_claims": [],
                "missing_requirements": [],
                "corrections": [],
                "satisfied": True,
                "critique": "",
                "refined_search_query": None,
            }
            if self.response:
                result = self.response(data, ids, attempt)
                if result is not None:
                    return result
            return {"message": {"content": json.dumps(data, ensure_ascii=False)}}
        finally:
            self.active -= 1


@pytest.fixture(autouse=True)
def critic_settings(monkeypatch):
    monkeypatch.setenv("DVD_CRITIC_LINES_PER_CALL", "3")
    monkeypatch.setenv("DVD_CRITIC_MAX_PARALLEL", "6")
    monkeypatch.delenv("DVD_PARALLEL_CRITIC", raising=False)
    monkeypatch.delenv("DVD_CRITIC_MAX_OUTPUT_TOKENS", raising=False)


async def review(llm, lines=LINES, **kwargs):
    return await AnswerCritic(llm).review(
        "qwen", "Какие условия?", CONTEXT, "\n".join(lines), **kwargs
    )


def line_enum(call):
    return call["format"]["$defs"]["LineClaim"]["properties"]["line"].get("enum", [])


@pytest.mark.parametrize("count", [1, 7])
async def test_critic_request_schema_puts_evidence_before_verdict(count):
    llm = AuditLlm()
    await review(llm, LINES[:count])
    fields = [
        "claims",
        "unsupported_claims",
        "missing_requirements",
        "corrections",
        "satisfied",
        "critique",
        "refined_search_query",
    ]
    for call in llm.calls:
        schema = call["format"]
        assert list(schema["properties"]) == fields
        assert schema["required"] == fields


async def test_compact_ids_map_to_exact_lines_and_correction_targets():
    def reject(data, ids, attempt):
        data.update(
            satisfied=False,
            unsupported_claims=["L1", "L1"],
            corrections=[{"target": "L1", "instruction": "Исправь условие"}],
        )
        data["claims"][0]["status"] = "contradicted"

    llm = AuditLlm(reject)
    verdict = await review(llm, LINES[:1])
    assert verdict.claims[0].text == LINES[0]
    assert verdict.corrections[0].target == LINES[0]
    assert verdict.claims[0].evidence[0].quote == "при наличии доступа"
    schema = llm.calls[0]["format"]
    assert line_enum(llm.calls[0]) == ["L1"]
    assert "text" not in schema["$defs"]["LineClaim"]["properties"]
    assert schema["$defs"]["Correction"]["properties"]["target"]["enum"] == ["L1", ""]
    assert schema["properties"]["unsupported_claims"]["items"]["enum"] == ["L1"]
    assert schema["$defs"]["ClaimEvidence"]["properties"]["source_id"]["enum"] == [
        "[1]"
    ]
    assert not verdict.satisfied


@pytest.mark.parametrize("field", ["claims", "corrections", "unsupported_claims"])
async def test_unknown_id_is_retried_and_rejected(field):
    def invalid(data, ids, attempt):
        data[field] = {
            "claims": [{"line": "L99", "status": "supported"}],
            "corrections": [{"target": "L99", "instruction": "Исправь"}],
            "unsupported_claims": ["Свободный текст вместо ID"],
        }[field]

    llm = AuditLlm(invalid)
    metrics = DvdRunMetrics("invalid")
    with pytest.raises(CriticResponseError):
        await review(llm, LINES[:1], metrics=metrics)
    assert len(llm.calls) == metrics.counters["critic_calls"] == 3
    assert (
        "Твой предыдущий JSON нарушает схему" in llm.calls[1]["messages"][-1]["content"]
    )


async def test_seven_lines_run_three_groups_and_omissions_concurrently():
    llm = AuditLlm()
    metrics = DvdRunMetrics("parallel")
    verdict = await review(llm, metrics=metrics)
    assert verdict.satisfied
    assert [c.text for c in verdict.claims] == LINES
    assert len(llm.calls) == metrics.counters["critic_calls"] == 4
    assert llm.peak > 1 and llm.active == 0
    assert metrics.decisions["critic_groups"] == 3
    assert metrics.decisions["critic_omissions"] is True
    assert [line_enum(c) for c in llm.calls] == [
        ["L1", "L2", "L3"],
        ["L4", "L5", "L6"],
        ["L7"],
        [],
    ]
    for call in llm.calls:
        schema = call["format"]
        ids = line_enum(call)
        targets = schema["$defs"]["Correction"]["properties"]["target"]["enum"]
        assert targets == (ids or [""])
        assert call["options"]["num_predict"] == critic_output_cap(len(ids))
        payload = call["messages"][1]["content"]
        assert CONTEXT in payload and "\n".join(LINES) in payload
        if ids:
            assert schema["properties"]["missing_requirements"]["maxItems"] == 0
        else:
            assert schema["properties"]["claims"]["maxItems"] == 0
            assert schema["properties"]["unsupported_claims"]["maxItems"] == 0


async def test_merge_keeps_group_corrections_before_omissions_and_its_query():
    def defects(data, ids, attempt):
        data["refined_search_query"] = "запрос группы"
        if ids and ids[0] == "L4":
            data.update(
                satisfied=False,
                critique="Неверное условие",
                corrections=[{"target": "L4", "instruction": "Исправь условие"}],
                unsupported_claims=["L4"],
            )
        elif not ids:
            data.update(
                satisfied=False,
                critique="Пропущено требование",
                missing_requirements=["Ещё одно требование"],
                corrections=[{"target": "", "instruction": "Добавь требование"}],
                refined_search_query="запрос пропусков",
            )

    verdict = await review(AuditLlm(defects))
    assert not verdict.satisfied
    assert [c.target for c in verdict.corrections] == [LINES[3], ""]
    assert verdict.critique == "Неверное условие; Пропущено требование"
    assert verdict.refined_search_query == "запрос пропусков"
    assert verdict.needs_evidence


async def test_one_rejected_group_gets_local_corrections_without_new_search():
    def reject(data, ids, attempt):
        if ids and ids[0] == "L4":
            data["satisfied"] = False
            for claim in data["claims"]:
                if claim["line"] == "L4":
                    claim["status"] = "insufficient"

    verdict = await review(AuditLlm(reject))
    assert not verdict.satisfied and not verdict.needs_evidence
    assert [c.target for c in verdict.corrections] == [LINES[3]]
    assert verdict.corrections[0].quote == "при наличии доступа"


async def test_omissions_only_defect_needs_evidence():
    def missing(data, ids, attempt):
        if not ids:
            data.update(satisfied=False, missing_requirements=["Условие применения"])

    verdict = await review(AuditLlm(missing))
    assert not verdict.satisfied and verdict.needs_evidence
    assert verdict.corrections == []
    assert "Условие применения" in verdict.critique


async def test_omissions_query_wins_even_when_empty():
    def query(data, ids, attempt):
        if ids:
            data["refined_search_query"] = "запрос группы"

    verdict = await review(AuditLlm(query))
    assert verdict.refined_search_query is None


async def test_recheck_has_no_omissions_and_uses_first_nonempty_query():
    def query(data, ids, attempt):
        if ids[0] == "L4":
            data["refined_search_query"] = "уточнение"

    llm = AuditLlm(query)
    verdict = await review(llm, previous=[])
    assert len(llm.calls) == 3
    assert all(line_enum(c) for c in llm.calls)
    assert all(
        c["format"]["properties"]["missing_requirements"]["maxItems"] == 0
        for c in llm.calls
    )
    assert verdict.refined_search_query == "уточнение"


@pytest.mark.parametrize("count,flag", [(1, "true"), (3, "true"), (7, "false")])
async def test_small_answer_or_flag_off_uses_one_request(monkeypatch, count, flag):
    monkeypatch.setenv("DVD_PARALLEL_CRITIC", flag)
    llm = AuditLlm()
    await review(llm, LINES[:count])
    assert len(llm.calls) == 1
    assert (
        "maxItems" not in llm.calls[0]["format"]["properties"]["missing_requirements"]
    )


@pytest.mark.parametrize("limit", [1, 2])
async def test_max_parallel_includes_omissions(monkeypatch, limit):
    monkeypatch.setenv("DVD_CRITIC_MAX_PARALLEL", str(limit))
    llm = AuditLlm()
    await review(llm)
    assert llm.peak == limit
    assert len(llm.calls) == 4


async def test_each_request_respects_the_remaining_context_window(monkeypatch):
    monkeypatch.setenv("DVD_CONTEXT_WINDOW_TOKENS", "8192")
    llm = AuditLlm()
    await review(llm)
    assert all(c["options"]["num_ctx"] == 8192 for c in llm.calls)
    assert all(c["options"]["num_predict"] <= 8192 - 2000 - 256 for c in llm.calls)

    async def no_room(*args, **kwargs):
        return 8192 - 256 - 127

    llm.model_input_tokens = no_room
    llm.calls.clear()
    with pytest.raises(CriticResponseError):
        await review(llm)
    assert llm.calls == []


async def test_kept_lines_are_excluded_and_all_verified_forces_empty_claims():
    llm = AuditLlm()
    kept = [AuditedClaim(text=line, status="supported") for line in LINES]
    verdict = await review(llm, verified=kept)
    assert verdict.satisfied and verdict.claims == kept
    assert len(llm.calls) == 1
    assert llm.calls[0]["format"]["properties"]["claims"]["maxItems"] == 0


async def test_recheck_groups_keep_ids_from_full_answer():
    llm = AuditLlm()
    kept = [AuditedClaim(text=LINES[1], status="supported")]
    verdict = await review(llm, verified=kept, previous=[])
    assert [line_enum(c) for c in llm.calls] == [["L1", "L3", "L4"], ["L5", "L6", "L7"]]
    assert kept[0] in verdict.claims


@pytest.mark.parametrize("defect", ["foreign_id", "omission", "addition"])
async def test_group_may_only_audit_and_correct_its_own_lines(defect):
    def invalid(data, ids, attempt):
        if ids and ids[0] == "L1":
            if defect == "foreign_id":
                data["claims"][0]["line"] = "L7"
            elif defect == "omission":
                data["missing_requirements"] = ["Пропуск"]
            else:
                data["corrections"] = [{"target": "", "instruction": "Добавь"}]

    llm = AuditLlm(invalid)
    with pytest.raises(CriticResponseError):
        await review(llm)
    assert llm.active == 0


async def test_literal_defects_short_circuit_without_requests():
    llm = AuditLlm()
    verdict = await review(llm, ["Условие [N].", *LINES])
    assert not verdict.satisfied and llm.calls == []


@pytest.mark.parametrize("which", ["group", "omissions"])
async def test_capped_request_fails_the_review_as_a_budget_error(which):
    def capped(data, ids, attempt):
        if (which == "group" and ids and ids[0] == "L4") or (
            which == "omissions" and not ids
        ):
            return {"done_reason": "length"}

    llm = AuditLlm(capped)
    with pytest.raises(CriticBudgetError):
        await review(llm)
    assert llm.active == 0


async def test_capped_group_retries_at_low_with_same_cap(monkeypatch):
    monkeypatch.setattr(
        dvd_reasoning, "critic_reasoning_effort", lambda *args: "medium"
    )

    def capped(data, ids, attempt):
        if ids and ids[0] == "L4" and attempt == 1:
            return {"done_reason": "length"}

    llm = AuditLlm(capped)
    verdict = await review(llm)
    calls = [c for c in llm.calls if line_enum(c) == ["L4", "L5", "L6"]]
    assert verdict.satisfied and len(llm.calls) == 5
    assert [c["reasoning_effort"] for c in calls] == ["medium", "low"]
    assert {c["options"]["num_predict"] for c in calls} == {critic_output_cap(3)}


async def test_short_audit_confirmation_is_per_group(monkeypatch):
    monkeypatch.setattr(
        dvd_reasoning,
        "critic_reasoning_effort",
        lambda llm, model, risk=None: "medium" if risk is None else "low",
    )

    def confirm(data, ids, attempt):
        if ids and ids[0] == "L1" and attempt == 1:
            for claim in data["claims"]:
                claim["status"] = "insufficient"
            data["satisfied"] = False

    llm = AuditLlm(confirm)
    metrics = DvdRunMetrics("retry")
    verdict = await review(llm, risk=AnswerRisk("high", ("numbers",)), metrics=metrics)
    assert verdict.satisfied
    assert metrics.counters["critic_calls"] == 5
    assert [
        c["reasoning_effort"] for c in llm.calls if line_enum(c) == ["L1", "L2", "L3"]
    ] == ["low", "medium"]


async def test_capped_full_confirmation_keeps_short_group_verdict(monkeypatch):
    monkeypatch.setattr(
        dvd_reasoning,
        "critic_reasoning_effort",
        lambda llm, model, risk=None: "medium" if risk is None else "low",
    )

    def capped(data, ids, attempt):
        if ids and ids[0] == "L1":
            if attempt == 2:
                return {"done_reason": "length"}
            for claim in data["claims"]:
                claim["status"] = "insufficient"
            data["satisfied"] = False

    llm = AuditLlm(capped)
    verdict = await review(llm, risk=AnswerRisk("high", ("numbers",)))
    assert not verdict.satisfied
    assert len(llm.calls) == 5
    assert [c.target for c in verdict.corrections] == LINES[:3]


@pytest.mark.parametrize("value,wanted", [("0", 1), ("-5", 1), ("bad", 3), ("", 3)])
def test_lines_per_call_has_a_minimum_and_default(monkeypatch, value, wanted):
    monkeypatch.setenv("DVD_CRITIC_LINES_PER_CALL", value)
    assert dvd_reasoning._critic_limit("DVD_CRITIC_LINES_PER_CALL", 3) == wanted
