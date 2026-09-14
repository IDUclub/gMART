from copy import deepcopy

import pytest

from tests.integration.industrial.acceptance import verify_result
from tests.integration.industrial.judge import validate_judgment
from tests.integration.industrial.runner import series_verdict
from tests.integration.industrial.scenarios import episodes


@pytest.mark.asyncio
async def test_judge_repairs_missing_proof_once_without_changing_failures():
    import json
    from unittest.mock import AsyncMock, Mock

    from tests.integration.industrial.judge import BASE_RUBRIC, evaluate

    final, context = sample()
    rows = [
        {
            "id": name,
            "verdict": "pass",
            "quote_id": "Q1",
            "evidence_ids": [],
            "reason": "Проверено",
        }
        for name in BASE_RUBRIC
    ]
    rows[0]["verdict"] = "fail"
    invalid = {"choices": [{"message": {"content": json.dumps({"criteria": rows})}}]}
    repaired = deepcopy(rows)
    for row in repaired:
        row["evidence_ids"] = ["E1"]
    repaired[0]["verdict"] = "pass"  # A format repair must not overturn a failure.
    valid = {"choices": [{"message": {"content": json.dumps({"criteria": repaired})}}]}
    http = Mock(
        post=AsyncMock(
            side_effect=[Mock(json=lambda: invalid), Mock(json=lambda: valid)]
        )
    )
    result = await evaluate(
        http,
        {"LLM_BASE_URL": "http://model.test/v1", "LLM_MODEL": "m"},
        {"rubric": []},
        [{"query": "Оцени", "final": final}],
        context,
    )
    assert http.post.await_count == 2
    assert result["verdict"] == "fail"
    assert len(result["attempts"]) == 2
    # The server must require a rationale, not merely parse any JSON object.
    request = http.post.await_args.kwargs["json"]
    assert request["response_format"]["type"] == "json_schema"
    import jsonschema

    schema = request["response_format"]["json_schema"]["schema"]
    without_reasons = deepcopy(repaired)
    for row in without_reasons:
        row.pop("reason")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"criteria": without_reasons}, schema)


@pytest.mark.asyncio
async def test_truncated_judge_reply_gets_output_budget_without_lowering_reasoning():
    import json
    from unittest.mock import AsyncMock, Mock

    from tests.integration.industrial.judge import BASE_RUBRIC, evaluate

    final, context = sample()
    rows = [
        {"id": name, "verdict": "fail", "reason": "Вывод не подтверждён"}
        for name in BASE_RUBRIC
    ]
    first = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
    second = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"criteria": rows})},
            }
        ]
    }
    http = Mock(
        post=AsyncMock(
            side_effect=[Mock(json=lambda: first), Mock(json=lambda: second)]
        )
    )
    result = await evaluate(
        http,
        {"LLM_BASE_URL": "http://model.test/v1", "LLM_MODEL": "m"},
        {"rubric": []},
        [{"query": "Оцени", "final": final}],
        context,
    )
    request = http.post.await_args.kwargs["json"]
    assert request["max_tokens"] == 12000
    assert request["reasoning_effort"] == "medium"
    assert result["verdict"] == "fail"


def sample():
    return {
        "status": "completed",
        "answer": "Дефицит школ 400 мест.",
        "missing": [],
        "evidence_ids": ["a"],
    }, {
        "completed": [
            {"request_id": "r", "step": 1, "scenario_id": 91001, "status": "completed"}
        ],
        "artifacts": [
            {
                "id": "a",
                "request_id": "r",
                "step": 1,
                "confirmed": True,
                "kind": "table",
                "content": {
                    "name": "provision_summary",
                    "rows": [
                        {
                            "service": "Школа",
                            "capacity": 900,
                            "demand": 1300,
                            "deficit": 400,
                        }
                    ],
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_judge_deduplicates_same_scope_evidence_without_skipping_evaluation():
    import json
    from unittest.mock import AsyncMock, Mock

    from tests.integration.industrial.judge import BASE_RUBRIC, evaluate

    final, context = sample()
    context["artifacts"][0]["content"]["rows"] *= 80
    template = context["artifacts"][0]
    context["artifacts"] = [
        {**deepcopy(template), "id": f"copy-{i}"} for i in range(20)
    ]
    rows = [
        {"id": name, "verdict": "fail", "reason": "Неподтверждённый вывод"}
        for name in BASE_RUBRIC
    ]
    http = Mock(
        post=AsyncMock(
            return_value=Mock(
                json=lambda: {
                    "choices": [
                        {"message": {"content": json.dumps({"criteria": rows})}}
                    ]
                }
            )
        )
    )
    result = await evaluate(
        http,
        {"LLM_BASE_URL": "http://model.test/v1", "LLM_MODEL": "m"},
        {"rubric": []},
        [{"query": "Оцени", "final": final}],
        context,
    )
    assert http.post.await_count == 1
    assert result["verdict"] == "fail"
    payload = json.loads(http.post.await_args.kwargs["json"]["messages"][1]["content"])
    assert len(payload["evidence"]) == 20
    assert payload["evidence"]["E2"]["identical_to"] == "E1"


def test_judge_never_deduplicates_evidence_across_scenarios():
    from tests.integration.industrial.judge import evidence_for_judge

    _, context = sample()
    context["artifacts"].append(
        {**deepcopy(context["artifacts"][0]), "id": "b", "request_id": "other"}
    )
    context["completed"].append(
        {"request_id": "other", "step": 1, "scenario_id": 91002}
    )
    evidence = evidence_for_judge(context)
    assert evidence["a"]["scenario_id"] == 91001
    assert evidence["b"]["scenario_id"] == 91002
    assert "content" in evidence["a"] and "content" in evidence["b"]


def test_judge_compaction_preserves_verdict_scope_and_source_without_mutating_artifact():
    from tests.integration.industrial.judge import evidence_for_judge

    _, context = sample()
    result = {
        "restriction_id": "r",
        "compliance_status": "violated",
        "verification_status": "complete",
        "coverage": {"checked_objects": 1},
        "summary": {"violated_objects": 1},
        "source": {"document_name": "TEST", "version": "2026", "clause_number": "1.1"},
        "violated_features": {
            "type": "FeatureCollection",
            "features": [{"geometry": {"coordinates": [[30, 60]] * 1000}}],
        },
        "evidence": [{"object_ref": {"id": "school-1"}, "threshold": 50, "unit": "m"}],
    }
    saved = deepcopy(result)
    context["artifacts"] = [
        {**context["artifacts"][0], "kind": "compliance_result", "content": result}
    ]
    proof = evidence_for_judge(context)["a"]
    assert proof["scenario_id"] == 91001
    assert proof["content"]["coverage"] == result["coverage"]
    assert proof["content"]["summary"] == result["summary"]
    assert proof["content"]["source"] == result["source"]
    assert proof["content"]["evidence"] == result["evidence"]
    assert "violated_features" not in proof["content"]
    assert result == saved


def test_numeric_acceptance_rejects_correct_number_from_wrong_version():
    final, context = sample()
    contract = {
        "provision": [
            {"scenario_id": 91001, "service": "Школа", "values": [900, 1300, 400]}
        ]
    }
    assert verify_result(final, context, contract)["passed"]
    stale = deepcopy(context)
    stale["completed"][0]["scenario_id"] = 91005
    assert not verify_result(final, stale, contract)["passed"]


def test_honest_blocker_is_not_positive_acceptance():
    final, context = sample()
    final.update(status="blocked", missing=[{"question": "Добавьте норматив"}])
    assert not verify_result(final, context, {})["passed"]


def test_judge_cannot_pass_with_invented_quote_or_uncertain_criterion():
    answers = ["Дефицит школ 400 мест."]
    evidence = {"a": {"deficit": 400}}
    review = {
        "criteria": [
            {
                "id": "grounding",
                "verdict": "pass",
                "answer_quote": answers[0],
                "evidence_ids": ["a"],
                "reason": "400 совпадает с результатом расчёта.",
            }
        ]
    }
    assert (
        validate_judgment(review, ["grounding"], answers, evidence)["verdict"] == "pass"
    )
    review["criteria"][0]["answer_quote"] = "Дефицита нет."
    assert (
        validate_judgment(review, ["grounding"], answers, evidence)["verdict"]
        == "needs_review"
    )
    review["criteria"][0]["answer_quote"] = answers[0]
    review["criteria"][0]["verdict"] = "needs_review"
    assert (
        validate_judgment(review, ["grounding"], answers, evidence)["verdict"]
        == "needs_review"
    )


def test_judge_reference_decoding_does_not_accept_fabricated_proof():
    from tests.integration.industrial.judge import decode_references

    quotes, aliases = {"Q1": "Дефицит 400."}, {"E1": "actual-source"}
    review = {
        "criteria": [
            {
                "id": "grounding",
                "verdict": "pass",
                "quote_id": "Q1",
                "evidence_ids": ["E1"],
                "reason": "400 соответствует таблице",
            }
        ]
    }
    decoded = decode_references(review, quotes, aliases)
    assert (
        validate_judgment(
            decoded, ["grounding"], list(quotes.values()), {"actual-source": {}}
        )["verdict"]
        == "pass"
    )
    review["criteria"][0]["evidence_ids"] = ["E99"]
    decoded = decode_references(review, quotes, aliases)
    assert (
        validate_judgment(
            decoded, ["grounding"], list(quotes.values()), {"actual-source": {}}
        )["verdict"]
        == "needs_review"
    )


def test_fifteen_independent_episodes_required_on_identical_build():
    report = {
        "preflight_passed": True,
        "fingerprint": {"build": "one"},
        "final_fingerprint": {"build": "one"},
        "episodes": [
            {
                "id": e["id"],
                "formulation": e["formulation"],
                "passed": True,
                "turns": [
                    {"request_id": f"{e['id']}-{e['formulation']}-{i}"}
                    for i in range(2)
                ],
            }
            for e in episodes()
        ],
    }
    assert series_verdict(report)
    broken = deepcopy(report)
    broken["episodes"].pop()
    assert not series_verdict(broken)
    broken = deepcopy(report)
    broken["episodes"][0]["passed"] = False
    assert not series_verdict(broken)
    broken = deepcopy(report)
    broken["final_fingerprint"]["build"] = "two"
    assert not series_verdict(broken)


def test_correct_geometry_with_stale_source_version_is_rejected():
    from tests.integration.industrial.control import entities

    final, context = sample()
    layer = entities(91001, "physical_object", 8)
    context["artifacts"].append(
        {
            "id": "park",
            "request_id": "r",
            "step": 1,
            "kind": "feature_collection",
            "confirmed": True,
            "content": {"name": "Парк", "feature_collection": layer},
        }
    )
    context["artifacts"].append(
        {
            "id": "park-table",
            "request_id": "r",
            "step": 1,
            "kind": "table",
            "confirmed": True,
            "content": {"rows": [deepcopy(f["properties"]) for f in layer["features"]]},
        }
    )
    contract = {
        "source_layers": [
            {"scenario_id": 91001, "domain": "physical_object", "type_id": 8}
        ]
    }
    assert verify_result(final, context, contract)["passed"]
    layer["features"][0]["properties"]["source_version"] = "stale"
    assert not verify_result(final, context, contract)["passed"]


def test_matching_violation_count_from_another_norm_cannot_pass():
    from tests.integration.industrial.control import entities
    from tests.unit.test_harness_source_oracle import records

    final, context = sample()
    sources = records()
    wrong = deepcopy(sources["norms"][0])
    wrong.update(id="wrong", value={"number": 100, "operator": ">=", "unit": "м"})
    sources["norms"].append(wrong)
    layer = entities(91001, "physical_object", 7)
    layer["features"][0]["properties"].update(
        restriction_id="restriction", compliance_status="violated"
    )
    for system, rows in sources.items():
        context["artifacts"].append(
            {
                "id": system,
                "request_id": "r",
                "step": 1,
                "confirmed": True,
                "kind": "source_evidence",
                "content": {"system": system, "sources": rows},
            }
        )
    context["artifacts"].append(
        {
            "id": "check",
            "request_id": "r",
            "step": 1,
            "confirmed": True,
            "kind": "compliance_result",
            "content": {
                "restriction_id": "restriction",
                "verification_status": "complete",
                "coverage": {
                    "applicable_objects": 1,
                    "checked_objects": 1,
                    "unchecked_objects": 0,
                },
                "summary": {"violated_objects": 1, "passed_objects": 0},
                "compliance_status": "violated",
                "violated_features": layer,
                "passed_features": {"type": "FeatureCollection", "features": []},
            },
        }
    )
    contract = {
        "sources": True,
        "compliance": [{"scenario_id": 91001, "violations": 1}],
    }
    assert verify_result(final, context, contract)["passed"]
    context["artifacts"][-1]["content"]["restriction_id"] = "wrong"
    layer["features"][0]["properties"]["restriction_id"] = "wrong"
    assert not verify_result(final, context, contract)["passed"]
