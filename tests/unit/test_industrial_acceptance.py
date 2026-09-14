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
