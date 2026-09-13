from copy import deepcopy

from tests.integration.industrial.acceptance import verify_result
from tests.integration.industrial.judge import validate_judgment
from tests.integration.industrial.runner import series_verdict
from tests.integration.industrial.scenarios import episodes


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
    contract = {
        "source_layers": [
            {"scenario_id": 91001, "domain": "physical_object", "type_id": 8}
        ]
    }
    assert verify_result(final, context, contract)["passed"]
    layer["features"][0]["properties"]["source_version"] = "stale"
    assert not verify_result(final, context, contract)["passed"]
