from copy import deepcopy

import pytest

from tests.integration.industrial.progress import scenario_progress
from tests.integration.industrial.scenarios import CASES


def report_for(count=3):
    return {
        "fingerprint": {"commit": "fixed-build"},
        "preflight_passed": True,
        "passed": False,
        "episodes": [
            {
                "id": case["id"],
                "formulation": formulation,
                "passed": True,
                "judge_verdict": "pass",
                "turns": [
                    {
                        "request_id": f"{case['id']}-{formulation}-{turn}",
                        "passed": True,
                        "checks": {"passed": True},
                        "replay_passed": True,
                        "final": {"status": "completed"},
                    }
                    for turn in (1, 2)
                ],
            }
            for case in CASES[:count]
            for formulation in (1, 2, 3)
        ],
    }


def test_three_complete_scenarios_are_a_milestone_not_full_acceptance():
    report = report_for()
    result = scenario_progress(report, report["fingerprint"], [CASES[0]["id"]])
    assert result["half_scenarios_passed"]
    assert len(result["new_scenarios"]) == 2
    assert not result["full_acceptance_passed"]


@pytest.mark.parametrize(
    "failure", ["judge", "turn", "replay", "checks", "status", "duplicate", "missing"]
)
def test_three_formulations_must_all_meet_independent_acceptance(failure):
    report = report_for(1)
    row = report["episodes"][-1]
    if failure == "judge":
        row["judge_verdict"] = "needs_review"
    elif failure == "turn":
        row["turns"][1]["passed"] = False
    elif failure == "replay":
        row["turns"][1]["replay_passed"] = False
    elif failure == "checks":
        row["turns"][1]["checks"]["passed"] = False
    elif failure == "status":
        row["turns"][1]["final"]["status"] = "blocked"
    elif failure == "duplicate":
        row["turns"][1]["request_id"] = row["turns"][0]["request_id"]
    else:
        report["episodes"].pop()
    assert not scenario_progress(report, report["fingerprint"])["successful_scenarios"]


def test_new_success_cannot_hide_a_regression_in_previous_scenarios():
    report = report_for(4)
    report["episodes"][0]["passed"] = False
    result = scenario_progress(report, report["fingerprint"], [CASES[0]["id"]])
    assert len(result["successful_scenarios"]) == 3
    assert not result["previous_preserved"]
    assert not result["new_scenarios"]
    assert not result["half_scenarios_passed"]


@pytest.mark.parametrize("change", ["current", "final", "preflight"])
def test_changed_build_or_failed_preflight_cannot_form_a_milestone(change):
    report = report_for()
    current = deepcopy(report["fingerprint"])
    if change == "current":
        current["commit"] = "different"
    elif change == "final":
        report["final_fingerprint"] = {"commit": "different"}
    else:
        report["preflight_passed"] = False
    assert not scenario_progress(report, current)["successful_scenarios"]
