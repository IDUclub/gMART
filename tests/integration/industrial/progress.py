"""Scenario milestones complement, but never replace, full-series acceptance."""

from .scenarios import CASES


def scenario_progress(report, current_fingerprint, previous=()):
    """Require three complete independent dialogues and recheck prior milestones.

    While a series is running, the caller obtains current_fingerprint from its
    immutable checkout. At completion, the recorded final fingerprint also has
    to agree. A successful dialogue from another build is never carried forward.
    """
    known = {case["id"] for case in CASES}
    previous = set(previous)
    if not previous <= known:
        raise ValueError("Unknown previously accepted scenario")
    unchanged = bool(report.get("fingerprint")) and (
        report["fingerprint"] == current_fingerprint
        and report.get("final_fingerprint", current_fingerprint) == current_fingerprint
    )
    rows = report.get("episodes", [])
    request_ids = [
        turn.get("request_id") for row in rows for turn in row.get("turns", [])
    ]
    independent = None not in request_ids and len(request_ids) == len(set(request_ids))
    successful = []
    for scenario in sorted(known):
        attempts = [row for row in rows if row.get("id") == scenario]
        if (
            report.get("preflight_passed") is True
            and unchanged
            and independent
            and len(attempts) == 3
            and {row.get("formulation") for row in attempts} == {1, 2, 3}
            and all(
                row.get("passed") is True
                and row.get("judge_verdict") == "pass"
                and len(row.get("turns", [])) == 2
                and all(
                    turn.get("passed") is True
                    and turn.get("checks", {}).get("passed") is True
                    and turn.get("replay_passed") is True
                    and turn.get("final", {}).get("status") == "completed"
                    for turn in row["turns"]
                )
                for row in attempts
            )
        ):
            successful.append(scenario)
    preserved = previous <= set(successful)
    return {
        "successful_scenarios": successful,
        "previous_preserved": preserved,
        "new_scenarios": sorted(set(successful) - previous) if preserved else [],
        "fingerprint_unchanged": unchanged,
        "half_scenarios_passed": preserved and len(successful) >= 3,
        "full_acceptance_passed": report.get("passed") is True,
    }
