#!/usr/bin/env python3
"""Watch a running experiment for faults that are worth stopping it over.

A long run fails in two ways. It dies — visible immediately — or it keeps going
while producing rows that mean nothing, which is invisible until the report is
built hours later. Both of the defects found in the main run were the second
kind: gpt-oss returned empty completions for every row, and the recorded plan
blanked the entity the model had named. Each would have been caught in minutes
by looking at what the rows contained rather than at how many there were.

The check is deliberately about *shape*, not about score. A model that answers
badly is a result; a model that answers nothing, or a taxonomy that files every
failure under `other`, is a fault. So a low success rate is never reported and a
uniform end state always is.

Exit code 1 when anything fired, so a monitor can key off it.

    python benchmarks/harness/health_check.py --results benchmarks/data/results_inproc
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# A row that reached neither a plan nor an error did not exercise anything.
MIN_ROWS_BEFORE_JUDGING = 10
# `other` is by definition a hole in the taxonomy; a few are a to-do, a third of
# the arm means failures are not being classified at all.
OTHER_SHARE_ALERT = 0.20
# Every row landing in one state means the arm is measuring one code path.
UNIFORM_STATE_ALERT = 0.98
# The plan is the object of study; if it is missing everywhere, the capture broke.
NO_PLAN_SHARE_ALERT = 0.90
# Timeouts are a threshold artefact, not a model property — see the README.
TIMEOUT_SHARE_ALERT = 0.15


def load_arms(root: Path) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for path in sorted(root.glob("*/*/results.jsonl")):
        rows = []
        for line in path.open(encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if rows:
            arms[f"{path.parts[-3]}/{path.parts[-2]}"] = rows
    return arms


def check_arm(label: str, rows: list[dict]) -> list[str]:
    """Faults in one arm, as human-readable lines."""

    alerts: list[str] = []
    n = len(rows)
    if n < MIN_ROWS_BEFORE_JUDGING:
        return alerts

    ablation = "no_catalog" in label
    states = Counter(r.get("end_state") for r in rows)
    top_state, top_count = states.most_common(1)[0]
    # In the ablation arm a single end state is the *result* — stripping the
    # catalog is expected to collapse every row into clarification — so
    # uniformity there is not evidence of a fault.
    if (
        not ablation
        and top_count / n >= UNIFORM_STATE_ALERT
        and top_state != "full_success"
    ):
        alerts.append(
            f"{label}: {top_count}/{n} rows are all `{top_state}` — one code path, "
            f"likely a configuration fault rather than model behaviour"
        )

    classes = Counter(r.get("error_class") for r in rows if r.get("error_class"))
    other = classes.get("other", 0)
    if other / n >= OTHER_SHARE_ALERT:
        alerts.append(
            f"{label}: {other}/{n} failures classified `other` — the taxonomy is "
            f"not covering what is actually failing"
        )

    timeouts = states.get("timeout", 0)
    if timeouts / n >= TIMEOUT_SHARE_ALERT:
        alerts.append(
            f"{label}: {timeouts}/{n} rows timed out — that is the harness "
            f"threshold showing up as a result; raise --timeout and redo them"
        )

    # The plan is what the experiment is about. Rows that failed before planning
    # legitimately have none, so this only fires when almost nothing was captured.
    planned = sum(1 for r in rows if r.get("model_plan") or r.get("restriction_plan"))
    if (n - planned) / n >= NO_PLAN_SHARE_ALERT:
        alerts.append(
            f"{label}: {n - planned}/{n} rows recorded no plan at all — either the "
            f"model returns nothing or the plan capture is broken"
        )

    # A model that emits plans but never names an entity is the signature of the
    # ablation arm; in a `base` arm it means the catalog never reached the prompt.
    if "no_catalog" not in label:
        with_entities = sum(
            1
            for r in rows
            if (r.get("model_plan") or {}).get("source_entities")
            or (r.get("model_plan") or {}).get("target_entities")
        )
        if planned and with_entities == 0:
            alerts.append(
                f"{label}: {planned} plans recorded and not one names an entity — "
                f"the domain catalog is probably not reaching the prompt"
            )

    return alerts


def check_dataset(rows_by_arm: dict[str, list[dict]]) -> list[str]:
    """Faults visible across arms — i.e. in the dataset rather than the model."""

    alerts: list[str] = []
    # A row every model fails identically is more likely a bad row than a
    # coincidence; a scenario every model fails on is a data problem.
    per_scenario_fail: dict[int, list[bool]] = {}
    for rows in rows_by_arm.values():
        for row in rows:
            if "no_catalog" in str(row.get("arm", "")):
                continue
            scenario = row.get("scenario_id")
            if scenario is None:
                continue
            per_scenario_fail.setdefault(int(scenario), []).append(
                row.get("end_state") == "data_unavailable"
            )
    for scenario, flags in sorted(per_scenario_fail.items()):
        if len(flags) >= MIN_ROWS_BEFORE_JUDGING and all(flags):
            alerts.append(
                f"scenario {scenario}: every one of {len(flags)} rows is "
                f"`data_unavailable` — the offline store is missing this scenario"
            )
    return alerts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="benchmarks/data/results_inproc")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only when something fired (for a monitor loop)",
    )
    args = parser.parse_args()

    arms = load_arms(Path(args.results))
    if not arms:
        if not args.quiet:
            print("no results yet")
        return 0

    alerts: list[str] = []
    for label, rows in arms.items():
        alerts.extend(check_arm(label, rows))
    alerts.extend(check_dataset(arms))

    if alerts:
        print("HEALTH ALERT")
        for alert in alerts:
            print(f"  - {alert}")
        return 1
    if not args.quiet:
        for label, rows in arms.items():
            states = Counter(r.get("end_state") for r in rows)
            print(f"  {label}: {len(rows)} rows {dict(states)}")
        print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
