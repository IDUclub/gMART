#!/usr/bin/env python3
"""Aggregate the validated matrix with scenario-clustered uncertainty.

Rows from the same scenario and repeated generations are not independent.  The
bootstrap therefore resamples scenario clusters and keeps every prompt/repeat in
the sampled cluster together.  Paired deltas use only keys present on both sides.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks" / "eval"))
sys.path.insert(0, str(ROOT / "benchmarks" / "harness"))

import inproc_eval as ev  # noqa: E402
import inproc_runner as runner  # noqa: E402
from gold_parser import load_gold, norm  # noqa: E402

BOOTSTRAPS = 10_000
SEED = 20260831


def latest_rows(path: Path) -> list[dict]:
    latest: dict[int, dict] = {}
    if not path.exists():
        return []
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[int(row["idx"])] = row
    return [latest[idx] for idx in sorted(latest)]


def result_path(root: Path, cell: dict) -> Path:
    return (
        root
        / cell["slug"]
        / runner._safe_name(cell["model"])
        / f"{cell['arm']}--{runner.LOCAL}"
        / "results.jsonl"
    )


def gold_index(path: str) -> dict[tuple[int, str], object]:
    return {(item.scenario_id, norm(item.question)): item for item in load_gold(path)}


def actual_success(row: dict, gold: object | None) -> bool:
    state = ev.resolved_end_state(row, gold)
    if gold is not None and getattr(gold, "intent", None) == "needs_clarification":
        return state == runner.STATE_CLARIFICATION
    return state == runner.STATE_FULL_SUCCESS


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def cluster_ci(
    values: list[tuple[int, float]], seed_offset: int = 0
) -> tuple[float, float]:
    clusters: dict[int, list[float]] = defaultdict(list)
    for scenario, value in values:
        clusters[scenario].append(value)
    labels = sorted(clusters)
    if not labels:
        return float("nan"), float("nan")
    rng = random.Random(SEED + seed_offset)
    draws: list[float] = []
    for _ in range(BOOTSTRAPS):
        sampled = [rng.choice(labels) for _ in labels]
        observations = [value for label in sampled for value in clusters[label]]
        draws.append(statistics.fmean(observations))
    return percentile(draws, 0.025), percentile(draws, 0.975)


def paired_delta(
    left: list[dict], right: list[dict], gold: dict[tuple[int, str], object]
) -> dict | None:
    def keyed(rows: list[dict]) -> dict[tuple[int, str, str], dict]:
        return {
            (
                int(row["scenario_id"]),
                norm(row["prompt"]),
                str(row.get("repeat_id", "1")),
            ): row
            for row in rows
        }

    a, b = keyed(left), keyed(right)
    common = sorted(set(a) & set(b))
    if not common:
        return None
    values: list[tuple[int, float]] = []
    wins = losses = ties = 0
    for scenario, prompt, repeat in common:
        task = gold.get((scenario, prompt))
        delta = float(actual_success(a[(scenario, prompt, repeat)], task)) - float(
            actual_success(b[(scenario, prompt, repeat)], task)
        )
        values.append((scenario, delta))
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    low, high = cluster_ci(values, seed_offset=len(common))
    return {
        "n": len(common),
        "delta": statistics.fmean(value for _, value in values),
        "low": low,
        "high": high,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def table(headers: list[str], rows: list[list[object]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    plan = json.loads((root / "matrix_plan.json").read_text(encoding="utf-8"))
    expert_gold = gold_index("benchmarks/data/gold/exp_data_restrictions.csv")

    loaded: dict[str, list[dict]] = {}
    cells: dict[str, dict] = {}
    evaluations: dict[str, dict] = {}
    for cell in plan["cells"]:
        rows = latest_rows(result_path(root, cell))
        if not rows:
            continue
        loaded[cell["slug"]] = rows
        cells[cell["slug"]] = cell
        if cell["dataset"].endswith("exp_data_restrictions.csv"):
            arm = ev.Arm(cell["model"], cell["arm"], runner.LOCAL, rows)
            evaluations[cell["slug"]] = ev.evaluate(arm, expert_gold)

    metric_rows: list[list[object]] = []
    for slug, evaluation in sorted(evaluations.items()):
        cell = cells[slug]
        if cell["phase"] == "smoke":
            continue
        observations = []
        for row in loaded[slug]:
            task = expert_gold.get((int(row["scenario_id"]), norm(row["prompt"])))
            observations.append(
                (int(row["scenario_id"]), float(actual_success(row, task)))
            )
        low, high = cluster_ci(observations)
        repeat_rates: dict[str, list[float]] = defaultdict(list)
        for row in loaded[slug]:
            task = expert_gold.get((int(row["scenario_id"]), norm(row["prompt"])))
            repeat_rates[str(row.get("repeat_id", "1"))].append(
                float(actual_success(row, task))
            )
        rate = statistics.fmean(value for _, value in observations)
        plan_metric = evaluation["plan"]
        metric_rows.append(
            [
                cell["model"],
                cell["arm"],
                cell["schema_arm"],
                cell["repeat"],
                len(observations),
                f"{fmt_pct(rate)} [{fmt_pct(low)}, {fmt_pct(high)}]",
                plan_metric["intent"].cell(),
                plan_metric["buffered_entity"].cell(),
                plan_metric["counted_entity"].cell(),
                plan_metric["distance"].cell(),
                evaluation["infra_failures"],
            ]
        )

    pair_rows: list[list[object]] = []

    def grouped(predicate) -> dict[tuple, list[dict]]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for slug, rows in loaded.items():
            cell = cells[slug]
            if predicate(cell):
                groups[(cell["model"], cell["repeat"])].extend(rows)
        return groups

    required_base = grouped(
        lambda c: c["dataset"].endswith("exp_data_restrictions.csv")
        and c["schema_arm"] == runner.SCHEMA_REQUIRED
        and c["arm"] == runner.ARM_BASE
        and c["phase"] != "smoke"
    )
    required_no_catalog = grouped(
        lambda c: c["dataset"].endswith("exp_data_restrictions.csv")
        and c["schema_arm"] == runner.SCHEMA_REQUIRED
        and c["arm"] == runner.ARM_NO_CATALOG
        and c["phase"] != "smoke"
    )
    optional_base = grouped(
        lambda c: c["dataset"].endswith("exp_data_restrictions.csv")
        and c["schema_arm"] == runner.SCHEMA_OPTIONAL
        and c["arm"] == runner.ARM_BASE
        and c["phase"] != "smoke"
    )

    def add_pairs(label: str, left_groups: dict, right_groups: dict) -> None:
        for key in sorted(set(left_groups) & set(right_groups)):
            result = paired_delta(left_groups[key], right_groups[key], expert_gold)
            if result is None:
                continue
            model, repeat = key
            pair_rows.append(
                [
                    label,
                    model,
                    repeat,
                    result["n"],
                    f"{fmt_pct(result['delta'])} [{fmt_pct(result['low'])}, {fmt_pct(result['high'])}]",
                    f"{result['wins']}/{result['losses']}/{result['ties']}",
                ]
            )

    add_pairs("catalog: base − no_catalog", required_base, required_no_catalog)
    add_pairs("schema: required − optional", required_base, optional_base)

    # Modern-model comparison, paired by prompt and repeat under the production schema.
    by_model_repeat = required_base
    for repeat in sorted({key[1] for key in by_model_repeat}):
        left = by_model_repeat.get(("gemma4:12b", repeat))
        right = by_model_repeat.get(("gpt-oss:20b", repeat))
        if left and right:
            result = paired_delta(left, right, expert_gold)
            if result:
                pair_rows.append(
                    [
                        "model: Gemma 4 − GPT-OSS",
                        "paired",
                        repeat,
                        result["n"],
                        f"{fmt_pct(result['delta'])} [{fmt_pct(result['low'])}, {fmt_pct(result['high'])}]",
                        f"{result['wins']}/{result['losses']}/{result['ties']}",
                    ]
                )

    synthetic_rows: list[list[object]] = []
    for slug, rows in sorted(loaded.items()):
        cell = cells[slug]
        if not cell["dataset"].endswith("expanded_catalog.csv"):
            continue
        observations = [
            (
                int(row["scenario_id"]),
                float(row.get("end_state") == runner.STATE_FULL_SUCCESS),
            )
            for row in rows
        ]
        low, high = cluster_ci(observations)
        rate = statistics.fmean(value for _, value in observations)
        synthetic_rows.append(
            [
                cell["model"],
                len(rows),
                f"{fmt_pct(rate)} [{fmt_pct(low)}, {fmt_pct(high)}]",
            ]
        )

    complete = sum(
        len(loaded.get(cell["slug"], [])) >= int(cell["expected_rows"])
        for cell in plan["cells"]
        if cell["phase"] != "smoke"
    )
    planned = sum(cell["phase"] != "smoke" for cell in plan["cells"])
    lines = [
        "# Validated restriction experiments\n",
        f"Run `{plan['run_id']}`. Complete full cells: {complete}/{planned}. ",
        "Intervals are 95% percentile bootstrap intervals over scenario clusters ",
        f"({BOOTSTRAPS:,} resamples); repeated generations remain inside their scenario cluster.\n",
        "## Expert-authored primary set\n",
        table(
            [
                "Model",
                "Catalog",
                "Schema",
                "Repeat",
                "N",
                "Actual success (95% CI)",
                "Intent",
                "Buffered entity",
                "Counted entity",
                "Distance",
                "Infra failures",
            ],
            metric_rows,
        ),
        "\n\n## Paired effects\n",
        table(
            [
                "Contrast",
                "Model",
                "Repeat",
                "Pairs",
                "Δ success (95% CI)",
                "wins/losses/ties",
            ],
            pair_rows,
        ),
        "\n\n## Synthetic robustness slice (secondary evidence)\n",
        "The slice is deterministic and cluster-balanced. It is not independent expert gold: the prompts were model-generated, so it can support robustness but not replace the primary set.\n",
        table(["Model", "N", "Operational success (95% CI)"], synthetic_rows),
        "\n\n## Interpretation rules\n",
        "- A paired effect is treated as supported only when its cluster-bootstrap interval excludes zero.\n"
        "- Model ranking is not claimed from overlapping or zero-crossing paired intervals.\n"
        "- Geometry/object-selection conclusions require reference GeoJSON and are reported separately; absence of those files means those claims are not computable.\n"
        "- Incomplete cells, any infrastructure failures, or a changed manifest invalidate cross-cell comparison until rerun.\n",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
