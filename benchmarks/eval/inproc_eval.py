#!/usr/bin/env python3
"""Score the in-process runs against the expert gold set.

Reads the records `inproc_runner.py` writes and produces the tables the reviewer
asked for. Three things separate this from `semantic_eval.py`, which scores the
older HTTP runs:

* **The plan is there.** Intent, source and target entity and the buffer
  parameter are scored directly against the `RestrictionPlan` the pipeline
  executed. On the HTTP runs they were not recoverable at all — the pipeline
  emits no plan event, so nothing in the stream carried it.
* **The failure taxonomy is read, not re-derived.** Each record carries the class
  decided where the exception was raised, so `semantic_eval.classify_error`'s
  substring matching over error messages is gone, and with it the risk of filing
  a model failure under infrastructure or the reverse.
* **A data gap is not a model failure.** Rows whose end state is
  `data_unavailable` — the offline store had no answer — are excluded from every
  model-facing denominator and reported separately as data coverage.

Success is task-aware per the reviewer's second point: a `restrictions` task
needs both output layers, a `buffers_only` task has no `objects` layer by design
and is not penalised for lacking one, and a `needs_clarification` task succeeds
by asking. Results are broken out by task type rather than pooled into one proxy.

Usage::

    python benchmarks/eval/inproc_eval.py \\
        --results benchmarks/data/results_inproc \\
        --gold benchmarks/data/gold/exp_data.csv \\
        --out benchmarks/out/inproc_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "harness"))

import inproc_runner as runner  # noqa: E402  (the taxonomy lives with the runner)
from gold_parser import GoldRecord, load_gold, norm  # noqa: E402

END_STATES = [
    runner.STATE_FULL_SUCCESS,
    runner.STATE_PARTIAL_SPATIAL,
    runner.STATE_CLARIFICATION,
    runner.STATE_PLANNING_FAILURE,
    runner.STATE_TOOL_INFRA_FAILURE,
    runner.STATE_TIMEOUT,
    runner.STATE_EMPTY,
]
FAILURE_CLASSES = [
    runner.CLS_INVALID_PLAN,
    runner.CLS_UNEXECUTABLE_PLAN,
    runner.CLS_LLM_BACKEND,
    runner.CLS_TOOL_EXECUTION,
    runner.CLS_URBAN_API,
    runner.CLS_TRANSPORT,
    runner.CLS_TIMEOUT,
    runner.CLS_TOKEN_EXPIRED,
    runner.CLS_OTHER,
]
# Which side of the model/infrastructure line each class falls on. The reviewer
# asks for exactly this split; it is a property of the class, not a guess.
MODEL_CLASSES = {
    runner.CLS_INVALID_PLAN,
    runner.CLS_UNEXECUTABLE_PLAN,
    runner.CLS_LLM_BACKEND,
}
INFRA_CLASSES = {
    runner.CLS_TOOL_EXECUTION,
    runner.CLS_URBAN_API,
    runner.CLS_TRANSPORT,
    runner.CLS_TIMEOUT,
    runner.CLS_TOKEN_EXPIRED,
}

TASK_TYPES = ["buffers_only", "restrictions", "needs_clarification"]

DISTANCE_TOLERANCE_M = 0.5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """One (model, ablation arm, transport) run."""

    model: str
    arm: str
    transport: str
    records: list[dict] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.model} / {self.arm} / {self.transport}"


def load_arms(results_dir: Path) -> list[Arm]:
    """Every ``<model>/<arm>--<transport>/results.jsonl`` under the results dir."""

    arms: list[Arm] = []
    for path in sorted(results_dir.glob("*/*/results.jsonl")):
        arm_dir = path.parent.name
        arm_name, _, transport = arm_dir.partition("--")
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # A run killed mid-write leaves one torn last line; losing it
                    # is right, failing the whole report over it is not.
                    continue
        if not records:
            continue
        arms.append(
            Arm(
                model=records[0].get("model") or path.parent.parent.name,
                arm=arm_name or runner.ARM_BASE,
                transport=transport or runner.LOCAL,
                records=records,
            )
        )
    return arms


def join_gold(
    records: list[dict], gold_index: dict[tuple[int, str], GoldRecord]
) -> list[tuple[dict, GoldRecord]]:
    """Pair each record with its gold row, one pairing per gold row."""

    seen: set[tuple[int, str]] = set()
    paired: list[tuple[dict, GoldRecord]] = []
    for record in records:
        key = (int(record["scenario_id"]), norm(record["prompt"]))
        gold = gold_index.get(key)
        if gold is None or key in seen:
            continue
        seen.add(key)
        paired.append((record, gold))
    return paired


# --------------------------------------------------------------------------- #
# Plan scoring
# --------------------------------------------------------------------------- #
def plan_entity_names(plan: dict, key: str) -> set[str]:
    return {
        norm(entity.get("name"))
        for entity in (plan.get(key) or [])
        if isinstance(entity, dict) and entity.get("name")
    }


def plan_distances(plan: dict) -> set[float]:
    distances: set[float] = set()
    for rule in plan.get("buffer_rules") or []:
        if isinstance(rule, dict) and rule.get("buffer_size") is not None:
            try:
                distances.add(float(rule["buffer_size"]))
            except (TypeError, ValueError):
                continue
    return distances


@dataclass
class Tally:
    """Correct out of comparable, so partial coverage is never hidden."""

    correct: int = 0
    comparable: int = 0

    def add(self, is_correct: bool) -> None:
        self.comparable += 1
        self.correct += bool(is_correct)

    @property
    def pct(self) -> float | None:
        return 100.0 * self.correct / self.comparable if self.comparable else None

    def cell(self) -> str:
        if not self.comparable:
            return "n/a"
        return f"{self.pct:.1f}% ({self.correct}/{self.comparable})"


def model_plan_of(record: dict) -> dict | None:
    """The plan the *model* produced, which is what these metrics are about.

    ``restriction_plan`` is the plan after the pipeline canonicalised it, and
    that step is lossy: a plan whose ``restriction_rules`` omit ``target_names``
    loses every rule, which flips the mode to ``needs_clarification``, which
    blanks ``target_entities``. Scoring entity accuracy there reports 0 % for a
    model that named the entity correctly — measured on gemma3:12b, which does
    exactly this on every row. Runs made before ``model_plan`` was recorded fall
    back to the old field so their reports still build.
    """

    return record.get("model_plan") or record.get("restriction_plan")


def score_plan(
    paired: list[tuple[dict, GoldRecord]],
) -> dict[str, Tally]:
    """Intent, source/target entity and buffer parameter accuracy.

    A field is scored only on gold rows whose confidence flag for it is set: the
    gold ground truth is extracted from expert prose, and scoring a model against
    a value the parser itself is unsure of produces a number nobody can defend.
    The comparable count is reported next to every percentage.
    """

    tallies = {
        "intent": Tally(),
        "buffered_entity": Tally(),
        "counted_entity": Tally(),
        "distance": Tally(),
    }
    for record, gold in paired:
        plan = model_plan_of(record)
        if not plan:
            # No plan means the row failed before or during planning. That is a
            # planning failure in the end states; counting it as a wrong entity
            # here would charge the same failure twice.
            continue
        tallies["intent"].add(plan.get("mode") == gold.intent)

        # The two sides name the roles in opposite directions, and the tally keys
        # here are the *meaning*, not either side's word for it:
        #
        #   gold.target_entity  == the BUFFERED entity == plan.source_entities
        #   gold.source_entity  == the COUNTED entity  == plan.target_entities
        #
        # gold_parser says so in as many words ("this is inverted with respect to
        # the RestrictionPlan schema, where buffers are built around
        # source_entities"). Comparing the two `source` fields to each other
        # looks obviously right and is wrong; it would report near-zero entity
        # accuracy and charge it to the models.
        if gold.conf.get("target") and gold.target_entity:
            tallies["buffered_entity"].add(
                norm(gold.target_entity) in plan_entity_names(plan, "source_entities")
            )
        if gold.conf.get("source") and gold.source_entity:
            tallies["counted_entity"].add(
                norm(gold.source_entity) in plan_entity_names(plan, "target_entities")
            )
        if gold.conf.get("distance") and gold.distance_m is not None:
            tallies["distance"].add(
                any(
                    abs(distance - gold.distance_m) <= DISTANCE_TOLERANCE_M
                    for distance in plan_distances(plan)
                )
            )
    return tallies


# --------------------------------------------------------------------------- #
# Arm-level metrics
# --------------------------------------------------------------------------- #
def scored_rows(records: list[dict]) -> list[dict]:
    """Rows that say something about the model — data gaps excluded."""

    return [
        record
        for record in records
        if record.get("end_state") != runner.STATE_DATA_UNAVAILABLE
    ]


def schema_loss(records: list[dict]) -> dict[str, int]:
    """Rows where the model planned a restriction and the pipeline refused it.

    This is the price of the typed schema, and it is a result rather than a bug
    to be tidied away: the model produced a plan naming both entities and the
    radius, and the pipeline discarded it because one nested, redundant field was
    missing. ``missing_target_names`` is the case observed so far — the counted
    entity is present in ``target_entities`` and absent from
    ``restriction_rules[].target_names``, which are required to agree.
    """

    loss = {"downgraded": 0, "missing_target_names": 0}
    for record in records:
        model = record.get("model_plan") or {}
        pipeline = record.get("restriction_plan") or {}
        if not model or not pipeline:
            continue
        if model.get("mode") != "restrictions":
            continue
        if pipeline.get("mode") == "restrictions":
            continue
        loss["downgraded"] += 1
        rules = model.get("restriction_rules") or []
        if rules and not any(rule.get("target_names") for rule in rules):
            loss["missing_target_names"] += 1
    return loss


# The layer pair a restrictions task must produce, in either naming. Kept here
# rather than imported from the runner because this is the *gold* criterion:
# what the task asked for, not what the model decided to attempt.
RESTRICTION_LAYER_SETS = (
    {"objects", "generators"},
    {"Объекты в зоне ограничений", "Источники ограничений"},
)


def produced_restriction_layers(record: dict) -> bool:
    produced = set(record.get("layer_counts") or {})
    return any(required <= produced for required in RESTRICTION_LAYER_SETS)


def resolved_end_state(record: dict, gold: GoldRecord | None) -> str:
    """The runner's end state, re-judged against what the gold task asked for.

    ``inproc_runner.end_state`` decides completeness from the mode the *model*
    declared, because at run time that is the only intent available. That lets a
    model grade its own exam: declaring ``buffers_only`` on a task that asks for
    a count reduces the criterion to "some layer exists", which any run that
    builds a buffer satisfies. It is not a hypothetical — llama3.1:8b declared
    ``buffers_only`` on 1153 of 1705 restrictions tasks, which the runner scored
    as 1122 successes against 1 by the criterion the tasks actually set.

    Here the gold intent is known, so the criterion comes from the task. Failures,
    clarifications and timeouts are left exactly as the runner classified them —
    only the success/partial judgement is re-decided, and only downwards.
    """

    state = record.get("end_state")
    if gold is None or state != runner.STATE_FULL_SUCCESS:
        return state
    if gold.intent != "restrictions":
        return state
    if produced_restriction_layers(record):
        return state
    return runner.STATE_PARTIAL_SPATIAL


def evaluate(arm: Arm, gold_index: dict[tuple[int, str], GoldRecord]) -> dict:
    paired_all = join_gold(arm.records, gold_index)
    paired = [
        (record, gold)
        for record, gold in paired_all
        if record.get("end_state") != runner.STATE_DATA_UNAVAILABLE
    ]
    scored = scored_rows(arm.records)
    total = len(arm.records)
    gaps = total - len(scored)

    # Every table below reads the state through the gold task, never the mode the
    # model chose for itself — see `resolved_end_state`.
    gold_of = {id(record): gold for record, gold in paired_all}
    state_of = {
        id(record): resolved_end_state(record, gold_of.get(id(record)))
        for record in arm.records
    }

    states = {
        state: sum(1 for record in scored if state_of[id(record)] == state)
        for state in END_STATES
    }
    failures = {
        failure: sum(1 for record in scored if record.get("error_class") == failure)
        for failure in FAILURE_CLASSES
    }
    stages: dict[str, int] = {}
    for record in scored:
        if record.get("error_stage"):
            stages[record["error_stage"]] = stages.get(record["error_stage"], 0) + 1

    by_task: dict[str, Tally] = {task: Tally() for task in TASK_TYPES}
    for record, gold in paired:
        tally = by_task.get(gold.intent)
        if tally is None:
            continue
        state = state_of[id(record)]
        if gold.intent == "needs_clarification":
            tally.add(state == runner.STATE_CLARIFICATION)
        else:
            tally.add(state == runner.STATE_FULL_SUCCESS)

    overall = Tally()
    for record, _ in paired:
        overall.add(state_of[id(record)] == runner.STATE_FULL_SUCCESS)

    durations = sorted(float(record.get("duration_sec") or 0.0) for record in scored)

    return {
        "schema_loss": schema_loss(scored),
        "arm": arm,
        "n_total": total,
        "n_scored": len(scored),
        "n_gaps": gaps,
        "n_gold": len(paired),
        "n_gold_gaps": len(paired_all) - len(paired),
        "states": states,
        "failures": failures,
        "stages": stages,
        "model_failures": sum(
            count for cls, count in failures.items() if cls in MODEL_CLASSES
        ),
        "infra_failures": sum(
            count for cls, count in failures.items() if cls in INFRA_CLASSES
        ),
        "by_task": by_task,
        "overall": overall,
        # Both halves of Table 1b: how often the model swapped the task for an
        # easier one, and what the pre-fix completion test would have reported.
        "mode_evasion": sum(
            1
            for record, gold in paired
            if gold.intent == "restrictions"
            and (record.get("restriction_plan") or {}).get("mode") == "buffers_only"
        ),
        "naive_success": sum(
            1
            for record, _ in paired
            if record.get("end_state") == runner.STATE_FULL_SUCCESS
        ),
        "plan": score_plan(paired),
        "median_sec": durations[len(durations) // 2] if durations else 0.0,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def md_table(headers: list[str], rows: list[list]) -> str:
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def pct_cell(count: int, total: int) -> str:
    return f"{100.0 * count / total:.1f}" if total else "—"


def report(evals: list[dict]) -> str:
    evals = sorted(evals, key=lambda e: (-(e["overall"].pct or -1), e["arm"].label))
    lines = [
        "# In-process evaluation — restrictions pipeline vs expert gold\n",
        "Runs driven in code (no HTTP interface, no Redis, no ChatStorage). "
        "Every percentage below is over rows that say something about the model: "
        "rows whose data was unavailable offline are excluded and reported "
        "separately in Table 0.\n",
    ]

    lines.append("\n## Table 0. Coverage\n")
    lines.append(
        "`data_unavailable` means the offline store had no answer for a request "
        "the plan made — a gap in the prefetched data, not a model failure. It is "
        "excluded from every other table. A non-zero column here is an "
        "instruction to re-run `prefetch_scenarios.py`, not a result.\n"
    )
    lines.append(
        md_table(
            ["Run", "Rows", "Scored", "Data gaps", "Gold rows", "Median s"],
            [
                [
                    e["arm"].label,
                    e["n_total"],
                    e["n_scored"],
                    e["n_gaps"],
                    e["n_gold"],
                    f"{e['median_sec']:.1f}",
                ]
                for e in evals
            ],
        )
    )

    lines.append("\n\n## Table 1. Task-aware success by task type (gold set)\n")
    lines.append(
        "Success is what the task actually requires: a `restrictions` task needs "
        "both output layers and an answer; a `buffers_only` task has no `objects` "
        "layer by design and is not penalised for lacking one; a "
        "`needs_clarification` task succeeds by asking. The single universal "
        "completion proxy the previous report used understates the first and "
        "overstates nothing — which is why it is replaced rather than kept.\n\n"
        "The criterion comes from the gold task, never from the mode the model "
        "declared for itself. Reading it from the model's own mode lets a run "
        "score by lowering the bar — see Table 1b.\n"
    )
    lines.append(
        md_table(
            ["Run", "Buffers-only", "Restrictions", "Clarification", "Overall"],
            [
                [
                    e["arm"].label,
                    e["by_task"]["buffers_only"].cell(),
                    e["by_task"]["restrictions"].cell(),
                    e["by_task"]["needs_clarification"].cell(),
                    e["overall"].cell(),
                ]
                for e in evals
            ],
        )
    )

    lines.append("\n\n## Table 1b. Mode evasion\n")
    lines.append(
        "Rows whose gold task asks for a count and where the model declared "
        "`buffers_only` — it draws the zone and never counts anything inside it. "
        "The pipeline runs such a plan happily and it produces a layer, so a "
        "completion test that trusts the declared mode reads it as a success. "
        "**Reported** is what that test would have said; **actual** is the same "
        "rows judged by what the task asked for. A large gap between the two "
        "columns means the model is answering an easier question than the one "
        "put to it, and any success rate quoted for it is about the substitution "
        "rather than about the task.\n"
    )
    lines.append(
        md_table(
            ["Run", "Declared buffers_only", "Reported success", "Actual success"],
            [
                [
                    e["arm"].label,
                    f"{e['mode_evasion']} ({100 * e['mode_evasion'] / max(e['n_gold'], 1):.1f}%)",
                    f"{e['naive_success']} ({100 * e['naive_success'] / max(e['n_gold'], 1):.1f}%)",
                    e["overall"].cell(),
                ]
                for e in evals
            ],
        )
    )

    lines.append("\n\n## Table 2. Plan correctness (gold set)\n")
    lines.append(
        "Scored against the `RestrictionPlan` **the model emitted** (`model_plan`), "
        "not the one the pipeline went on to execute. The two differ: "
        "canonicalisation drops restriction rules that carry no `target_names`, an "
        "empty rule list flips the mode to `needs_clarification`, and that flip "
        "blanks `target_entities`. Scoring the executed plan therefore reports a "
        "missing target entity for a model that named it correctly — see Table 2b. "
        "Each field is scored only on gold rows the parser is confident about, and "
        "the comparable count is shown next to the percentage. Rows that failed "
        "before a plan existed are not counted here — they are already counted as "
        "planning failures in Table 3.\n"
    )
    lines.append(
        "The entity columns are named by role rather than by either side's field "
        "name, because the two disagree: the gold set calls the buffered entity "
        "`target` while the plan schema builds buffers around `source_entities`. "
        "**Buffered** is the entity the zone is drawn around (plan "
        "`source_entities`, gold `target_entity`); **counted** is the entity "
        "found inside it (plan `target_entities`, gold `source_entity`).\n"
    )
    lines.append(
        md_table(
            ["Run", "Intent", "Buffered entity", "Counted entity", "Distance"],
            [
                [
                    e["arm"].label,
                    e["plan"]["intent"].cell(),
                    e["plan"]["buffered_entity"].cell(),
                    e["plan"]["counted_entity"].cell(),
                    e["plan"]["distance"].cell(),
                ]
                for e in evals
            ],
        )
    )

    lines.append("\n\n## Table 2b. What the typed schema costs\n")
    lines.append(
        "Rows where the model planned `restrictions` and the pipeline refused the "
        "plan. These are not model errors of intent or of entity choice — Table 2 "
        "scores those on the model's own plan — but failures to satisfy the "
        "schema exactly, and the pipeline treats a partial plan as no plan at "
        "all. `missing target_names` is the observed cause: the counted entity is "
        "named in `target_entities` and omitted from "
        "`restriction_rules[].target_names`, which must agree.\n"
    )
    lines.append(
        md_table(
            ["Run", "Plans downgraded", "of which: missing `target_names`"],
            [
                [
                    e["arm"].label,
                    e["schema_loss"]["downgraded"],
                    e["schema_loss"]["missing_target_names"],
                ]
                for e in evals
            ],
        )
    )

    lines.append(
        "\n\n## Table 3. Mutually-exclusive end states (% of scored rows, sums to 100)\n"
    )
    lines.append(
        md_table(
            ["Run"] + END_STATES,
            [
                [e["arm"].label]
                + [pct_cell(e["states"][state], e["n_scored"]) for state in END_STATES]
                for e in evals
            ],
        )
    )

    lines.append("\n\n## Table 4. Failure taxonomy (row counts)\n")
    lines.append(
        "Each class is decided where the exception was raised, not by matching "
        "the error message afterwards. `other` is an unclassified failure — a gap "
        "in this taxonomy — and each such record keeps a truncated traceback.\n"
    )
    lines.append(
        md_table(
            ["Run"] + FAILURE_CLASSES + ["model", "infra"],
            [
                [e["arm"].label]
                + [e["failures"][cls] for cls in FAILURE_CLASSES]
                + [e["model_failures"], e["infra_failures"]]
                for e in evals
            ],
        )
    )

    stage_names = sorted({stage for e in evals for stage in e["stages"]})
    if stage_names:
        lines.append("\n\n## Table 5. Where failures happen (pipeline stage)\n")
        lines.append(
            md_table(
                ["Run"] + stage_names,
                [
                    [e["arm"].label]
                    + [e["stages"].get(stage, 0) for stage in stage_names]
                    for e in evals
                ],
            )
        )

    lines.extend(_comparison_sections(evals))
    lines.append(
        "\n\n## Not computable here\n"
        "- **Object-selection P/R/F1 and geometry IoU** — scored against the "
        "reference GeoJSON by `benchmarks/harness/geometry_eval.py`, which reads "
        "the layers this run wrote to disk (`--save-layers`).\n"
    )
    return "\n".join(lines)


def _comparison_sections(evals: list[dict]) -> list[str]:
    """Ablation and transport deltas, when both sides of a pair were run."""

    lines: list[str] = []
    by_key = {(e["arm"].model, e["arm"].arm, e["arm"].transport): e for e in evals}

    ablation_rows = []
    for (model, arm, transport), evaluation in sorted(by_key.items()):
        if arm != runner.ARM_BASE:
            continue
        other = by_key.get((model, runner.ARM_NO_CATALOG, transport))
        if other is None:
            continue
        ablation_rows.append(
            [
                f"{model} / {transport}",
                evaluation["plan"]["intent"].cell(),
                other["plan"]["intent"].cell(),
                evaluation["plan"]["buffered_entity"].cell(),
                other["plan"]["buffered_entity"].cell(),
                evaluation["overall"].cell(),
                other["overall"].cell(),
            ]
        )
    if ablation_rows:
        lines.append("\n\n## Table 6. Ablation — domain-catalog grounding\n")
        lines.append(
            "The same model and transport with and without the scenario catalog "
            "in the planning prompt. This is what shows the architecture does "
            "work, rather than merely providing a harness to compare models in.\n"
        )
        lines.append(
            md_table(
                [
                    "Model / transport",
                    "Intent +cat",
                    "Intent −cat",
                    "Buffered +cat",
                    "Buffered −cat",
                    "Success +cat",
                    "Success −cat",
                ],
                ablation_rows,
            )
        )

    transport_rows = []
    for (model, arm, transport), evaluation in sorted(by_key.items()):
        if transport != runner.LOCAL:
            continue
        other = by_key.get((model, arm, runner.MCP_HTTP))
        if other is None:
            continue
        transport_rows.append(
            [
                f"{model} / {arm}",
                evaluation["n_scored"],
                evaluation["infra_failures"],
                other["infra_failures"],
                evaluation["model_failures"],
                other["model_failures"],
                f"{evaluation['median_sec']:.1f}",
                f"{other['median_sec']:.1f}",
            ]
        )
    if transport_rows:
        lines.append("\n\n## Table 7. Transport cost — in-process vs MCP over HTTP\n")
        lines.append(
            "Same model, same data, same plan-building code; only the tool "
            "transport differs. The infrastructure-failure columns are the "
            "measured answer to how much of the previous experiment's error rate "
            "belonged to the transport rather than to the model.\n"
        )
        lines.append(
            md_table(
                [
                    "Model / arm",
                    "Rows",
                    "Infra local",
                    "Infra HTTP",
                    "Model local",
                    "Model HTTP",
                    "Median s local",
                    "Median s HTTP",
                ],
                transport_rows,
            )
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", default="benchmarks/data/results_inproc")
    parser.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    parser.add_argument("--out", default="benchmarks/out/inproc_report.md")
    args = parser.parse_args()

    results_dir = Path(args.results)
    arms = load_arms(results_dir)
    if not arms:
        raise SystemExit(
            f"no results found under {results_dir}/<model>/<arm>--<transport>/results.jsonl"
        )

    gold = load_gold(args.gold)
    gold_index = {
        (record.scenario_id, norm(record.question)): record for record in gold
    }
    print(f"gold: {len(gold)} records; runs: {len(arms)}")

    evals = [evaluate(arm, gold_index) for arm in arms]
    for evaluation in evals:
        overall = evaluation["overall"]
        print(
            f"  {evaluation['arm'].label:44s} "
            f"scored={evaluation['n_scored']:4d} gaps={evaluation['n_gaps']:3d} "
            f"gold={evaluation['n_gold']:3d} success={overall.cell()}"
        )
        if evaluation["n_gaps"]:
            print(
                f"    {evaluation['n_gaps']} rows had no offline data — "
                f"re-run prefetch_scenarios.py before trusting the coverage"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report(evals), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
