"""Unit tests for the in-process evaluation report.

These numbers go into the paper, so the tests pin the claims the tables make:

* a data gap never enters a model-facing denominator;
* success is task-aware — a buffers_only task is not judged against a layer it is
  not supposed to produce;
* plan fields are scored only where the gold parser is confident, and the
  comparable count travels with every percentage;
* a row that failed before planning is counted once, as a planning failure, and
  not again as a wrong entity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "benchmarks" / "eval"))
sys.path.insert(0, str(_ROOT / "benchmarks" / "harness"))

import inproc_eval as ev  # noqa: E402
import inproc_runner as runner  # noqa: E402
from gold_parser import GoldRecord  # noqa: E402


def _gold(
    scenario_id: int = 7,
    question: str = "вопрос",
    intent: str = "restrictions",
    source: str | None = "Школа",
    target: str | None = "Жилой дом",
    distance: int | None = 100,
    conf: dict | None = None,
) -> GoldRecord:
    record = GoldRecord(
        scenario_id=scenario_id,
        project="проект",
        question=question,
        answer="ответ",
        layers_spec="слои",
        catalog=["Школа", "Жилой дом"],
        intent=intent,
        source_entity=source,
        target_entity=target,
        distance_m=distance,
    )
    record.conf = (
        conf
        if conf is not None
        else {
            "intent": True,
            "source": True,
            "target": True,
            "distance": True,
        }
    )
    return record


def _plan(
    mode: str = "restrictions",
    sources: list[str] | None = None,
    targets: list[str] | None = None,
    distances: list[float] | None = None,
) -> dict:
    return {
        "mode": mode,
        "source_entities": [
            {"name": name, "entity_type": "service"}
            for name in (sources if sources is not None else ["Школа"])
        ],
        "target_entities": [
            {"name": name, "entity_type": "physical_object"}
            for name in (targets if targets is not None else ["Жилой дом"])
        ],
        "buffer_rules": [
            {
                "source_name": "Школа",
                "buffer_size": d,
                "buffer_type": "round",
                "title": "t",
            }
            for d in (distances if distances is not None else [100.0])
        ],
    }


def _record(
    idx: int = 0,
    prompt: str = "вопрос",
    scenario_id: int = 7,
    end_state: str = runner.STATE_FULL_SUCCESS,
    plan: dict | None = None,
    error_class: str | None = None,
    error_stage: str | None = None,
    duration: float = 1.0,
) -> dict:
    return {
        "idx": idx,
        "model": "gemma3:12b",
        "transport": runner.LOCAL,
        "arm": runner.ARM_BASE,
        "prompt": prompt,
        "scenario_id": scenario_id,
        "restriction_plan": plan if plan is not None else _plan(),
        "layer_counts": {"objects": 3, "generators": 1},
        "end_state": end_state,
        "error_class": error_class,
        "error_stage": error_stage,
        "duration_sec": duration,
    }


def _arm(
    records: list[dict], arm: str = runner.ARM_BASE, transport: str = runner.LOCAL
):
    return ev.Arm(model="gemma3:12b", arm=arm, transport=transport, records=records)


def _index(*golds: GoldRecord) -> dict:
    return {(g.scenario_id, ev.norm(g.question)): g for g in golds}


# --------------------------------------------------------------------------- #
# data gaps
# --------------------------------------------------------------------------- #
def test_data_gaps_are_excluded_from_the_denominator():
    gold = _index(_gold(question="a"), _gold(question="b"))
    records = [
        _record(idx=0, prompt="a"),
        _record(
            idx=1,
            prompt="b",
            end_state=runner.STATE_DATA_UNAVAILABLE,
            error_class=runner.CLS_DATA_UNAVAILABLE,
        ),
    ]

    result = ev.evaluate(_arm(records), gold)

    assert result["n_total"] == 2
    assert result["n_scored"] == 1
    assert result["n_gaps"] == 1
    # one scored gold row, and it succeeded — not 50 %
    assert result["overall"].comparable == 1
    assert result["overall"].pct == 100.0


def test_data_gaps_are_reported_so_they_cannot_pass_unnoticed():
    gold = _index(_gold(question="a"))
    records = [
        _record(
            idx=0,
            prompt="a",
            end_state=runner.STATE_DATA_UNAVAILABLE,
            error_class=runner.CLS_DATA_UNAVAILABLE,
        )
    ]

    result = ev.evaluate(_arm(records), gold)
    text = ev.report([result])

    assert result["n_gaps"] == 1
    assert "Data gaps" in text
    assert "prefetch_scenarios.py" in text


# --------------------------------------------------------------------------- #
# task-aware success
# --------------------------------------------------------------------------- #
def test_success_is_broken_out_by_task_type():
    gold = _index(
        _gold(question="a", intent="restrictions"),
        _gold(question="b", intent="buffers_only"),
        _gold(question="c", intent="needs_clarification"),
    )
    records = [
        _record(idx=0, prompt="a", end_state=runner.STATE_FULL_SUCCESS),
        _record(idx=1, prompt="b", end_state=runner.STATE_FULL_SUCCESS),
        _record(idx=2, prompt="c", end_state=runner.STATE_CLARIFICATION),
    ]

    result = ev.evaluate(_arm(records), gold)

    assert result["by_task"]["restrictions"].cell() == "100.0% (1/1)"
    assert result["by_task"]["buffers_only"].cell() == "100.0% (1/1)"
    assert result["by_task"]["needs_clarification"].cell() == "100.0% (1/1)"


def test_a_clarification_task_succeeds_by_asking_not_by_producing_layers():
    gold = _index(_gold(question="c", intent="needs_clarification"))
    produced = [_record(idx=0, prompt="c", end_state=runner.STATE_FULL_SUCCESS)]
    asked = [_record(idx=0, prompt="c", end_state=runner.STATE_CLARIFICATION)]

    assert (
        ev.evaluate(_arm(produced), gold)["by_task"]["needs_clarification"].correct == 0
    )
    assert ev.evaluate(_arm(asked), gold)["by_task"]["needs_clarification"].correct == 1


def test_task_types_absent_from_the_gold_slice_read_as_not_applicable():
    gold = _index(_gold(question="a", intent="restrictions"))
    result = ev.evaluate(_arm([_record(prompt="a")]), gold)

    assert result["by_task"]["buffers_only"].cell() == "n/a"


# --------------------------------------------------------------------------- #
# plan scoring
# --------------------------------------------------------------------------- #
def test_the_gold_and_plan_role_conventions_are_opposite():
    """The single most dangerous thing in this file, pinned.

    gold_parser calls the BUFFERED entity `target` and the COUNTED entity
    `source`; the RestrictionPlan schema builds buffers around `source_entities`.
    Comparing the two `source` fields to each other looks obviously right, is
    wrong, and would report near-zero entity accuracy charged to the models.

    Here the plan buffers around "Школа" and counts "Жилой дом" inside it, and
    the gold row says the same thing in its own vocabulary — inverted.
    """

    gold = _index(
        _gold(
            question="a",
            source="Жилой дом",  # counted inside the zone
            target="Школа",  # the zone is drawn around this
        )
    )
    records = [
        _record(prompt="a", plan=_plan(sources=["Школа"], targets=["Жилой дом"]))
    ]

    plan = ev.evaluate(_arm(records), gold)["plan"]

    assert plan["buffered_entity"].cell() == "100.0% (1/1)"
    assert plan["counted_entity"].cell() == "100.0% (1/1)"


def test_a_plan_with_the_roles_swapped_scores_zero():
    """A model that buffers the wrong way round must not be scored as correct."""

    gold = _index(_gold(question="a", source="Жилой дом", target="Школа"))
    records = [
        _record(prompt="a", plan=_plan(sources=["Жилой дом"], targets=["Школа"]))
    ]

    plan = ev.evaluate(_arm(records), gold)["plan"]

    assert plan["buffered_entity"].correct == 0
    assert plan["counted_entity"].correct == 0


def test_plan_fields_are_scored_against_the_executed_plan():
    gold = _index(_gold(question="a", source="Жилой дом", target="Школа", distance=100))
    records = [
        _record(prompt="a", plan=_plan(sources=["Школа"], targets=["Жилой дом"]))
    ]

    plan = ev.evaluate(_arm(records), gold)["plan"]

    assert plan["intent"].cell() == "100.0% (1/1)"
    assert plan["buffered_entity"].cell() == "100.0% (1/1)"
    assert plan["counted_entity"].cell() == "100.0% (1/1)"
    assert plan["distance"].cell() == "100.0% (1/1)"


def test_a_wrong_entity_and_a_wrong_distance_are_counted_as_wrong():
    gold = _index(_gold(question="a", target="Школа", distance=100))
    records = [
        _record(prompt="a", plan=_plan(sources=["Поликлиника"], distances=[300.0]))
    ]

    plan = ev.evaluate(_arm(records), gold)["plan"]

    assert plan["buffered_entity"].cell() == "0.0% (0/1)"
    assert plan["distance"].cell() == "0.0% (0/1)"


def test_entity_match_ignores_case_and_yo():
    gold = _index(_gold(question="a", target="Приёмная", source=None))
    records = [_record(prompt="a", plan=_plan(sources=["приемная"]))]

    assert ev.evaluate(_arm(records), gold)["plan"]["buffered_entity"].correct == 1


def test_a_plan_naming_extra_entities_still_counts_the_gold_one():
    """The plan may legitimately carry more than the gold row names."""

    gold = _index(_gold(question="a", target="Школа"))
    records = [_record(prompt="a", plan=_plan(sources=["Детский сад", "Школа"]))]

    assert ev.evaluate(_arm(records), gold)["plan"]["buffered_entity"].correct == 1


def test_low_confidence_gold_fields_are_not_scored():
    """Scoring a model against a value the gold parser is unsure of is indefensible."""

    gold = _index(
        _gold(
            question="a",
            conf={"intent": True, "source": False, "target": False, "distance": False},
        )
    )
    records = [_record(prompt="a", plan=_plan(sources=["что-то другое"]))]

    plan = ev.evaluate(_arm(records), gold)["plan"]

    assert plan["buffered_entity"].cell() == "n/a"
    assert plan["counted_entity"].cell() == "n/a"
    assert plan["distance"].cell() == "n/a"
    assert plan["intent"].comparable == 1


def test_a_row_with_no_plan_is_not_counted_as_a_wrong_entity():
    """It is already a planning failure; charging it twice double-counts."""

    gold = _index(_gold(question="a"))
    records = [
        _record(
            prompt="a",
            plan=None,
            end_state=runner.STATE_PLANNING_FAILURE,
            error_class=runner.CLS_INVALID_PLAN,
        )
    ]
    records[0]["restriction_plan"] = None

    result = ev.evaluate(_arm(records), gold)

    assert result["plan"]["buffered_entity"].comparable == 0
    assert result["states"][runner.STATE_PLANNING_FAILURE] == 1


def test_the_report_names_the_entity_columns_by_role_not_by_field_name():
    """A reader must not have to know which side's vocabulary a column uses."""

    gold = _index(_gold(question="a"))
    text = ev.report([ev.evaluate(_arm([_record(prompt="a")]), gold)])

    assert "Buffered entity" in text and "Counted entity" in text
    assert "| Source entity |" not in text


def test_distance_comparison_tolerates_a_float_representation():
    gold = _index(_gold(question="a", distance=100))
    records = [_record(prompt="a", plan=_plan(distances=[100.0000001]))]

    assert ev.evaluate(_arm(records), gold)["plan"]["distance"].correct == 1


# --------------------------------------------------------------------------- #
# taxonomy
# --------------------------------------------------------------------------- #
def test_model_and_infrastructure_failures_are_summed_separately():
    gold = _index(*(_gold(question=q) for q in "abcd"))
    records = [
        _record(
            idx=0,
            prompt="a",
            end_state=runner.STATE_PLANNING_FAILURE,
            error_class=runner.CLS_INVALID_PLAN,
        ),
        _record(
            idx=1,
            prompt="b",
            end_state=runner.STATE_PLANNING_FAILURE,
            error_class=runner.CLS_LLM_BACKEND,
        ),
        _record(
            idx=2,
            prompt="c",
            end_state=runner.STATE_TOOL_INFRA_FAILURE,
            error_class=runner.CLS_URBAN_API,
        ),
        _record(
            idx=3,
            prompt="d",
            end_state=runner.STATE_TOOL_INFRA_FAILURE,
            error_class=runner.CLS_TRANSPORT,
        ),
    ]

    result = ev.evaluate(_arm(records), gold)

    assert result["model_failures"] == 2
    assert result["infra_failures"] == 2


def test_the_failure_class_is_read_not_re_derived_from_the_message():
    """No substring matching: the record already carries the decided class."""

    gold = _index(_gold(question="a"))
    records = [
        _record(
            idx=0,
            prompt="a",
            end_state=runner.STATE_TOOL_INFRA_FAILURE,
            error_class=runner.CLS_TOOL_EXECUTION,
        )
    ]
    # a message that a substring classifier would file under urban_api
    records[0]["error"] = "ToolError: Urban API mentioned in passing"

    result = ev.evaluate(_arm(records), gold)

    assert result["failures"][runner.CLS_TOOL_EXECUTION] == 1
    assert result["failures"][runner.CLS_URBAN_API] == 0


def test_end_states_sum_to_the_scored_row_count():
    gold = _index(*(_gold(question=q) for q in "abc"))
    records = [
        _record(idx=0, prompt="a", end_state=runner.STATE_FULL_SUCCESS),
        _record(idx=1, prompt="b", end_state=runner.STATE_PARTIAL_SPATIAL),
        _record(
            idx=2,
            prompt="c",
            end_state=runner.STATE_TIMEOUT,
            error_class=runner.CLS_TIMEOUT,
        ),
    ]

    result = ev.evaluate(_arm(records), gold)

    assert sum(result["states"].values()) == result["n_scored"] == 3


def test_failure_stages_are_counted():
    gold = _index(_gold(question="a"))
    records = [
        _record(
            prompt="a",
            end_state=runner.STATE_TOOL_INFRA_FAILURE,
            error_class=runner.CLS_TOOL_EXECUTION,
            error_stage="buffer_creation",
        )
    ]

    assert ev.evaluate(_arm(records), gold)["stages"] == {"buffer_creation": 1}


# --------------------------------------------------------------------------- #
# loading and reporting
# --------------------------------------------------------------------------- #
def test_arms_are_discovered_from_the_directory_layout(tmp_path):
    for arm_dir in ("base--local", "no_catalog--local", "base--mcp-http"):
        path = tmp_path / "gemma3_12b" / arm_dir
        path.mkdir(parents=True)
        (path / "results.jsonl").write_text(
            json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8"
        )

    arms = ev.load_arms(tmp_path)

    assert {(a.arm, a.transport) for a in arms} == {
        ("base", "local"),
        ("no_catalog", "local"),
        ("base", "mcp-http"),
    }


def test_a_torn_last_line_does_not_fail_the_whole_report(tmp_path):
    path = tmp_path / "m" / "base--local"
    path.mkdir(parents=True)
    (path / "results.jsonl").write_text(
        json.dumps(_record(), ensure_ascii=False) + "\n" + '{"idx": 1, "prom',
        encoding="utf-8",
    )

    arms = ev.load_arms(tmp_path)

    assert len(arms) == 1
    assert len(arms[0].records) == 1


def test_each_gold_row_is_counted_once_even_with_paraphrases():
    """The augmented set repeats a gold question; the gold slice must not double it."""

    gold = _index(_gold(question="вопрос"))
    records = [
        _record(idx=0, prompt="вопрос"),
        _record(idx=1, prompt="Вопрос"),  # normalises to the same key
    ]

    assert ev.evaluate(_arm(records), gold)["n_gold"] == 1


def test_the_report_renders_the_ablation_and_transport_comparisons():
    gold = _index(_gold(question="a"))
    base_local = ev.evaluate(_arm([_record(prompt="a")]), gold)
    no_catalog = ev.evaluate(
        _arm([_record(prompt="a")], arm=runner.ARM_NO_CATALOG), gold
    )
    base_http = ev.evaluate(
        _arm([_record(prompt="a")], transport=runner.MCP_HTTP), gold
    )

    text = ev.report([base_local, no_catalog, base_http])

    assert "Ablation — domain-catalog grounding" in text
    assert "Transport cost" in text
    assert "Task-aware success by task type" in text
    assert "Plan correctness" in text


def test_comparison_tables_are_omitted_when_only_one_side_was_run():
    gold = _index(_gold(question="a"))
    text = ev.report([ev.evaluate(_arm([_record(prompt="a")]), gold)])

    assert "Ablation — domain-catalog grounding" not in text
    assert "Transport cost" not in text


def test_an_empty_results_directory_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["inproc_eval.py", "--results", str(tmp_path)])
    with pytest.raises(SystemExit):
        ev.main()
