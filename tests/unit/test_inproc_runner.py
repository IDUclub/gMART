"""Unit tests for the in-process experiment runner.

The runner produces the numbers the paper reports, so the two things tested hardest
are the ones a reviewer would attack:

* **end states are mutually exclusive and mean what they say** — in particular a
  ``buffers_only`` task is not marked incomplete for lacking an ``objects`` layer,
  which is the flaw in the old universal completion proxy;
* **a data gap is never scored as a model failure** — an offline store miss has
  its own class and its own end state, outside every model-facing denominator.

There is also a full pipeline run with no network at all: a fake LLM, a fake tool
client, and an assertion that the plan is captured — which the SSE harness could
not do, since the pipeline emits no plan event.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "harness"))

import inproc_runner as runner  # noqa: E402

from src.idu_mcp.common.api_handlers.urban_data_store import (  # noqa: E402
    UrbanDataUnavailable,
)


def _record(**kwargs) -> runner.RunRecord:
    base = {
        "idx": 0,
        "model": "gemma3:12b",
        "transport": runner.LOCAL,
        "arm": runner.ARM_BASE,
        "prompt": "сколько жилых домов в 100 м от школы",
        "scenario_id": 7,
    }
    base.update(kwargs)
    return runner.RunRecord(**base)


# --------------------------------------------------------------------------- #
# failure classification
# --------------------------------------------------------------------------- #
def test_offline_gap_is_its_own_class():
    exc = UrbanDataUnavailable("v1/scenarios/7/services_with_geometry", {"id": 3})

    assert runner.classify(exc) == runner.CLS_DATA_UNAVAILABLE


def test_urban_api_and_geometry_tool_errors_are_told_apart():
    from fastmcp.exceptions import ToolError

    urban = ToolError("Urban API (http://urban) returned status 502: bad gateway")
    geometry = ToolError("Ошибка при построении буферов: invalid geometry")

    assert runner.classify(urban) == runner.CLS_URBAN_API
    assert runner.classify(geometry) == runner.CLS_TOOL_EXECUTION


def test_model_backend_failure_is_not_an_infrastructure_failure():
    from src.agents.model_clients.llm_base import LlmResponseError

    assert (
        runner.classify(LlmResponseError("context length exceeded", status_code=400))
        == runner.CLS_LLM_BACKEND
    )


def test_token_expiry_is_classed_separately():
    from src.agents.common.exceptions.token_exceptions import TokenExpiredError

    assert runner.classify(TokenExpiredError("401")) == runner.CLS_TOKEN_EXPIRED


def test_a_plan_that_resolves_to_nothing_runnable_is_the_model_s_failure():
    """The executor's own ValueError must not land in the infrastructure column.

    These two are what remains once `target_names` is required: the plan is valid
    JSON and valid against the schema, but names entities that resolve to no
    buildable layer. Filed as `other` they became `tool_infra_failure`, which
    blames the tools for a plan the model wrote.
    """

    for message in (
        "No source layers found for buffer construction",
        "No valid restriction relations found in the plan",
    ):
        assert runner.classify(ValueError(message)) == runner.CLS_UNEXECUTABLE_PLAN

    record = _record(error_class=runner.CLS_UNEXECUTABLE_PLAN)
    assert runner.end_state(record) == runner.STATE_PLANNING_FAILURE


def test_unclassified_failure_falls_back_to_other():
    assert runner.classify(RuntimeError("something new")) == runner.CLS_OTHER


# --------------------------------------------------------------------------- #
# end states
# --------------------------------------------------------------------------- #
def test_restrictions_success_needs_both_layers_and_an_answer():
    record = _record(
        restriction_plan={"mode": "restrictions"},
        layer_counts={"objects": 12, "generators": 3},
        llm_response="В радиусе 100 м находится 12 домов.",
    )

    assert runner.end_state(record) == runner.STATE_FULL_SUCCESS


def test_restrictions_missing_a_layer_is_partial_not_success():
    record = _record(
        restriction_plan={"mode": "restrictions"},
        layer_counts={"generators": 3},
        llm_response="текст",
    )

    assert runner.end_state(record) == runner.STATE_PARTIAL_SPATIAL


def test_buffers_only_success_does_not_require_an_objects_layer():
    """The flaw in the old universal completion proxy, made explicit.

    A correct buffers_only task has no target `objects` layer by design; scoring
    it against one understates every buffers_only row.
    """

    record = _record(
        restriction_plan={"mode": "buffers_only"},
        layer_counts={"буфер школа": 5},
        llm_response="Построены буферы 100 м вокруг школ.",
    )

    assert runner.end_state(record) == runner.STATE_FULL_SUCCESS


def test_layers_without_a_final_answer_are_partial():
    record = _record(
        restriction_plan={"mode": "restrictions"},
        layer_counts={"objects": 12, "generators": 3},
        llm_response="   ",
    )

    assert runner.end_state(record) == runner.STATE_PARTIAL_SPATIAL


def test_display_layer_names_count_as_the_restriction_layers():
    """The pipeline renames objects/generators for display before emitting them."""

    record = _record(
        restriction_plan={"mode": "restrictions"},
        layer_counts={"Объекты в зоне ограничений": 12, "Источники ограничений": 3},
        llm_response="ответ",
    )

    assert runner.end_state(record) == runner.STATE_FULL_SUCCESS


def test_clarification_is_its_own_end_state():
    record = _record(
        restriction_plan={"mode": "needs_clarification"},
        clarification="Какое расстояние вас интересует?",
    )

    assert runner.end_state(record) == runner.STATE_CLARIFICATION


def test_data_gap_is_not_a_model_failure():
    record = _record(
        error="Urban API data unavailable offline",
        error_class=runner.CLS_DATA_UNAVAILABLE,
    )

    state = runner.end_state(record)
    assert state == runner.STATE_DATA_UNAVAILABLE
    assert state not in {runner.STATE_PLANNING_FAILURE, runner.STATE_TOOL_INFRA_FAILURE}


def test_planning_and_infrastructure_failures_are_separate_states():
    planning = _record(error_class=runner.CLS_INVALID_PLAN, error="bad plan")
    infra = _record(error_class=runner.CLS_URBAN_API, error="502")

    assert runner.end_state(planning) == runner.STATE_PLANNING_FAILURE
    assert runner.end_state(infra) == runner.STATE_TOOL_INFRA_FAILURE


def test_no_error_and_no_layers_is_empty():
    assert runner.end_state(_record()) == runner.STATE_EMPTY


def test_every_failure_class_maps_to_exactly_one_end_state():
    """The end states must partition the rows — the reviewer asks them to sum to 100 %."""

    classes = [
        runner.CLS_DATA_UNAVAILABLE,
        runner.CLS_URBAN_API,
        runner.CLS_TOOL_EXECUTION,
        runner.CLS_INVALID_PLAN,
        runner.CLS_LLM_BACKEND,
        runner.CLS_TIMEOUT,
        runner.CLS_TOKEN_EXPIRED,
        runner.CLS_TRANSPORT,
        runner.CLS_OTHER,
    ]
    states = {
        runner.STATE_FULL_SUCCESS,
        runner.STATE_PARTIAL_SPATIAL,
        runner.STATE_CLARIFICATION,
        runner.STATE_PLANNING_FAILURE,
        runner.STATE_TOOL_INFRA_FAILURE,
        runner.STATE_TIMEOUT,
        runner.STATE_EMPTY,
        runner.STATE_DATA_UNAVAILABLE,
    }

    for failure_class in classes:
        state = runner.end_state(_record(error="x", error_class=failure_class))
        assert state in states, failure_class


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def test_resume_redoes_failed_rows_but_not_successful_ones(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False)
            for row in (
                {"idx": 0, "error": None},
                {"idx": 1, "error": "Urban API unreachable"},
                {"idx": 2, "error": None},
            )
        ),
        encoding="utf-8",
    )

    assert runner.done_indices(results) == {0, 2}


def test_missing_results_file_means_nothing_is_done(tmp_path):
    assert runner.done_indices(tmp_path / "absent.jsonl") == set()


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def test_dataset_missing_a_required_column_fails_loudly(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        runner.load_dataset(path, None)
    assert runner.COL_Q in str(excinfo.value)


def test_dataset_rows_without_a_scenario_are_dropped(tmp_path):
    path = tmp_path / "gold.csv"
    path.write_text(
        f"{runner.COL_Q},{runner.COL_SID}\nвопрос,7\nбез сценария,\n",
        encoding="utf-8",
    )

    frame = runner.load_dataset(path, None)
    assert len(frame) == 1


def test_schema_ablation_reproduces_the_historical_optional_target_names():
    required = runner.RestrictionPlan.model_json_schema()["$defs"]["RestrictionRule"]
    optional = runner.OptionalTargetNamesPlan.model_json_schema()["$defs"][
        "OptionalTargetNamesRule"
    ]

    assert "target_names" in required["required"]
    assert required["properties"]["target_names"]["minItems"] == 1
    assert "target_names" not in optional["required"]
    assert "minItems" not in optional["properties"]["target_names"]


def test_cluster_balanced_sample_is_deterministic(tmp_path):
    path = tmp_path / "sample.csv"
    rows = [f"q-{sid}-{i},{sid}" for sid in (1, 2) for i in range(6)]
    path.write_text(
        f"{runner.COL_Q},{runner.COL_SID}\n" + "\n".join(rows), encoding="utf-8"
    )

    first = runner.load_dataset(path, None, sample_per_scenario=3, sample_seed=17)
    second = runner.load_dataset(path, None, sample_per_scenario=3, sample_seed=17)

    assert len(first) == 6
    assert first[runner.COL_Q].tolist() == second[runner.COL_Q].tolist()
    assert first.groupby(runner.COL_SID).size().to_dict() == {1: 3, 2: 3}


# --------------------------------------------------------------------------- #
# a whole run, with no network
# --------------------------------------------------------------------------- #
class FakePlan:
    """Stands in for a RestrictionPlan: only model_dump and mode are used."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def model_dump(self, mode="json"):
        return {"mode": self.mode, "source_entities": [{"name": "школа"}]}


class FakeService:
    """A pipeline that yields the real event shapes without any dependencies."""

    def __init__(self, events, plan: FakePlan | None = None, error=None) -> None:
        self._events = events
        self._error = error
        self.plan = plan or FakePlan("restrictions")

        class Builder:
            async def build_plan(inner, *args, **kwargs):  # noqa: N805
                return self.plan

        self.plan_builder = Builder()

    async def run_restriction_execution_pipline(self, **kwargs):
        # the pipeline builds the plan partway through, as the real one does
        for event in self._events:
            if event == "__plan__":
                await self.plan_builder.build_plan()
                continue
            yield event
        if self._error:
            raise self._error


def _events_for_a_successful_restrictions_run():
    return [
        {"type": "pipeline_started", "content": {"request_id": "r1"}},
        {"type": "status", "content": {"status": "data_retrievement", "text": ""}},
        "__plan__",
        {
            "type": "tool_call",
            "content": {
                "execution_mode": "data_retrievement",
                "mcp_source": "LOCAL_IDU_TOOLS",
                "tool_calls": [{"function": {"name": "GetServices"}}],
            },
        },
        {"type": "status", "content": {"status": "buffer_creation", "text": ""}},
        {
            "type": "feature_collection",
            "content": {
                "name": "objects",
                "feature_collection": {"type": "FeatureCollection", "features": [1, 2]},
            },
        },
        {
            "type": "feature_collection",
            "content": {
                "name": "generators",
                "feature_collection": {"type": "FeatureCollection", "features": [1]},
            },
        },
        {"type": "chunk", "content": {"text": "Найдено 2 объекта.", "done": True}},
    ]


async def test_a_full_run_records_plan_layers_and_stages(tmp_path):
    service = FakeService(_events_for_a_successful_restrictions_run())

    record = await runner.run_one(
        service,
        client=None,
        idx=3,
        model="gemma3:12b",
        prompt="вопрос",
        scenario_id=7,
        transport=runner.LOCAL,
        arm=runner.ARM_BASE,
        temperature=0.0,
        timeout=30,
        layers_dir=tmp_path,
    )

    # the plan the SSE harness could not see, because no event carries it
    assert record.restriction_plan == {
        "mode": "restrictions",
        "source_entities": [{"name": "школа"}],
    }
    assert record.layer_counts == {"objects": 2, "generators": 1}
    assert record.llm_response == "Найдено 2 объекта."
    assert record.end_state == runner.STATE_FULL_SUCCESS
    assert record.error is None
    assert [call["calls"] for call in record.tool_calls] == [["GetServices"]]
    assert {stage["stage"] for stage in record.stages} >= {
        runner.STAGE_START,
        "data_retrievement",
        "buffer_creation",
    }


async def test_layers_are_written_to_disk_for_geometry_scoring(tmp_path):
    service = FakeService(_events_for_a_successful_restrictions_run())

    record = await runner.run_one(
        service,
        client=None,
        idx=3,
        model="m",
        prompt="q",
        scenario_id=7,
        transport=runner.LOCAL,
        arm=runner.ARM_BASE,
        temperature=0.0,
        timeout=30,
        layers_dir=tmp_path,
    )

    written = sorted(path.name for path in (tmp_path / "00003").iterdir())
    assert written == ["generators.geojson", "objects.geojson"]
    stored = json.loads((tmp_path / "00003" / "objects.geojson").read_text("utf-8"))
    assert stored["features"] == [1, 2]
    assert record.layer_files["objects"].endswith("objects.geojson")


async def test_a_failure_records_the_stage_it_happened_in(tmp_path):
    from fastmcp.exceptions import ToolError

    events = [
        {"type": "status", "content": {"status": "data_retrievement", "text": ""}},
        {"type": "status", "content": {"status": "buffer_creation", "text": ""}},
    ]
    service = FakeService(events, error=ToolError("Ошибка построения буферов"))

    record = await runner.run_one(
        service,
        client=None,
        idx=0,
        model="m",
        prompt="q",
        scenario_id=7,
        transport=runner.LOCAL,
        arm=runner.ARM_BASE,
        temperature=0.0,
        timeout=30,
        layers_dir=None,
    )

    assert record.error_stage == "buffer_creation"
    assert record.error_class == runner.CLS_TOOL_EXECUTION
    assert record.end_state == runner.STATE_TOOL_INFRA_FAILURE


async def test_an_offline_gap_records_what_was_missing():
    service = FakeService(
        [{"type": "status", "content": {"status": "data_retrievement", "text": ""}}],
        error=UrbanDataUnavailable(
            "v1/scenarios/7/services_with_geometry", {"service_type_id": 3}
        ),
    )

    record = await runner.run_one(
        service,
        client=None,
        idx=0,
        model="m",
        prompt="q",
        scenario_id=7,
        transport=runner.LOCAL,
        arm=runner.ARM_BASE,
        temperature=0.0,
        timeout=30,
        layers_dir=None,
    )

    assert record.end_state == runner.STATE_DATA_UNAVAILABLE
    # enough for a prefetch pass to be told exactly what to fill
    assert record.missing_data == {
        "endpoint": "v1/scenarios/7/services_with_geometry",
        "params": {"service_type_id": 3},
    }


async def test_a_clarification_run_is_not_counted_as_empty():
    service = FakeService(
        [{"type": "chunk", "content": {"text": "Уточните радиус.", "done": True}}],
        plan=FakePlan("needs_clarification"),
    )
    service._events.insert(0, "__plan__")

    record = await runner.run_one(
        service,
        client=None,
        idx=0,
        model="m",
        prompt="q",
        scenario_id=7,
        transport=runner.LOCAL,
        arm=runner.ARM_BASE,
        temperature=0.0,
        timeout=30,
        layers_dir=None,
    )

    assert record.clarification == "Уточните радиус."
    assert record.end_state == runner.STATE_CLARIFICATION


async def test_the_plan_capture_is_installed_once_and_does_not_stack():
    """Rows share one service; a wrapper per row would nest on every run."""

    service = FakeService(_events_for_a_successful_restrictions_run())

    runner.install_plan_capture(service)
    after_first = service.plan_builder.build_plan
    runner.install_plan_capture(service)

    assert service.plan_builder.build_plan is after_first


async def test_concurrent_rows_do_not_pick_up_each_others_plan():
    """Rows share one service and one plan builder, as they do in a real arm.

    Each row must record the plan *its own* query produced. A capture kept on the
    builder, or a wrapper installed per row, would let the two cross — and with
    the default concurrency of 2 that would silently mis-attribute plans in the
    results the paper is built from.

    The interleaving is chosen so the ordering alone cannot save a broken
    implementation: the row that builds its plan *first* is the one that finishes
    *last*, so a single shared slot hands it the other row's plan.
    """

    import asyncio

    class InterleavingService(FakeService):
        """One builder, a plan per query, and a scripted schedule per query."""

        # query -> (delay before its plan is built, delay after)
        SCHEDULE = {"ранний": (0.01, 0.08), "поздний": (0.04, 0.01)}
        PLANS = {"ранний": "restrictions", "поздний": "buffers_only"}

        def __init__(self) -> None:
            super().__init__([])
            self.plans = {query: FakePlan(mode) for query, mode in self.PLANS.items()}
            service = self

            class Builder:
                async def build_plan(inner, query, **kwargs):  # noqa: N805
                    return service.plans[query]

            self.plan_builder = Builder()

        async def run_restriction_execution_pipline(self, **kwargs):
            query = kwargs["user_query"]
            before, after = self.SCHEDULE[query]
            await asyncio.sleep(before)
            await self.plan_builder.build_plan(query)
            await asyncio.sleep(after)
            yield {"type": "chunk", "content": {"text": f"ответ {query}", "done": True}}

    service = InterleavingService()

    def _row(query, idx):
        return runner.run_one(
            service,
            client=None,
            idx=idx,
            model="m",
            prompt=query,
            scenario_id=7,
            transport=runner.LOCAL,
            arm=runner.ARM_BASE,
            temperature=0.0,
            timeout=30,
            layers_dir=None,
        )

    early, late = await asyncio.gather(_row("ранний", 0), _row("поздний", 1))

    assert early.restriction_plan["mode"] == "restrictions"
    assert late.restriction_plan["mode"] == "buffers_only"


async def test_a_row_that_never_built_a_plan_records_none():
    """A failure before planning must not inherit the previous row's plan."""

    service = FakeService([], error=RuntimeError("died early"))

    record = await runner.run_one(
        service,
        client=None,
        idx=0,
        model="m",
        prompt="q",
        scenario_id=7,
        transport=runner.LOCAL,
        arm=runner.ARM_BASE,
        temperature=0.0,
        timeout=30,
        layers_dir=None,
    )

    assert record.restriction_plan is None
