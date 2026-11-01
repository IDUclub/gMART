"""The LLM benchmark harness (benchmarks/): checks, case files and a run on a fake model."""

from __future__ import annotations

import json

import pytest

from benchmarks.harness.cases import load_cases
from benchmarks.harness.checks import evaluate_check, resolve
from benchmarks.harness.config import (
    BenchConfig,
    EndpointConfig,
    ModelConfig,
    ProfileConfig,
)
from benchmarks.harness.report import render_review, summarize
from benchmarks.harness.runner import Runner
from benchmarks.harness.stage import load_stages
from src.agents.model_clients.llm_base import (
    BaseLlmAdapter,
    LlmChatResponse,
    LlmMessage,
)
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter


class FakeAdapter(BaseLlmAdapter):
    """Answers each request kind with a canned reply and remembers what it got."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.requests: list[dict] = []

    async def chat(
        self, model, messages=None, *, stream=False, request_kind=None, **kwargs
    ):
        self.requests.append({"model": model, "request_kind": request_kind, **kwargs})
        content = self.replies[request_kind]
        if not stream:
            return LlmChatResponse(model=model, message=LlmMessage(content=content))

        async def parts():
            for piece in (content[:5], content[5:]):
                yield LlmChatResponse(
                    model=model, message=LlmMessage(content=piece), done=False
                )

        return parts()

    async def generate(
        self, model, prompt, *, stream=False, request_kind=None, **kwargs
    ):
        raise NotImplementedError

    async def list(self):
        return {"models": [{"model": "fake"}]}

    async def ps(self):
        return {"models": []}


def _config() -> BenchConfig:
    return BenchConfig(
        endpoints={"fake": EndpointConfig(name="fake")},
        models={"m": ModelConfig(name="m", endpoint="fake", model="fake-model")},
        profiles={"p": ProfileConfig(name="p", default="m")},
    )


def _runner(adapter: FakeAdapter) -> Runner:
    runner = Runner(_config(), load_stages())
    runner._adapters["fake"] = adapter
    return runner


class TestChecks:
    OUTPUT = {
        "mode": "execute",
        "steps": [
            {"agent": "norms", "task": "Найти ограничения для школ"},
            {"agent": "documents", "task": "Найти документы"},
        ],
        "clarification_question": None,
        "by_topic": {"жилой дом": ["Жилой дом"]},
    }

    def test_wildcard_collects_field_across_list(self):
        assert resolve(self.OUTPUT, "steps[*].agent") == ["norms", "documents"]
        assert resolve(self.OUTPUT, "steps.1.agent") == "documents"
        assert resolve(self.OUTPUT, "by_topic.жилой дом") == ["Жилой дом"]

    @pytest.mark.parametrize(
        ("check", "passed"),
        [
            ({"path": "mode", "eq": "EXECUTE"}, True),
            ({"path": "steps[*].agent", "eq": ["documents", "norms"]}, False),
            ({"path": "steps[*].agent", "set_eq": ["documents", "norms"]}, True),
            ({"path": "steps.0.task", "contains": ["школ"]}, True),
            ({"path": "steps[*].agent", "not_contains": ["compliance"]}, True),
            ({"path": "clarification_question", "null": True}, True),
            ({"path": "missing.path", "null": True}, True),
            ({"path": "missing.path", "eq": None}, False),
            ({"path": "steps", "len": 2}, True),
            ({"path": "by_topic.жилой дом", "contains": ["жилой дом"]}, True),
        ],
    )
    def test_operators(self, check, passed):
        assert evaluate_check(self.OUTPUT, check).passed is passed

    def test_number_comparison_ignores_int_float(self):
        assert evaluate_check({"n": 30000}, {"path": "n", "eq": 30000.0}).passed
        assert evaluate_check({"n": 0.5}, {"path": "n", "lte": 0.6}).passed
        assert not evaluate_check({"n": None}, {"path": "n", "lte": 0.6}).passed


def test_every_case_file_loads_and_targets_a_registered_stage():
    stages = load_stages()
    cases = load_cases()

    assert cases
    assert {case.stage for case in cases} <= set(stages)
    # Every auto stage is covered by at least one case.
    assert {s for s, st in stages.items() if st.kind == "auto"} <= {
        c.stage for c in cases
    }


def test_recording_proxy_keeps_isinstance_for_reasoning_effort_choices():
    from benchmarks.harness.llm import RecordingLlm

    adapter = OpenAiCompatAdapter(base_url="http://localhost:1/v1")
    llm = RecordingLlm(
        adapter, ModelConfig(name="m", endpoint="e", model="gpt-oss-20b")
    )

    assert isinstance(llm, OpenAiCompatAdapter)


async def test_structured_stage_is_scored_and_uses_profile_model(tmp_path):
    plan = {
        "mode": "execute",
        "steps": [{"agent": "provision", "task": "Обеспеченность школами"}],
        "clarification_question": None,
    }
    adapter = FakeAdapter({"orchestrator.plan": json.dumps(plan, ensure_ascii=False)})
    case = next(
        c
        for c in load_cases()
        if c.key == "orchestrator.plan/provision_current_single_service"
    )

    results = await _runner(adapter).run(
        [case], [_config().profiles["p"]], tmp_path / "r.jsonl"
    )

    [result] = results
    assert result["status"] == "pass", result
    assert result["output"]["final"]["steps"][0]["agent"] == "provision"
    assert result["llm_calls"] == 1
    assert adapter.requests[0]["model"] == "fake-model"
    assert (tmp_path / "r.jsonl").read_text(encoding="utf-8").count("\n") == 1


async def test_wrong_plan_fails_with_the_failed_check(tmp_path):
    plan = {"mode": "execute", "steps": [{"agent": "scenario_data", "task": "x"}]}
    adapter = FakeAdapter({"orchestrator.plan": json.dumps(plan)})
    case = next(
        c
        for c in load_cases()
        if c.key == "orchestrator.plan/provision_current_single_service"
    )

    [result] = await _runner(adapter).run(
        [case], [_config().profiles["p"]], tmp_path / "r.jsonl"
    )

    assert result["status"] == "fail"
    failed = [c for c in result["checks"] if not c["passed"]]
    assert failed and failed[0]["actual"] == ["scenario_data"]


async def test_invalid_json_after_retries_is_an_error(tmp_path):
    adapter = FakeAdapter({"provision.plan": "not json"})
    case = next(
        c for c in load_cases() if c.key == "provision.plan/current_provision_single"
    )

    [result] = await _runner(adapter).run(
        [case], [_config().profiles["p"]], tmp_path / "r.jsonl"
    )

    assert result["status"] == "error"
    assert result["llm_calls"] == 3  # the first try and two repair retries


async def test_streamed_answer_is_recorded_for_manual_review(tmp_path):
    text = "Не менее 50 м [1] (СП 42.13330.2016, п. 10.4) и 10 м [2]."
    adapter = FakeAdapter({"norms.answer": text})
    case = next(
        c for c in load_cases() if c.key == "norms.answer/placement_with_two_sources"
    )

    results = await _runner(adapter).run(
        [case], [_config().profiles["p"]], tmp_path / "r.jsonl"
    )

    [result] = results
    assert result["output"]["text"] == text
    assert result["needs_review"] is True
    assert "[1] СП 42.13330.2016" in result["output"]["context"]
    assert text in render_review(results)
    assert summarize(results)["cells"][0]["kind"] == "manual"
