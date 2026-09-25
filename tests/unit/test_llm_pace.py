"""Deadlines follow the measured speed of the shared LLM server.

On 2026-09-25 the prod vLLM answered at ~13 tok/s instead of ~130, and scenario-data
runs stopped at a fixed 5 minutes while still making progress. These tests pin the
slowdown estimate, the stretched deadline and its cap.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.agents.model_clients import llm_pace as pace_module
from src.agents.model_clients import openai_adapter
from src.agents.model_clients.llm_pace import (
    LlmPace,
    PacedDeadline,
    PipelineDeadlineExceeded,
)
from tests.unit.test_llm_adapters import (
    _adapter_with,
    _Choice,
    _Completion,
    _Delta,
    _FakeStream,
)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _pace(clock, max_factor=4.0):
    return LlmPace(100.0, max_factor, clock=clock)


def _call(pace, clock, seconds, completion_tokens, prompt_tokens=0, max_tokens=None):
    call_id = pace.started(max_tokens)
    clock.now += seconds
    pace.finished(
        call_id, completion_tokens=completion_tokens, prompt_tokens=prompt_tokens
    )


def test_no_calls_mean_nominal_speed():
    assert _pace(Clock()).slowdown() == 1.0


def test_calls_at_the_nominal_speed_do_not_stretch_anything():
    clock = Clock()
    pace = _pace(clock)
    # 1 s latency + 5000/5000 s prefill + 800/100 s decode = 10 s.
    _call(pace, clock, 10, 800, prompt_tokens=5000)
    assert pace.slowdown() == pytest.approx(1.0)


def test_slow_calls_raise_the_factor_up_to_the_cap():
    clock = Clock()
    pace = _pace(clock, max_factor=10)
    _call(pace, clock, 30, 200)  # expected 3 s
    assert pace.slowdown() == pytest.approx(10.0)
    pace = _pace(clock, max_factor=4)
    _call(pace, clock, 30, 200)
    assert pace.slowdown() == 4.0


def test_a_call_still_running_past_its_whole_budget_counts_as_slowness():
    # The prod type-mapping call: max_tokens=1800, no answer after 135 s.
    clock = Clock()
    pace = _pace(clock, max_factor=10)
    pace.started(1800)
    clock.now += 135
    assert pace.slowdown() == pytest.approx(135 / (1 + 10 + 18))


def test_old_measurements_expire():
    clock = Clock()
    pace = _pace(clock)
    _call(pace, clock, 30, 200)
    clock.now += pace_module.WINDOW_SECONDS + 1
    assert pace.slowdown() == 1.0


def test_an_abandoned_call_stops_counting_after_the_window():
    clock = Clock()
    pace = _pace(clock)
    pace.started(100)
    clock.now += pace_module.WINDOW_SECONDS + 1
    assert pace.slowdown() == 1.0
    assert not pace._in_flight


def test_a_call_without_token_counts_is_not_a_sample():
    clock = Clock()
    pace = _pace(clock)
    _call(pace, clock, 30, None)
    call_id = pace.started()
    clock.now += 30
    pace.finished(call_id, completion_tokens=SimpleNamespace())
    assert pace.slowdown() == 1.0


def test_deadline_matches_wall_time_on_a_normal_server():
    clock = Clock()
    deadline = PacedDeadline(300, pace=_pace(clock), clock=clock)
    clock.now += 299
    assert not deadline.exceeded()
    clock.now += 1
    assert deadline.exceeded()


def test_deadline_stretches_while_the_server_is_slow():
    clock = Clock()
    pace = _pace(clock)
    deadline = PacedDeadline(300, pace=pace, clock=clock)
    for _ in range(19):
        _call(pace, clock, 60, 200)  # 20× slower, capped at 4×
        assert not deadline.exceeded()
    _call(pace, clock, 60, 200)
    assert deadline.exceeded()
    assert deadline.wall_seconds == pytest.approx(20 * 60)


def test_only_the_slow_spell_is_stretched():
    clock = Clock()
    pace = _pace(clock)
    deadline = PacedDeadline(300, pace=pace, clock=clock)
    _call(pace, clock, 30, 200)  # 10× slower, capped at 4×
    deadline.advance()
    assert deadline.consumed == pytest.approx(30 / 4)
    clock.now += pace_module.WINDOW_SECONDS  # the slow sample ages out
    deadline.advance()
    clock.now += 100
    assert deadline.advance() == pytest.approx(30 / 4 + 600 / 4 + 100)


def test_paused_time_is_not_consumed():
    clock = Clock()
    deadline = PacedDeadline(300, pace=_pace(clock), clock=clock)
    clock.now += 100
    assert deadline.advance(paused_seconds=60) == pytest.approx(40)
    clock.now += 10
    assert deadline.advance(paused_seconds=60) == pytest.approx(50)


def test_the_user_is_told_how_long_the_run_took():
    message = PipelineDeadlineExceeded("x", 17 * 60).user_message
    assert "около 17 мин" in message
    assert "загрузки языковой модели" in message


def test_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LLM_NOMINAL_TOKENS_PER_SECOND", "50")
    monkeypatch.setenv("LLM_DEADLINE_MAX_FACTOR", "2.5")
    pace = LlmPace.from_env()
    assert (pace.nominal_tokens_per_second, pace.max_factor) == (50.0, 2.5)
    monkeypatch.setenv("LLM_DEADLINE_MAX_FACTOR", "0.5")
    with pytest.raises(ValueError, match="LLM_DEADLINE_MAX_FACTOR"):
        LlmPace.from_env()


# --------------------------------------------------------------------------- #
# the adapter feeds the estimate
# --------------------------------------------------------------------------- #
@pytest.fixture
def fresh_pace(monkeypatch):
    clock = Clock()
    pace = _pace(clock, max_factor=100)
    monkeypatch.setattr(openai_adapter, "llm_pace", pace)
    return pace, clock


async def test_openai_completions_record_their_usage(fresh_pace):
    pace, clock = fresh_pace
    completion = _Completion([_Choice(message=_Delta("ok"), finish_reason="stop")])
    completion.usage = SimpleNamespace(completion_tokens=200, prompt_tokens=0)
    adapter, completions = _adapter_with(completion)
    create = completions.create

    async def slow_create(**kwargs):
        clock.now += 30
        return await create(**kwargs)

    completions.create = slow_create
    await adapter.chat("m", [], options={"num_predict": 300})
    assert pace.slowdown() == pytest.approx(10.0)
    assert not pace._in_flight


async def test_openai_streams_count_their_chunks(fresh_pace):
    pace, clock = fresh_pace

    class SlowStream(_FakeStream):
        def __aiter__(self):
            async def gen():
                for chunk in self._chunks:
                    clock.now += 1
                    yield chunk

            return gen()

    chunks = [_Completion([_Choice(delta=_Delta("т"))]) for _ in range(9)]
    chunks.append(_Completion([_Choice(delta=_Delta("т"), finish_reason="stop")]))
    adapter, _ = _adapter_with(SlowStream(chunks))
    parts = [p async for p in await adapter.chat("m", [], stream=True)]
    assert len(parts) == 10
    # 10 tokens in 10 s against 1 s + 0.1 s nominal.
    assert pace.slowdown() == pytest.approx(10 / 1.1)


async def test_a_failed_completion_leaves_no_trace(fresh_pace):
    pace, _ = fresh_pace
    adapter, completions = _adapter_with(None)

    async def failing(**kwargs):
        raise RuntimeError("down")

    completions.create = failing
    with pytest.raises(RuntimeError):
        await adapter.chat("m", [])
    assert not pace._in_flight and not pace._samples


# --------------------------------------------------------------------------- #
# long runs keep their Redis state, deadlines surface clearly
# --------------------------------------------------------------------------- #
async def test_keep_alive_restarts_the_ttl_of_a_running_pipeline(state_store):
    from src.agents.services.pipeline_state import PIPELINE_TTL

    redis = state_store._redis
    await state_store.create(
        "r1", chat_id="c1", user_query="q", scenario_id=1, model="m", temperature=0
    )
    await state_store.buffer_event("r1", {"type": "status"})
    assert await state_store.acquire_chat("c1", "r1")
    await state_store.acquire_chat("c2", "other")
    keys = [state_store._key("r1", kind) for kind in ("state", "events")]
    locks = [state_store._key(chat, "active_request") for chat in ("c1", "c2")]
    for key in keys + locks:
        await redis.expire(key, 5)

    await state_store.keep_alive("r1", chat_id="c1")
    assert all([await redis.ttl(key) > 5 for key in keys + locks[:1]])

    await state_store.keep_alive("r1", chat_id="c2")
    assert await redis.ttl(locks[1]) <= 5  # another request's lock is not touched
    assert await redis.ttl(keys[0]) <= PIPELINE_TTL


async def test_sse_reports_a_deadline_without_rerunning():
    from src.agents.common.executors.sse_executors import stream_with_error_handling
    from tests.unit.test_sse_executors import FakeRequest, ForbiddenErrorExplainer

    runs = []

    async def slow_pipeline(**kwargs):
        runs.append(kwargs)
        if False:
            yield {}
        raise PipelineDeadlineExceeded("deadline", 20 * 60)

    events = [
        event
        async for event in stream_with_error_handling(
            slow_pipeline,
            FakeRequest(),
            ForbiddenErrorExplainer(),
            "model",
            rerun=True,
        )
    ]
    assert len(runs) == 1
    assert "около 20 мин" in events[0]["content"]["text"]
    assert events[1] == {
        "type": "error",
        "content": {"message": "Pipeline deadline exceeded", "traceback": ""},
    }


async def test_document_run_explains_its_deadline(monkeypatch, service, fake_mcp):
    from src.agents.services.dvd import runs

    async def stuck(*args, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(runs, "RUN_DEADLINE_SECONDS", 0.3)
    service.llm_client.chat = stuck
    events = [
        e
        async for e in runs.stream_document_run(
            service,
            model="m",
            dvd_mcp_client=fake_mcp,
            token=None,
            user_query="вопрос",
            temperature=0,
            persist_history=False,
        )
    ]
    assert events[-1]["type"] == "error"
    assert "не уложился в отведённое время" in events[-1]["content"]["message"]


async def test_scenario_data_keeps_waiting_for_a_slow_server(monkeypatch, state_store):
    from src.agents.services.scenario_data import scenario_data_linear as linear

    clock = Clock()
    pace = _pace(clock)
    monkeypatch.setattr(
        linear,
        "PacedDeadline",
        lambda limit: PacedDeadline(limit, pace=pace, clock=clock),
    )
    await state_store.create(
        "r1", chat_id=None, user_query="q", scenario_id=1, model="m", temperature=0
    )
    workflow = linear.ScenarioDataLinearWorkflow.__new__(
        linear.ScenarioDataLinearWorkflow
    )
    workflow.owner = SimpleNamespace(state_store=state_store)
    deadline = linear._WorkflowDeadline()
    _call(pace, clock, 60, 200)  # 20× slower, capped at 4×
    # Six minutes of wall time is past the old fixed 5-minute limit.
    clock.now += 5 * 60
    assert not await workflow._deadline_exceeded("r1", deadline)
    clock.now += 15 * 60
    assert await workflow._deadline_exceeded("r1", deadline)
    assert isinstance(deadline.error(), PipelineDeadlineExceeded)


async def test_paced_wait_for_returns_or_times_out(monkeypatch):
    from src.agents.model_clients.llm_pace import paced_wait_for

    async def answer():
        return 42

    assert await paced_wait_for(answer(), 5) == 42
    with pytest.raises(TimeoutError):
        await paced_wait_for(asyncio.sleep(5), 0.05, poll_seconds=0.01)
