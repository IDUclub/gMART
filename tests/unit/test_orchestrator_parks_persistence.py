"""Parks geometry, economy and full analysis survive orchestrator ChatStorage."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import StructuredPartRequest
from src.agents.services.service_entities.orchestrator_plan import (
    OrchestratorAgent,
    OrchestratorPlan,
    OrchestratorPlanMode,
    OrchestratorStep,
)
from tests.unit.test_orchestrator_service_events import (
    FakePipeline,
    orchestrator,  # noqa: F401 -- reuse the established offline service fixture
    run_pipeline,
)


GEOMETRY = {
    "type": "FeatureCollection",
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
        "properties": {"name": "Парк", "area": 123.4},
    }],
}
ECONOMY = {"area": 123.4, "cost": {"total": 123456}, "objects": ["tree"]}
ANALYSIS = "Полный анализ парка. " * 200 + "Конец рекомендаций."


def park_events(economy=ECONOMY, text=ANALYSIS):
    events = [{
        "type": "feature_collection",
        "content": {
            "name": "Сгенерированный парк",
            "feature_collection": deepcopy(GEOMETRY),
            "task_id": "park-task",
            "event_id": 1,
        },
    }]
    if economy:
        events.append({
            "type": "status",
            "content": {
                "status": "economy", "economy": deepcopy(economy),
                "text": "Получены ТЭП парка.", "event_id": 2,
            },
        })
    events.extend([
        {"type": "chunk", "content": {"text": text, "event_id": 3}},
        {"type": "chunk", "content": {"text": "", "done": True, "event_id": 4}},
    ])
    return events


@asynccontextmanager
async def execution_lock(key):
    yield


@pytest.fixture
def parks(orchestrator):
    service = SimpleNamespace(
        run_parks_pipeline=FakePipeline(park_events()),
        pending_dialogue=AsyncMock(return_value=None),
        get_run=AsyncMock(return_value={"model": "m", "temperature": 0.5}),
        store=SimpleNamespace(execution_lock=execution_lock),
    )
    orchestrator.parks_service = service
    orchestrator.plan_builder.build_plan = AsyncMock(return_value=OrchestratorPlan(
        mode=OrchestratorPlanMode.EXECUTE,
        steps=[OrchestratorStep(agent=OrchestratorAgent.PARKS, task="Построй парк")],
    ))
    return service


def stored_parts(orchestrator):
    orchestrator.add_complex_message.assert_awaited_once()
    return orchestrator.add_complex_message.await_args.args[3]


def data_payloads(parts):
    return [part.model_dump()["payload"] for part in parts if part.kind == "data"]


@pytest.mark.asyncio
@pytest.mark.parametrize("economy", [ECONOMY, None])
async def test_normal_parks_answer_matches_direct_history(
    orchestrator, parks, economy
):
    from src.agents.services.parks.parks_service import ParksService

    source = park_events(economy)
    before = deepcopy(source)
    parks.run_parks_pipeline = FakePipeline(source)
    events = await run_pipeline(orchestrator)
    await asyncio.sleep(0)
    parts = stored_parts(orchestrator)
    assert all(isinstance(part, StructuredPartRequest) for part in parts if part.kind == "data")
    assert parks.run_parks_pipeline.calls[0]["persist_history"] is False
    assert source == before  # transport event IDs are removed only from stored copies
    assert ANALYSIS in parts[0].payload.text
    summary = next(e["content"]["summary"] for e in events if e["type"] == "step_finished")
    assert len(summary) <= orchestrator.DIGEST_MAX_CHARS
    assert summary.endswith("…")

    direct = SimpleNamespace(
        add_complex_message=AsyncMock(), store=SimpleNamespace(save=AsyncMock())
    )
    await ParksService._history(direct, {
        "persist_history": True, "chat_id": "chat-xyz", "result": GEOMETRY,
        "economy": economy, "task_id": "park-task", "scenario_id": 772,
        "request_id": "park-run",
    }, "tok", RoleEnum.ASSISTANT, ANALYSIS, history_key="result")
    expected = data_payloads(direct.add_complex_message.await_args.args[3])
    assert data_payloads(parts) == expected

    # Finished reconnects only replay; they never persist another assistant turn.
    request_id = events[0]["content"]["request_id"]
    await run_pipeline(orchestrator, request_id=request_id)
    await asyncio.sleep(0)
    orchestrator.add_complex_message.assert_awaited_once()
    assert len(parks.run_parks_pipeline.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["error", "pipeline_failed", "pipeline_suspended", "clarification"])
async def test_unconfirmed_park_outputs_are_not_saved(orchestrator, parks, terminal):
    parks.run_parks_pipeline = FakePipeline([
        *park_events(),
        {"type": terminal, "content": {"question": "Какой парк?", "message": "stop"}},
    ])
    await run_pipeline(orchestrator)
    await asyncio.sleep(0)
    parts = stored_parts(orchestrator)
    assert not data_payloads(parts)
    assert ANALYSIS not in parts[0].payload.text


@pytest.mark.asyncio
async def test_normal_opt_out_never_persists_parks(orchestrator, parks):
    await run_pipeline(orchestrator, chat_id="chat-xyz", persist_history=False)
    await asyncio.sleep(0)
    orchestrator.add_complex_message.assert_not_awaited()


async def seed_resume(orchestrator, state_store, *, earlier_park=False, buffered_count=3):
    request_id = state_store.new_request_id()
    await state_store.create(
        request_id, chat_id="chat-xyz", user_query="Построй парк",
        scenario_id=772, model="m", temperature=0.5,
    )
    plan = OrchestratorPlan(
        mode=OrchestratorPlanMode.EXECUTE,
        steps=[OrchestratorStep(agent=OrchestratorAgent.PARKS, task="Построй парк")
               for _ in range(2 if earlier_park else 1)],
    )
    buffered = [orchestrator._plan_event(plan)]
    number = len(plan.steps)
    if earlier_park:
        step = plan.steps[0]
        buffered.append(orchestrator._step_started_event(1, step, "old-park", step.task))
        buffered.extend(orchestrator._step_event(1, step, event) for event in park_events(text="Первый парк"))
        buffered.append(orchestrator._step_finished_event(1, step, "completed", "Кратко"))
    step = plan.steps[-1]
    parked = orchestrator._step_started_event(number, step, "park-run", step.task)
    buffered.append(parked)
    # Disconnect at different points in geometry/economy/analysis delivery.
    buffered.extend(orchestrator._step_event(number, step, event)
                    for event in park_events(text="Начало. ")[:buffered_count])
    for event in buffered:
        await state_store.buffer_event(request_id, event)
    return request_id, buffered, parked["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("earlier_park", [False, True])
@pytest.mark.parametrize("buffered_count", [1, 2, 3])
async def test_resume_restores_buffered_geometry_economy_and_full_text_once(
    orchestrator, parks, state_store, earlier_park, buffered_count
):
    request_id, buffered, parked = await seed_resume(
        orchestrator, state_store, earlier_park=earlier_park,
        buffered_count=buffered_count,
    )
    # Deliberately include an overlapping durable event to exercise deduplication.
    parks.run_parks_pipeline = FakePipeline([
        *park_events(text="Начало. ")[buffered_count - 1:3],
        {"type": "chunk", "content": {"text": ANALYSIS, "event_id": 4}},
        {"type": "chunk", "content": {"text": "", "done": True, "event_id": 5}},
    ])
    await run_pipeline(orchestrator, request_id=request_id)
    await asyncio.sleep(0)
    parts = stored_parts(orchestrator)
    payloads = data_payloads(parts)
    assert len(payloads) == (4 if earlier_park else 2)
    assert [p["type"] for p in payloads] == ["feature_collection", "parks_economy"] * (2 if earlier_park else 1)
    assert payloads[-2]["content"]["feature_collection"] == GEOMETRY
    assert payloads[-1]["content"] == ECONOMY
    text = "\n".join(p.payload.text for p in parts if p.kind == "text")
    assert "Начало. " + ANALYSIS in text
    assert text.count("Начало.") == 1
    if earlier_park:
        assert "Первый парк" in text
    assert parks.run_parks_pipeline.calls[0]["after"] == buffered_count
    assert parks.run_parks_pipeline.calls[0]["initial_request_id"] == "park-run"
    assert parks.run_parks_pipeline.calls[0]["persist_history"] is False

    await run_pipeline(orchestrator, request_id=request_id)
    await asyncio.sleep(0)
    orchestrator.add_complex_message.assert_awaited_once()
    # Even a reconnect with a stale pre-lock snapshot cannot save the turn twice.
    async for _ in orchestrator._resume_parks_orchestration(
        request_id, "tok", buffered, parked, {}
    ):
        pass
    await asyncio.sleep(0)
    orchestrator.add_complex_message.assert_awaited_once()
    assert len(parks.run_parks_pipeline.calls) == 1


@pytest.mark.asyncio
async def test_resume_respects_persistence_opt_out(orchestrator, parks, state_store):
    request_id, _, _ = await seed_resume(orchestrator, state_store)
    parks.run_parks_pipeline = FakePipeline([])
    await run_pipeline(orchestrator, request_id=request_id, persist_history=False)
    await asyncio.sleep(0)
    orchestrator.add_complex_message.assert_not_awaited()
