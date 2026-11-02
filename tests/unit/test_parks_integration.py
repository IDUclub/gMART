import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.agents.api_clients.parks_api_client import ParksApiClient
from src.agents.dto.parks_request_dto import ParksInputs
from src.agents.services.parks.parks_a2a_service import ParksA2AService
from src.agents.services.parks.parks_service import ParksService
from src.common.service_auth import internal_user_context_jwt

TOKEN = internal_user_context_jwt("park-user")
BOUNDARY = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"type": "boundary"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[30, 60], [30.01, 60], [30.01, 60.01], [30, 60]]],
            },
        }
    ],
}
RESULT = {"type": "FeatureCollection", "features": []}


def llm_response(value):
    return SimpleNamespace(message=SimpleNamespace(content=value))


@pytest.fixture
def parks(monkeypatch, state_store):
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=llm_response('{"tags":["CHILDREN","GREENERY"]}'))
    )
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *args, **kwargs: llm,
    )
    api = SimpleNamespace(
        generate=AsyncMock(return_value="task-1"),
        status=AsyncMock(return_value={"status": 1}),
        result=AsyncMock(return_value=RESULT),
        economy=AsyncMock(return_value={"economyParams": []}),
    )
    service = ParksService("http://localhost", Mock(), Mock(), state_store, api)
    service.resolve_model = AsyncMock(return_value="vision")
    service.create_chat = AsyncMock(return_value=("chat-1", "Парк"))
    service.add_single_message = AsyncMock()
    service.add_complex_message = AsyncMock()
    return service


async def collect(service, **kwargs):
    return [
        event
        async for event in service.run_parks_pipeline(
            token=TOKEN,
            model="vision",
            temperature=0,
            user_query=kwargs.pop("user_query", "Создай парк"),
            conversation_key="conversation",
            persist_history=False,
            **kwargs,
        )
    ]


def clarification(events):
    return next(
        event["content"] for event in events if event["type"] == "clarification"
    )


@pytest.mark.asyncio
async def test_confirmation_edits_generation_order_and_replay(parks):
    events = await collect(parks, inputs={"park_geojson": BOUNDARY})
    assert clarification(events)["revision"] == 1
    parks.api.generate.assert_not_called()
    parks.llm_client.chat.return_value = llm_response('{"tags":["REST"]}')
    revised = await collect(parks, user_query="Убери детскую площадку, добавь отдых")
    assert clarification(revised)["revision"] == 2
    stale = await collect(
        parks, user_query="Подтверждаю", inputs={"confirmed_revision": 1}
    )
    assert clarification(stale)["revision"] == 2
    parks.api.generate.assert_not_called()
    parks.llm_client.chat.return_value = llm_response("Анализ и рекомендации")
    generated = await collect(
        parks, user_query="Подтверждаю", inputs={"confirmed_revision": 2}
    )
    parks.api.generate.assert_awaited_once()
    assert parks.api.generate.call_args.args[0]["tags"] == ["REST"]
    result_index = next(
        i for i, event in enumerate(generated) if event["type"] == "feature_collection"
    )
    analysis_index = next(
        i
        for i, event in enumerate(generated)
        if event["content"].get("text") == "Анализ и рекомендации"
    )
    assert result_index < analysis_index
    request_id = generated[0]["content"]["request_id"]
    assert generated == await collect(parks, request_id=request_id)
    assert (
        await collect(
            parks, request_id=request_id, after=generated[-1]["content"]["event_id"]
        )
        == []
    )
    parks.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_geometry_never_generates(parks):
    events = await collect(parks, user_query="Да, создай парк")
    assert clarification(events)["ready"] is False
    parks.api.generate.assert_not_called()


@pytest.mark.asyncio
async def test_ambiguous_confirmation_and_edits_do_not_generate(parks):
    await collect(parks, inputs={"park_geojson": BOUNDARY})
    events = await collect(parks, user_query="Да, но добавь спорт")
    assert clarification(events)["revision"] == 2
    parks.api.generate.assert_not_called()


@pytest.mark.asyncio
async def test_foreign_and_unknown_request_ids(parks):
    events = await collect(parks)
    request_id = events[0]["content"]["request_id"]
    with pytest.raises(LookupError):
        await parks.get_run(request_id, internal_user_context_jwt("other-user"))
    with pytest.raises(LookupError):
        await collect(parks, request_id="unknown")


@pytest.mark.asyncio
async def test_lost_submission_response_never_resubmits(parks):
    await collect(parks, inputs={"park_geojson": BOUNDARY})
    parks.api.generate.side_effect = TimeoutError()
    events = await collect(parks, user_query="Подтверждаю")
    request_id = events[0]["content"]["request_id"]
    assert (await parks.get_run(request_id, TOKEN))["status"] == "submission_unknown"
    await collect(parks, request_id=request_id)
    await collect(parks, user_query="Подтверждаю")
    parks.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_analysis_retry_keeps_generation_and_delivered_result(parks):
    await collect(parks, inputs={"park_geojson": BOUNDARY})
    parks.llm_client.chat.side_effect = RuntimeError("model down")
    events = await collect(parks, user_query="Подтверждаю")
    assert any(event["type"] == "feature_collection" for event in events)
    request_id = events[0]["content"]["request_id"]
    parks.llm_client.chat.side_effect = None
    parks.llm_client.chat.return_value = llm_response("Восстановленный анализ")
    retry = await collect(parks, request_id=request_id)
    assert retry[-1]["content"]["done"] is True
    parks.api.generate.assert_awaited_once()
    parks.api.result.assert_awaited_once()


@pytest.mark.asyncio
async def test_disconnect_worker_and_heartbeat(parks):
    blocker = asyncio.Event()
    original = parks.llm_client.chat.return_value

    async def slow(*args, **kwargs):
        await blocker.wait()
        return original

    parks.llm_client.chat.side_effect = slow
    parks.HEARTBEAT_SECONDS = 0
    stream = parks.run_parks_pipeline(
        token=TOKEN,
        model="vision",
        temperature=0,
        user_query="Создай парк",
        persist_history=False,
        conversation_key="conversation",
    )
    first = await anext(stream)
    heartbeat = await anext(stream)
    assert heartbeat["content"]["status"] == "heartbeat"
    request_id = first["content"]["request_id"]
    await stream.aclose()
    blocker.set()
    replay = await collect(parks, request_id=request_id)
    assert any(event["type"] == "clarification" for event in replay)


@pytest.mark.asyncio
async def test_chatstorage_question_and_result(parks):
    awaitable = parks.run_parks_pipeline(
        token=TOKEN,
        model="vision",
        temperature=0,
        user_query="Создай парк",
        inputs={"park_geojson": BOUNDARY},
    )
    events = [event async for event in awaitable]
    assert any(event["type"] == "service_event" for event in events)
    assert parks.add_single_message.call_args.args[2].value == "assistant"
    parks.llm_client.chat.return_value = llm_response("Рекомендации")
    result = [
        event
        async for event in parks.run_parks_pipeline(
            token=TOKEN,
            model="vision",
            temperature=0,
            user_query="Подтверждаю",
            chat_id="chat-1",
        )
    ]
    assert result[-1]["content"]["done"] is True
    parts = parks.add_complex_message.call_args.args[3]
    assert parts[1].payload["content"]["task_id"] == "task-1"
    assert parts[1].payload["content"]["feature_collection"] == RESULT


@pytest.mark.asyncio
async def test_generated_layer_persists_when_vision_is_unavailable(parks):
    await collect(parks, inputs={"park_geojson": BOUNDARY})
    parks.llm_client.chat.side_effect = RuntimeError("vision unavailable")
    events = [
        event
        async for event in parks.run_parks_pipeline(
            token=TOKEN,
            model="vision",
            temperature=0,
            user_query="Подтверждаю",
            conversation_key="conversation",
            chat_id="chat-1",
        )
    ]
    request_id = events[0]["content"]["request_id"]
    assert any(event["type"] == "error" for event in events)
    state = await parks.get_run(request_id, TOKEN)
    assert state["history_result"] is True
    assert not state.get("history_assistant")
    parks.add_complex_message.assert_awaited_once()
    assert (
        parks.add_complex_message.call_args.args[3][1].payload["content"][
            "feature_collection"
        ]
        == RESULT
    )
    parks.llm_client.chat.side_effect = None
    parks.llm_client.chat.return_value = llm_response("Восстановленные рекомендации")
    await collect(parks, request_id=request_id)
    parks.add_complex_message.assert_awaited_once()
    assert parks.add_single_message.call_args.args[3].endswith(
        "Восстановленные рекомендации"
    )
    parks.api.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_a2a_context_confirmation_and_task_ownership(parks):
    a2a = ParksA2AService(parks)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "contextId": "park-context",
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Создай парк"}],
            },
            "metadata": {"inputs": {"park_geojson": BOUNDARY}},
        },
    }
    first = await a2a.handle_json_rpc(payload, TOKEN)
    assert first["result"]["status"]["state"] == "input-required"
    assert first["result"]["contextId"] == "park-context"
    assert first["result"]["status"]["message"]["parts"][1]["data"]["revision"] == 1
    task_id = first["result"]["id"]
    other = await a2a.handle_json_rpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task_id}},
        internal_user_context_jwt("other"),
    )
    assert other["error"]["code"] == -32001
    payload["params"]["message"]["parts"][0]["text"] = "Подтверждаю"
    payload["params"].pop("metadata")
    parks.llm_client.chat.return_value = llm_response("Рекомендации")
    second = await a2a.handle_json_rpc(payload, TOKEN)
    assert second["result"]["status"]["state"] == "completed"
    parks.api.generate.assert_awaited_once()
    parks.create_chat.assert_not_called()


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "FeatureCollection", "features": []},
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"type": "boundary"},
                    "geometry": {"type": "Point", "coordinates": [30, 60]},
                }
            ],
        },
    ],
)
def test_geometry_validation(geometry):
    with pytest.raises(ValueError):
        ParksInputs(park_geojson=geometry)


@pytest.mark.asyncio
async def test_upstream_wire_contract():
    def handler(request):
        if request.url.path == "/map/generate":
            body = json.loads(request.content)
            assert json.loads(body["parkGeoJson"]) == BOUNDARY
            assert body["tags"] == ["REST", "REST"]
            assert body["locationRequests"] == []
            return httpx.Response(200, json={"code": 0, "taskId": "task"})
        assert request.url.params["taskId"] == "task"
        return httpx.Response(200, json={"code": 0, "status": 1})

    api = ParksApiClient("https://parks.test", transport=httpx.MockTransport(handler))
    assert (
        await api.generate({"park_geojson": BOUNDARY, "tags": ["REST", "REST"]})
        == "task"
    )
    assert (await api.status("task"))["status"] == 1


@pytest.mark.asyncio
async def test_concurrent_confirmation_uses_one_external_task(parks):
    await collect(parks, inputs={"park_geojson": BOUNDARY})
    parks.llm_client.chat.return_value = llm_response("Анализ")
    replies = await asyncio.gather(
        collect(parks, user_query="Подтверждаю"),
        collect(parks, user_query="Подтверждаю"),
    )
    parks.api.generate.assert_awaited_once()
    assert any(
        any(event["type"] == "feature_collection" for event in reply)
        for reply in replies
    )


@pytest.mark.asyncio
async def test_recover_generation_in_new_service_instance(parks):
    # Redis contains a task checkpoint from a process that has stopped.
    state = dict(
        request_id="resumed",
        owner="park-user",
        status="running",
        phase="generating",
        model="vision",
        temperature=0,
        query="Подтверждаю",
        scenario_id=None,
        chat_id=None,
        conversation="conversation",
        persist_history=False,
        task_id="task-1",
        generation_started=__import__("time").time(),
        parameters={"tags": ["REST"], "park_geojson": BOUNDARY},
        event_seq=0,
    )
    await parks.store.save(state)
    parks.llm_client.chat.return_value = llm_response("Анализ")
    events = await collect(parks, request_id="resumed")
    assert events[-1]["content"]["done"] is True
    parks.api.generate.assert_not_called()
    parks.api.status.assert_awaited_once_with("task-1")


@pytest.mark.asyncio
async def test_vision_inline_image_is_passed_to_model(parks):
    await collect(parks, inputs={"park_geojson": BOUNDARY})
    picture = "data:image/png;base64,aGVsbG8="
    parks.api.result.return_value = dict(RESULT, images=[picture])
    parks.llm_client.chat.return_value = llm_response("Анализ изображения")
    await collect(parks, user_query="Подтверждаю")
    messages = parks.llm_client.chat.call_args.args[1]
    assert messages[1]["content"][1]["image_url"]["url"] == picture


def test_get_sse_contract_and_recovery_validation(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.agents.common.auth.auth import verify_bearer_token
    from src.agents.dependencies.dependencies import app_deps
    from src.agents.routers.parks_a2a_controller import parks_a2a_router
    from src.agents.routers.parks_controller import get_parks_service, parks_router

    class Service:
        calls = []

        async def run_parks_pipeline(self, **kwargs):
            self.calls.append(kwargs)
            yield {
                "type": "pipeline_started",
                "content": {"request_id": "id", "event_id": 1},
            }
            yield {
                "type": "chunk",
                "content": {"text": "", "done": True, "event_id": 2},
            }

        async def get_run(self, *args):
            raise LookupError()

    service = Service()
    app = FastAPI()
    app.include_router(parks_router)
    app.include_router(parks_a2a_router)
    monkeypatch.setitem(app_deps, "parks_a2a_service", ParksA2AService(service))
    monkeypatch.setattr(
        "src.agents.routers.parks_a2a_controller.resolve_a2a_caller",
        AsyncMock(return_value=SimpleNamespace(pipeline_token=TOKEN)),
    )
    app.dependency_overrides[get_parks_service] = lambda: service
    app.dependency_overrides[verify_bearer_token] = lambda: TOKEN
    with TestClient(app) as client:
        response = client.get(
            "/parks/generate/stream",
            params={
                "request": "Парк",
                "inputs": json.dumps({"park_geojson": BOUNDARY}),
            },
        )
        assert response.status_code == 200
        assert "id: 1" in response.text
        assert '"type":"pipeline_started"' in response.text.replace(" ", "")
        assert service.calls[0]["inputs"].park_geojson == BOUNDARY
        card = client.get("/parks/.well-known/agent-card.json").json()
        assert card["url"] == "http://testserver/parks/a2a"
        a2a = client.post(
            "/parks/a2a",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/stream",
                "params": {
                    "contextId": "ctx",
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "Парк"}],
                    },
                },
            },
        )
        assert a2a.status_code == 200
        assert "text/event-stream" in a2a.headers["content-type"]
        assert "completed" in a2a.text
        assert (
            client.get(
                "/parks/generate/stream", params={"request": "Парк", "inputs": "{bad"}
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/parks/generate/stream",
                params={"request": "Парк"},
                headers={"Last-Event-ID": "bad"},
            ).status_code
            == 422
        )
        assert (
            client.get(
                "/parks/generate/stream",
                params={
                    "request": "Парк",
                    "request_id": "00000000-0000-0000-0000-000000000000",
                },
            ).status_code
            == 404
        )


@pytest.mark.asyncio
async def test_orchestrator_reconnect_delivers_result_without_resubmitting(
    parks, monkeypatch
):
    from src.agents.services.orchestrator.orchestrator_service import (
        OrchestratorService,
    )
    from src.agents.services.service_entities.orchestrator_plan import OrchestratorPlan

    orchestrator = OrchestratorService(
        "http://localhost",
        Mock(),
        Mock(),
        parks.state_store,
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        SimpleNamespace(
            DVD_MCP_URL=None,
            NORM_GRAPH_MCP_URL=None,
            URBAN_MCP_URL=None,
            PARKS_API_URL="https://parks.test",
        ),
        parks_service=parks,
    )
    orchestrator.resolve_model = AsyncMock(return_value="vision")
    orchestrator.create_chat = AsyncMock(return_value=("chat-outer", "Парк"))
    orchestrator.get_chat_messages = AsyncMock(
        return_value=SimpleNamespace(messages=[])
    )
    orchestrator.add_single_message = AsyncMock()
    orchestrator._schedule_persist_summary = Mock()
    orchestrator.plan_builder.build_plan = AsyncMock(
        return_value=OrchestratorPlan.model_validate(
            {"mode": "execute", "steps": [{"agent": "parks", "task": "Создай парк"}]}
        )
    )
    kwargs = dict(
        idu_mcp_client=None,
        effects_mcp_client=None,
        dvd_mcp_client=None,
        normgraph_mcp_client=None,
        token=TOKEN,
        model="vision",
        temperature=0,
        user_query="Создай парк",
        parks_inputs={"park_geojson": BOUNDARY},
    )
    first = [event async for event in orchestrator.run_orchestration_pipeline(**kwargs)]
    assert any(
        event["type"] == "step_event"
        and event["content"]["event"]["type"] == "clarification"
        for event in first
    )
    kwargs.update(user_query="Подтверждаю", parks_inputs=None, chat_id="chat-outer")
    parks.llm_client.chat.return_value = llm_response("Итоговый анализ")
    stream = orchestrator.run_orchestration_pipeline(**kwargs)
    outer_id = None
    async for event in stream:
        if event["type"] == "pipeline_started":
            outer_id = event["content"]["request_id"]
        if (
            event["type"] == "step_event"
            and event["content"]["event"]["content"].get("status") == "submitting"
        ):
            break
    await stream.aclose()
    kwargs["request_id"] = outer_id
    resumed = [
        event async for event in orchestrator.run_orchestration_pipeline(**kwargs)
    ]
    assert any(
        event["type"] == "step_event"
        and event["content"]["event"]["type"] == "feature_collection"
        for event in resumed
    )
    assert resumed[-1]["type"] == "orchestrator_final"
    parks.api.generate.assert_awaited_once()
    parks.create_chat.assert_not_called()
