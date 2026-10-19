"""REST, A2A and orchestrator entry points of the project creation from files."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.a2a.project_executor import ProjectAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware
from src.agents.dependencies.dependencies import (
    get_compliance_upload_service,
    get_project_a2a_service,
    get_project_files_service,
)
from src.agents.routers.compliance_controller import compliance_router
from src.agents.routers.project_a2a_controller import project_a2a_router
from src.agents.routers.project_controller import project_router
from src.agents.services.compilance.compliance_uploads import ComplianceUploadService
from src.agents.services.orchestrator.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.project_files.project_a2a_service import ProjectA2AService
from src.agents.services.service_entities.orchestrator_plan import OrchestratorAgent

ALICE = "e30.eyJzdWIiOiJhbGljZSJ9.sig"  # {"sub": "alice"}
CHAT = "550e8400-e29b-41d4-a716-446655440000"
GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [37.48, 55.67]},
            "properties": {"type": "buildings", "building": "residential"},
        }
    ],
}


class Service:
    def __init__(self, uploads, events=None):
        self.calls = []
        self.uploads = uploads
        self.events = events or [
            {"type": "chunk", "content": {"text": "Готово", "done": True}}
        ]

    async def run_project_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


@pytest.fixture
def client():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    store = PipelineStateStore(redis)
    uploads = ComplianceUploadService(store)
    service = Service(uploads)
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)
    app.include_router(project_router)
    app.include_router(project_a2a_router)
    app.include_router(compliance_router)
    app.dependency_overrides[verify_bearer_token] = lambda: ALICE
    app.dependency_overrides[get_project_files_service] = lambda: service
    app.dependency_overrides[get_compliance_upload_service] = lambda: uploads
    app.dependency_overrides[get_project_a2a_service] = lambda: ProjectA2AService(
        service
    )
    return TestClient(app), service, store


def test_post_stream_passes_uploads_and_inputs(client):
    http, service, _ = client

    response = http.post(
        "/projects/from-files/stream",
        json={
            "request": "Создай проект",
            "upload_ids": ["u1"],
            "chat_id": CHAT,
            "inputs": {"project_name": "Сад", "existing_objects": "keep"},
        },
    )

    assert response.status_code == 200
    assert '"type": "chunk"' in response.text
    call = service.calls[-1]
    assert call["upload_ids"] == ["u1"] and call["chat_id"] == CHAT
    assert call["inputs"].project_name == "Сад"
    assert call["inputs"].existing_objects == "keep"


def test_get_stream_takes_query_parameters(client):
    http, service, _ = client

    response = http.get(
        "/projects/from-files/stream",
        params=[
            ("request", "Создай проект"),
            ("upload_ids", "u1"),
            ("upload_ids", "u2"),
            ("territory_id", "3"),
        ],
    )

    assert response.status_code == 200
    assert service.calls[-1]["upload_ids"] == ["u1", "u2"]
    assert service.calls[-1]["inputs"].territory_id == 3


def test_unknown_input_is_rejected(client):
    http, _, _ = client

    response = http.post(
        "/projects/from-files/stream",
        json={"request": "Создай проект", "inputs": {"region": "Москва"}},
    )

    assert response.status_code == 422


def test_upload_with_a_chat_is_offered_later(client):
    http, _, store = client

    response = http.post(
        "/compliance/uploads",
        files=[("files", ("design.geojson", json.dumps(GEOJSON), "application/json"))],
        data={"chat_id": CHAT},
    )

    assert response.status_code == 201
    assert asyncio.run(store.get_chat_uploads(CHAT)) == [response.json()["upload_id"]]


def test_agent_card(client):
    http, _, _ = client

    card = http.get("/projects/.well-known/agent-card.json").json()

    assert card["name"] == "project-files-agent"
    assert card["url"].endswith("/projects/a2a")
    assert card["skills"][0]["id"] == "create-project-from-files"


def a2a_params(data, text="Создай проект"):
    return {
        "contextId": "ctx-1",
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": text}, {"type": "data", "data": data}],
        },
    }


def test_a2a_execution_reads_uploads_and_inputs():
    executor = ProjectAgentExecutor(SimpleNamespace(), A2ATaskStore())

    execution = executor._prepare_execution(
        a2a_params({"upload_ids": ["u1"], "project_name": "Сад", "files": []})
    )

    assert execution["upload_ids"] == ["u1"]
    assert execution["inputs"].project_name == "Сад"
    assert "files" not in execution["metadata"]
    with pytest.raises(A2AInvalidParamsError):
        executor._prepare_execution(a2a_params({"upload_ids": "u1"}))
    with pytest.raises(A2AInvalidParamsError):
        executor._prepare_execution(a2a_params({"inputs": {"territory_id": -1}}))


async def test_a2a_question_needs_input_and_summary_is_an_artifact():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    uploads = ComplianceUploadService(PipelineStateStore(redis))
    service = Service(
        uploads,
        events=[
            {"type": "project_summary", "content": {"project_id": 10}},
            {
                "type": "clarification",
                "content": {"question": "Выберите регион", "kind": "region"},
            },
        ],
    )
    executor = ProjectAgentExecutor(service, A2ATaskStore())

    events = [
        event
        async for event in executor.stream(
            a2a_params({"files": [{"name": "design", "geojson": GEOJSON}]}), ALICE
        )
    ]

    call = service.calls[-1]
    assert call["conversation_key"] == "a2a-project:ctx-1"
    assert call["persist_history"] is False
    assert len(call["upload_ids"]) == 1  # the inline layer became an upload
    artifact = next(event for event in events if event.get("kind") == "artifact-update")
    assert artifact["artifact"]["name"] == "project_summary"
    assert events[-1]["status"]["state"] == "input-required"
    assert events[-1]["final"] is True


async def test_orchestrator_routes_an_answer_back_to_the_project_dialogue():
    service = object.__new__(OrchestratorService)
    answers = AsyncMock(return_value=True)
    service.project_files_service = SimpleNamespace(answers_dialogue=answers)

    plan = await service._project_dialogue_plan("m", "2", "chat-1")

    assert [step.agent for step in plan.steps] == [OrchestratorAgent.PROJECT_FILES]
    answers.return_value = False
    assert await service._project_dialogue_plan("m", "погода?", "chat-1") is None
    assert await service._project_dialogue_plan("m", "2", None) is None


def test_orchestrator_step_passes_uploads_and_the_chat():
    service = object.__new__(OrchestratorService)
    project = Service(None)
    service.project_files_service = project
    step = SimpleNamespace(agent=OrchestratorAgent.PROJECT_FILES, task="создать проект")

    pipeline = service._build_step_pipeline(
        step,
        "создать проект",
        "step-1",
        idu_mcp_client=None,
        effects_mcp_client=None,
        dvd_mcp_client=None,
        normgraph_mcp_client=None,
        urban_mcp_client=None,
        token=ALICE,
        model="m",
        temperature=0,
        scenario_id=None,
        original_query="Создай проект «Сад» из файлов",
        conversation_key="chat-1",
        compliance_upload_ids=["u1"],
    )

    async def drain():
        return [event async for event in pipeline]

    asyncio.run(drain())
    call = project.calls[-1]
    assert call["user_query"] == "Создай проект «Сад» из файлов"
    assert call["conversation_key"] == "chat-1"
    assert call["upload_ids"] == ["u1"]
    assert call["persist_history"] is False
