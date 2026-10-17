"""REST, A2A and orchestrator entry points of a compliance check of user files."""

import json
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agents.a2a.compliance_executor import ComplianceAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.auth.auth import verify_bearer_token
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.common.middlewares.exception_handler import ExceptionHandlerMiddleware
from src.agents.dependencies.dependencies import (
    get_compliance_upload_service,
    get_idu_mcp_client,
    get_optional_dvd_mcp_client,
    get_optional_normgraph_mcp_client,
    get_restriction_parser_service,
)
from src.agents.routers.compliance_controller import compliance_router
from src.agents.services.compilance.compliance_uploads import ComplianceUploadService
from src.agents.services.orchestrator.orchestrator_catalog import (
    OrchestratorAgent,
    available_agents,
)
from src.agents.services.pipeline_state import PipelineStateStore

ALICE = "e30.eyJzdWIiOiJhbGljZSJ9.sig"  # {"sub": "alice"}
BOB = "e30.eyJzdWIiOiJib2IifQ.sig"  # {"sub": "bob"}

GEOJSON = json.dumps(
    {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [37.48, 55.67]},
                "properties": {"type": "buildings", "building": "residential"},
            }
        ],
    }
).encode("utf-8")


class Service:
    def __init__(self, uploads):
        self.calls = []
        self.compliance_uploads = uploads

    async def run_compliance_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        yield {"type": "chunk", "content": {"text": "Готово", "done": True}}


@pytest.fixture
def client():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    uploads = ComplianceUploadService(PipelineStateStore(redis))
    service = Service(uploads)
    token = {"value": ALICE}
    app = FastAPI()
    app.add_middleware(ExceptionHandlerMiddleware)
    app.include_router(compliance_router)
    app.dependency_overrides[verify_bearer_token] = lambda: token["value"]
    app.dependency_overrides[get_compliance_upload_service] = lambda: uploads
    app.dependency_overrides[get_restriction_parser_service] = lambda: service
    app.dependency_overrides[get_idu_mcp_client] = lambda: object()
    app.dependency_overrides[get_optional_normgraph_mcp_client] = lambda: None
    app.dependency_overrides[get_optional_dvd_mcp_client] = lambda: None
    return TestClient(app), service, token


def test_upload_roles_and_owner(client):
    http, _, token = client
    response = http.post(
        "/compliance/uploads",
        files=[
            ("files", ("336195_map_objects.geojson", GEOJSON, "application/geo+json")),
            (
                "files",
                ("336195_osm_surroundings.geojson", GEOJSON, "application/geo+json"),
            ),
        ],
    )
    assert response.status_code == 201
    upload = response.json()
    assert [layer["role"] for layer in upload["layers"]] == ["project", "context"]
    upload_id = upload["upload_id"]

    patched = http.patch(
        f"/compliance/uploads/{upload_id}",
        json={"roles": {"336195_osm_surroundings": "project"}},
    )
    assert [layer["role"] for layer in patched.json()["layers"]] == [
        "project",
        "project",
    ]
    assert (
        http.patch(
            f"/compliance/uploads/{upload_id}", json={"roles": {"x": "nope"}}
        ).status_code
        == 422
    )

    token["value"] = BOB
    assert http.get(f"/compliance/uploads/{upload_id}").status_code == 404
    token["value"] = ALICE
    assert http.delete(f"/compliance/uploads/{upload_id}").status_code == 204
    assert http.get(f"/compliance/uploads/{upload_id}").status_code == 404


def test_bad_file_is_a_client_error(client):
    http, _, _ = client
    response = http.post(
        "/compliance/uploads", files=[("files", ("plan.dxf", b"0", "text/plain"))]
    )
    assert response.status_code == 400


def test_check_takes_uploads_instead_of_a_scenario(client):
    http, service, _ = client
    assert (
        http.get("/compliance/check/stream", params={"request": "Проверь"}).status_code
        == 422
    )
    response = http.get(
        "/compliance/check/stream",
        params=[("request", "Проверь"), ("upload_ids", "u1"), ("upload_ids", "u2")],
    )
    assert response.status_code == 200
    assert service.calls[-1]["upload_ids"] == ["u1", "u2"]
    assert service.calls[-1]["scenario_id"] is None
    body = http.post(
        "/compliance/check/stream",
        json={"request": "Проверь", "scenario_id": 772, "upload_ids": ["u3"]},
    )
    assert body.status_code == 200
    assert service.calls[-1]["upload_ids"] == ["u3"]
    assert service.calls[-1]["scenario_id"] == 772


def a2a_params(data):
    return {
        "message": {
            "role": "user",
            "parts": [
                {"type": "text", "text": "Проверь мой проект"},
                {"type": "data", "data": data},
            ],
        }
    }


def test_a2a_accepts_files_without_a_scenario():
    executor = ComplianceAgentExecutor(SimpleNamespace(), A2ATaskStore())
    execution = executor._prepare_execution(a2a_params({"upload_ids": "u1"}))
    assert execution["scenario_id"] is None
    assert execution["upload_ids"] == ["u1"]
    inline = executor._prepare_execution(
        a2a_params(
            {
                "files": [
                    {
                        "name": "map",
                        "role": "project",
                        "geojson": {"type": "FeatureCollection", "features": []},
                    }
                ]
            }
        )
    )
    assert inline["inline_files"][0]["name"] == "map"
    assert "files" not in inline["metadata"]
    with pytest.raises(A2AInvalidParamsError, match="scenario_id"):
        executor._prepare_execution(a2a_params({}))
    with pytest.raises(A2AInvalidParamsError, match="role"):
        executor._prepare_execution(
            a2a_params({"files": [{"role": "design", "geojson": {}}]})
        )


async def test_a2a_inline_files_become_uploads_of_the_caller():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    uploads = ComplianceUploadService(PipelineStateStore(redis))
    executor = ComplianceAgentExecutor(Service(uploads), A2ATaskStore())
    (upload_id,) = await executor._store_inline_files(
        [{"name": "map", "geojson": json.loads(GEOJSON)}], ALICE
    )
    record = await uploads.get(upload_id, "alice")
    assert record["files"] == ["map.geojson"]


def test_compliance_is_available_for_files_without_a_scenario():
    config = SimpleNamespace(
        DVD_MCP_URL=None, NORM_GRAPH_MCP_URL="x", URBAN_MCP_URL=None, PZZ_MCP_URL=None
    )
    keys = lambda agents: {entry.key for entry in agents}  # noqa: E731
    assert OrchestratorAgent.COMPLIANCE not in keys(available_agents(config, None))
    with_files = keys(available_agents(config, None, user_files=True))
    assert OrchestratorAgent.COMPLIANCE in with_files
    assert OrchestratorAgent.PROVISION not in with_files


async def test_orchestrator_routes_a_mapping_answer_back_to_compliance():
    from unittest.mock import AsyncMock

    from src.agents.services.compilance.compliance_entity_mapping import MappingReply
    from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
    from src.agents.services.orchestrator.orchestrator_service import (
        OrchestratorService,
    )

    pending = {
        "upload_ids": ["u1"],
        "mapping": {"proposals": [], "pending": []},
        "candidates": [],
    }
    service = object.__new__(OrchestratorService)
    service.state_store = SimpleNamespace(
        set_compliance_mapping=AsyncMock(),
        get_compliance_choice=AsyncMock(return_value=None),
    )
    resolve_reply = AsyncMock(return_value=MappingReply(kind="accept"))
    service.restriction_service = SimpleNamespace(
        entity_mapper=SimpleNamespace(resolve_reply=resolve_reply)
    )
    agents = [AGENT_CATALOG[OrchestratorAgent.COMPLIANCE]]

    plan = await service._compliance_choice_plan(
        "m", "да", "chat-1", agents, object(), pending_mapping=pending
    )
    assert [step.agent for step in plan.steps] == [OrchestratorAgent.COMPLIANCE]

    resolve_reply.return_value = MappingReply(kind="new_request")
    assert (
        await service._compliance_choice_plan(
            "m", "что такое СП 42?", "chat-1", agents, object(), pending_mapping=pending
        )
        is None
    )
    service.state_store.set_compliance_mapping.assert_awaited_once_with("chat-1", None)
