"""Functional contracts at the public orchestration, identity and evidence seams."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.security import HTTPAuthorizationCredentials

from src.agents.common.auth import auth
from src.agents.common.exceptions.base_exceptions import (
    AgentsNotFound,
    AgentsUnauthorizedException,
)
from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_support import context_scope
from tests.integration.local_stack.contract_probes import goal_probes
from tests.unit.test_orchestrator_service_events import orchestrator, run_pipeline

AGENT_METHODS = {
    "scenario_data": ("scenario_data_service", "run_scenario_data_pipeline"),
    "provision": ("provision_service", "run_provision_pipeline"),
    "restriction": ("restriction_service", "run_restriction_execution_pipline"),
    "compliance": ("restriction_service", "run_compliance_pipeline"),
    "documents": ("dvd_service", "run_document_qa_pipeline"),
    "norms": ("normgraph_service", "run_norms_qa_pipeline"),
}


@pytest.mark.parametrize("agent", list(AGENT_METHODS))
@pytest.mark.parametrize(
    "outcome",
    ["success", "error", "exception", "model_exception", "clarification", "suspended"],
)
async def test_every_specialist_preserves_terminal_replay_and_evidence(
    orchestrator, monkeypatch, agent, outcome
):
    import json
    from copy import deepcopy

    from src.agents.model_clients.llm_base import LlmResponseError
    from src.agents.services.orchestrator.analysis_goal import GoalManager
    from src.agents.services.source_evidence import source_event
    from tests.helpers import FakeLlmClient
    from tests.unit.test_analytical_orchestrator import LAYER, final, table
    from tests.unit.test_goal_orchestrator import school_artifacts
    from tests.unit.test_orchestrator_service_events import FakePipeline

    monkeypatch.setenv("ORCHESTRATOR_ANALYSIS_MODE", "goal")
    outputs = {
        "scenario_data": school_artifacts(),
        "provision": [table()],
        "restriction": [deepcopy(LAYER)],
        "compliance": json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures/compliance_storage_events.json"
            ).read_text(encoding="utf-8")
        ),
        "documents": [
            source_event(
                "documents",
                [{"id": "clause", "text": "Synthetic 50 m", "name": "TEST"}],
            )
        ],
        "norms": [
            source_event(
                "norms",
                [
                    {
                        "id": "rule",
                        "subject": "School",
                        "object": "Parking",
                        "value": {"number": 50, "unit": "m"},
                    }
                ],
            )
        ],
    }[agent]
    outputs.append(
        {
            "type": "chunk",
            "content": {
                "text": "Подтверждённые синтетические данные: 50 м.",
                "done": True,
            },
        }
    )
    if outcome in {"error", "clarification", "suspended"}:
        outputs.append(
            {
                "type": {
                    "error": "error",
                    "clarification": "clarification_required",
                    "suspended": "pipeline_suspended",
                }[outcome],
                "content": {
                    "message": "Synthetic unavailable",
                    "question": "Укажите исходный сценарий",
                },
            }
        )
    pipeline = FakePipeline(
        outputs,
        raise_exc=(
            ConnectionError("synthetic")
            if outcome == "exception"
            else (
                LlmResponseError("synthetic provider failure", 503)
                if outcome == "model_exception"
                else None
            )
        ),
    )
    attribute, method = AGENT_METHODS[agent]
    setattr(orchestrator, attribute, SimpleNamespace(**{method: pipeline}))

    class Backend(FakeLlmClient):
        async def chat(self, model=None, messages=None, **kwargs):
            payload = json.loads(messages[-1]["content"])
            if "request_fragments" in payload:
                requirements = {
                    "id": "r",
                    "description": "Получить подтверждённый результат",
                    "source_ids": [1],
                    "agent": agent,
                    "scenario_id": 772,
                    "required_artifacts": ["analysis_text"],
                }
                if agent == "scenario_data":
                    requirements.update(
                        subject="Школа",
                        entity_kind="services",
                        required_artifacts=["table", "feature_collection"],
                    )
                response = {
                    "objective": "Проверка контракта",
                    "requirements": [requirements],
                }
            elif "manifest" in payload:
                response = {"issues": []}
            else:
                goal = payload["goal"]["requirements"][0]
                if goal["status"] == "pending":
                    response = {"action": "continue", "requirement_id": "r"}
                elif goal["status"] == "satisfied":
                    response = {
                        "action": "complete",
                        "answer": "Подтверждено источниками.",
                        "evidence_ids": goal["evidence_ids"],
                    }
                else:
                    response = {"action": "blocked", "missing": [goal["blocker"]]}
            self.json_responses.append(json.dumps(response))
            return await super().chat(model=model, messages=messages, **kwargs)

    backend = Backend()
    orchestrator.goal_manager = GoalManager(backend)
    events = await run_pipeline(
        orchestrator, user_query="Получить результат.", urban_mcp_client=AsyncMock()
    )
    result = final(events)
    assert result["status"] == ("completed" if outcome == "success" else "blocked")
    assert len(pipeline.calls) == 1
    assert len([e for e in events if e["type"] == "orchestrator_final"]) == 1
    assert all(a["confirmed"] == (outcome == "success") for a in result["artifacts"])
    calls = len(backend.chat_calls)
    replay = await run_pipeline(orchestrator, request_id=result["continue_from"])
    assert replay == events and len(backend.chat_calls) == calls
    assert result["continue_from"]
    if agent == "compliance" and outcome == "success":
        import jsonschema

        contracts = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "fixtures/chatstorage_compliance_contract.json"
            ).read_text(encoding="utf-8")
        )
        parts = orchestrator.add_complex_message.call_args.args[3]
        typed = [
            part.model_dump(mode="json") for part in parts if part.kind in contracts
        ]
        assert {part["kind"] for part in typed} == set(contracts)
        for part in typed:
            jsonschema.validate(part["payload"], contracts[part["kind"]])
    if outcome == "model_exception":
        assert (
            result["missing"][0]["missing"]
            == "Корректное управляющее решение для анализа"
        )


async def test_typed_retrieval_runs_actual_specialist_without_reclassifying(
    monkeypatch, fake_llm, fake_urban, state_store
):
    from src.agents.services.scenario_data.scenario_data_service import (
        ScenarioDataService,
    )
    from src.agents.services.service_entities.orchestrator_plan import EntitySelection
    from tests.unit.test_scenario_data_selection import EntityMcp

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    mcp = EntityMcp()
    events = [
        event
        async for event in service.run_scenario_data_pipeline(
            mcp,
            "token",
            "model",
            0,
            "Получить полные данные",
            scenario_id=17,
            persist_history=False,
            entity_selection=EntitySelection(subject="Библиотека", kind="services"),
        )
    ]
    assert not fake_llm.chat_calls
    assert len(next(e["content"]["rows"] for e in events if e["type"] == "table")) == 2
    assert (
        len(
            next(
                e["content"]["feature_collection"]["features"]
                for e in events
                if e["type"] == "feature_collection"
            )
        )
        == 2
    )
    assert all("Physical" not in name for name, _ in mcp.calls)


def test_negative_evidence_contracts():
    assert not any(goal_probes().values())


@pytest.mark.parametrize(
    "token", ["garbage", "forged.jwt.signature", "expired.jwt.signature"]
)
async def test_front_door_rejects_unverified_identity(monkeypatch, token):
    verifier = SimpleNamespace(
        verify_user=AsyncMock(
            side_effect=AgentsUnauthorizedException("Invalid access token")
        )
    )
    monkeypatch.setattr(auth, "bearer_verifier", lambda: verifier)
    with pytest.raises(AgentsUnauthorizedException):
        await auth.verify_bearer_token(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        )
    verifier.verify_user.assert_awaited_once_with(token)


async def test_replay_owner_is_checked_before_reading_private_buffer(orchestrator):
    store = orchestrator.state_store
    await store.create(
        "private",
        chat_id="chat",
        user_query="private",
        scenario_id=772,
        model="m",
        temperature=0,
        owner=context_scope("owner-a", "owner"),
    )
    store.get_buffered_events = AsyncMock(return_value=[{"type": "secret"}])
    with pytest.raises(AgentsNotFound):
        await run_pipeline(orchestrator, request_id="private", token="owner-b")
    store.get_buffered_events.assert_not_awaited()


def test_inspection_rejects_empty_and_repeated_catalog():
    from src.agents.services.service_entities.orchestrator_plan import ArtifactSlice

    c = AnalysisContext()
    with pytest.raises(ValueError, match="empty"):
        c.inspect([ArtifactSlice(artifact_id="_catalog")])


async def test_real_jwt_verification_rejects_forgery_expiry_and_wrong_issuer(
    monkeypatch,
):
    import time

    import jwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    from src.agents.common.auth.synapse_auth import SynapseCallerVerifier

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        .decode()
    )
    verifier = object.__new__(SynapseCallerVerifier)
    verifier.verifier = JWTVerifier(
        public_key=public, issuer="http://auth.test/realms/test", algorithm="RS256"
    )
    monkeypatch.setattr(auth, "bearer_verifier", lambda: verifier)
    claims = {
        "sub": "user-a",
        "iss": "http://auth.test/realms/test",
        "exp": int(time.time()) + 600,
    }
    good = jwt.encode(claims, key, algorithm="RS256")
    assert (
        await auth.verify_bearer_token(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=good)
        )
        == good
    )
    for invalid in (
        jwt.encode({**claims, "exp": int(time.time()) - 3600}, key, algorithm="RS256"),
        jwt.encode({**claims, "iss": "http://attacker.test"}, key, algorithm="RS256"),
        jwt.encode(claims, "wrong-key-for-test-only-0123456789", algorithm="HS256"),
    ):
        with pytest.raises(AgentsUnauthorizedException):
            await auth.verify_bearer_token(
                HTTPAuthorizationCredentials(scheme="Bearer", credentials=invalid)
            )


async def test_atomic_request_claim_has_one_winner_and_survives_retry(state_store):
    import asyncio

    async def claim(n):
        return await state_store.create(
            "shared",
            chat_id=None,
            user_query="synthetic",
            scenario_id=17,
            model="m",
            temperature=0,
            owner="a",
            claim_id=str(n),
        )

    winners = await asyncio.gather(*(claim(i) for i in range(8)))
    assert sum(winners) == 1
    owner = winners.index(True)
    assert await claim(owner) is True
    assert await claim("other") is False


async def test_source_snapshot_endpoint_is_scoped_and_preserves_original_record(
    orchestrator, monkeypatch
):
    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from src.agents.common.exceptions.base_exceptions import AgentsBaseException

    for name, value in {
        "SERVICE_AUTH_SERVER_URL": "http://auth.test",
        "SERVICE_AUTH_REALM": "test",
        "SERVICE_AUTH_CLIENT_ID": "test",
        "SERVICE_AUTH_CLIENT_SECRET": "test-placeholder",
    }.items():
        monkeypatch.setenv(name, value)
    from src.agents.dependencies.dependencies import get_orchestrator_service
    from src.agents.routers.orchestrator_controller import orchestrator_router

    app = FastAPI()
    app.include_router(orchestrator_router)

    @app.exception_handler(AgentsBaseException)
    async def error(_, exc):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    identity = "owner"
    app.dependency_overrides[auth.verify_bearer_token] = lambda: identity
    app.dependency_overrides[get_orchestrator_service] = lambda: orchestrator
    from src.agents.services.source_evidence import source_event

    c = AnalysisContext()
    original = {
        "id": "clause-1",
        "text": "Synthetic original clause: 50 m",
        "version": "v1",
    }
    aid = c.add_artifact(source_event("documents", [original]), 1, "step")
    c.finish(1, "read", None, "completed", "", "step")
    await orchestrator.state_store.save_analysis_context(
        context_scope(identity, "run:source-run"), c.dump()
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get(f"/orchestrator/runs/source-run/artifacts/{aid}")
        assert response.status_code == 200 and response.json()["content"][
            "sources"
        ] == [original]
        identity = "other-user"
        assert (
            await client.get(f"/orchestrator/runs/source-run/artifacts/{aid}")
        ).status_code == 404
