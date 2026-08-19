from types import SimpleNamespace

import pytest

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.services.provsion_service import ProvisionService
from src.agents.services.restriction_parser_service import RestrictionParserService
from src.agents.services.scenario_data_service import ScenarioDataService
from src.common.service_auth import ServiceTokenAuth


class NoUserTokenRefresh:
    async def is_cancelled(self, _request_id):
        return False

    async def wait_for_token(self, _request_id):
        raise AssertionError("pipeline must not wait for a user token")


class NoCredentialReplacement:
    def update_token(self, _token):
        raise AssertionError("pipeline must not replace M2M credentials")


async def expired_operation():
    raise TokenExpiredError("downstream rejected service credentials")


def m2m_client():
    auth = ServiceTokenAuth(object(), user_id="user-42")
    transport = SimpleNamespace(auth=auth)
    return SimpleNamespace(mcp_client=SimpleNamespace(transport=transport))


@pytest.mark.asyncio
async def test_restriction_entry_does_not_read_token_from_service_auth():
    service = object.__new__(RestrictionParserService)
    captured = {}

    async def resolve_model(model):
        return model

    async def run_inner(**kwargs):
        captured.update(kwargs)
        if False:  # pragma: no cover
            yield {}

    service.resolve_model = resolve_model
    service._run_restriction_execution_pipline = run_inner

    events = [
        event
        async for event in service._run_pipeline_entry(
            mcp_client=m2m_client(),
            token="user-jwt",
            temperature=0,
            model="model",
            user_query="query",
            scenario_id=772,
            persist_history=False,
        )
    ]

    assert events == []
    assert captured["token_ref"] == ["user-jwt"]


@pytest.mark.asyncio
async def test_provision_entry_does_not_read_token_from_service_auth():
    service = object.__new__(ProvisionService)
    captured = {}

    async def resolve_model(model):
        return model

    async def run_inner(**kwargs):
        captured.update(kwargs)
        if False:  # pragma: no cover
            yield {}

    service.resolve_model = resolve_model
    service._run_provision_pipeline = run_inner

    events = [
        event
        async for event in service.run_provision_pipeline(
            idu_mcp_client=m2m_client(),
            effects_mcp_client=m2m_client(),
            token="user-jwt",
            temperature=0,
            model="model",
            user_query="query",
            scenario_id=772,
            persist_history=False,
        )
    ]

    assert events == []
    assert captured["token_ref"] == ["user-jwt"]


@pytest.mark.asyncio
async def test_restriction_pipeline_does_not_replace_m2m_credentials():
    service = object.__new__(RestrictionParserService)
    service.state_store = NoUserTokenRefresh()

    with pytest.raises(TokenExpiredError):
        async for _ in service._retryable_step(
            "request-1",
            NoCredentialReplacement(),
            ["user-token"],
            expired_operation,
            [],
        ):
            pass


@pytest.mark.asyncio
async def test_provision_pipeline_does_not_replace_m2m_credentials():
    service = object.__new__(ProvisionService)
    service.state_store = NoUserTokenRefresh()

    with pytest.raises(TokenExpiredError):
        async for _ in service._retryable_step(
            "request-1",
            NoCredentialReplacement(),
            NoCredentialReplacement(),
            ["user-token"],
            expired_operation,
            [],
        ):
            pass


@pytest.mark.asyncio
async def test_scenario_pipeline_does_not_replace_m2m_credentials():
    service = object.__new__(ScenarioDataService)
    service.state_store = NoUserTokenRefresh()

    with pytest.raises(TokenExpiredError):
        async for _ in service._retryable_operation(
            "request-1",
            NoCredentialReplacement(),
            ["user-token"],
            expired_operation,
            [],
        ):
            pass
