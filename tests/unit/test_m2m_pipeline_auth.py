import pytest

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.services.provsion_service import ProvisionService
from src.agents.services.restriction_parser_service import RestrictionParserService
from src.agents.services.scenario_data_service import ScenarioDataService


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
