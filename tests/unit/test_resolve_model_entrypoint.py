"""``BaseLlmService.resolve_model`` is the one place a missing model is filled in.

REST and A2A both reach the pipelines through it, so this is what guarantees neither carries a
backend-specific literal. Also pins the request DTO's default to None, since a non-None default
there would quietly bypass the resolver for every REST caller.
"""

from __future__ import annotations

import pytest

from src.agents.dto.llm_request_dto import SimpleRequestDTO
from src.agents.model_clients.model_defaults import (
    DEFAULT_MODEL_ENV,
    DEFAULT_MODEL_HINT_ENV,
    invalidate_default_model,
)
from src.agents.services.base_llm_service import BaseLlmService


class FakeAdapter:
    def __init__(self, models: list[str]):
        self._models = models
        self.list_calls = 0

    async def list(self):
        self.list_calls += 1
        return {"models": [{"model": m, "name": m} for m in self._models]}


def _service(models: list[str]) -> BaseLlmService:
    service = BaseLlmService.__new__(BaseLlmService)  # no network in __init__
    service.llm_client = FakeAdapter(models)
    return service


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_MODEL_HINT_ENV, raising=False)
    invalidate_default_model()
    yield
    invalidate_default_model()


class TestResolveModel:
    @pytest.mark.asyncio
    async def test_none_is_filled_from_the_provider(self):
        service = _service(["llama3.1:8b", "gpt-oss-20b"])
        assert await service.resolve_model(None) == "gpt-oss-20b"

    @pytest.mark.asyncio
    async def test_an_empty_string_is_treated_as_absent(self):
        """The UI sends "" before it has loaded the provider's list."""
        service = _service(["gpt-oss-20b"])
        assert await service.resolve_model("") == "gpt-oss-20b"

    @pytest.mark.asyncio
    async def test_a_named_model_passes_through_without_a_provider_call(self):
        """Validating here would add a model-list round trip to every request."""
        service = _service(["gpt-oss-20b"])
        assert await service.resolve_model("some-other-model") == "some-other-model"
        assert service.llm_client.list_calls == 0


class TestRequestDto:
    def test_model_defaults_to_none(self):
        """A literal default here would bypass resolve_model for every REST caller."""
        assert SimpleRequestDTO(request="x").model is None

    def test_an_explicit_model_survives(self):
        assert SimpleRequestDTO(request="x", model="gpt-oss-20b").model == "gpt-oss-20b"

    def test_every_agent_dto_inherits_the_same_default(self):
        """The agent DTOs all extend SimpleRequestDTO — one default covers the REST surface."""
        from src.agents.dto.dvd_request_dto import DocumentQaRequestDTO
        from src.agents.dto.norms_request_dto import NormsQaRequestDTO
        from src.agents.dto.orchestrator_request_dto import OrchestratorRequestDTO
        from src.agents.dto.provision_request_dto import ProvisionRequestDTO
        from src.agents.dto.restriction_request_dto import RestrictionRequestDTO
        from src.agents.dto.scenario_data_request_dto import ScenarioDataRequestDTO

        for dto in (
            DocumentQaRequestDTO,
            NormsQaRequestDTO,
            OrchestratorRequestDTO,
            ProvisionRequestDTO,
            RestrictionRequestDTO,
            ScenarioDataRequestDTO,
        ):
            assert issubclass(dto, SimpleRequestDTO)
            assert dto.model_fields["model"].default is None, dto.__name__


class TestA2aExecutors:
    def test_no_executor_carries_a_hardcoded_model(self):
        """A2A used to default to "gpt-oss:20b" independently of the REST surface."""
        from src.agents.a2a.dvd_executor import DocumentQaAgentExecutor
        from src.agents.a2a.executor import RestrictionAgentExecutor
        from src.agents.a2a.normgraph_executor import NormGraphAgentExecutor
        from src.agents.a2a.provision_executor import ProvisionAgentExecutor
        from src.agents.a2a.scenario_data_executor import ScenarioDataAgentExecutor

        for executor in (
            DocumentQaAgentExecutor,
            RestrictionAgentExecutor,
            NormGraphAgentExecutor,
            ProvisionAgentExecutor,
            ScenarioDataAgentExecutor,
        ):
            assert executor.DEFAULT_MODEL is None, executor.__name__
