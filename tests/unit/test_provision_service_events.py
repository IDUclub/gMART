"""Unit tests for provision pipeline event/part helpers (no I/O)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agents.api_clients.chat_storage_client.request_models import (
    TablePartRequest,
)
from src.agents.schema.provision_response import ProvisionResponse
from src.agents.services.provision_tool_executor import ProvisionToolExecutor
from src.agents.services.provsion_service import POPULATION_HINT, ProvisionService

TABLE_EVENT = {
    "type": "table",
    "content": {
        "name": "provision_summary",
        "title": "Сводка обеспеченности сервисами",
        "columns": [
            {"key": "service", "label": "Сервис"},
            {"key": "deficit", "label": "Дефицит (чел)"},
        ],
        "rows": [{"service": "Школы", "deficit": 250}],
    },
}


def test_table_event_maps_to_table_part():
    part = ProvisionService._pipeline_item_to_chat_part(TABLE_EVENT)
    assert isinstance(part, TablePartRequest)
    assert part.kind == "table"
    assert part.payload.name == "provision_summary"
    assert part.payload.columns[0].key == "service"
    assert part.payload.rows == [{"service": "Школы", "deficit": 250}]


def test_table_event_validates_as_sse_response():
    response = ProvisionResponse(**TABLE_EVENT)
    assert response.type == "table"
    assert response.content.name == "provision_summary"


def test_provision_feature_collections_named_per_service():
    fc = {"type": "FeatureCollection", "features": []}
    services_result = {
        "services": {
            "22": {
                "name": "Школы",
                "layers": {"buildings": fc, "links": fc},
            },
            "21": {"name": "Детские сады", "layers": None},
        }
    }
    events = list(ProvisionService._provision_feature_collections(services_result))
    names = {event["content"]["name"] for event in events}
    assert names == {"provision.Школы.buildings", "provision.Школы.links"}
    assert all(event["type"] == "feature_collection" for event in events)


def test_single_service_result_handles_string_keys():
    services_result = {"services": {"22": {"name": "Школы"}}}
    assert ProvisionService._single_service_result(services_result, 22) == {
        "name": "Школы"
    }
    assert ProvisionService._single_service_result({"services": {}}, 22) is None


class FakeEffectsMcp:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def calculate_services_provision(self, **kwargs):
        self.kwargs = kwargs
        return {"services": {}}


@pytest.mark.asyncio
async def test_tool_executor_records_target_population():
    effects_mcp = FakeEffectsMcp()
    result = await ProvisionToolExecutor().calculate_services_provision(
        effects_mcp,
        scenario_id=192,
        services={22: {"name": "Школы", "as_layer": False}},
        target_population=25000,
    )
    assert effects_mcp.kwargs["target_population"] == 25000
    arguments = result.tool_calls[0]["function"]["arguments"]
    assert arguments["target_population"] == 25000


@pytest.mark.asyncio
async def test_tool_executor_omits_absent_target_population():
    effects_mcp = FakeEffectsMcp()
    result = await ProvisionToolExecutor().calculate_services_provision(
        effects_mcp,
        scenario_id=192,
        services={22: {"name": "Школы", "as_layer": False}},
    )
    arguments = result.tool_calls[0]["function"]["arguments"]
    assert "target_population" not in arguments


class FakeStreamingLlm:
    """Yields two chunks; the second is the terminal one (done=True)."""

    async def chat(self, model, messages, options=None, stream=True):
        async def stream_parts():
            yield SimpleNamespace(
                message=SimpleNamespace(content="Анализ. "), done=False
            )
            yield SimpleNamespace(message=SimpleNamespace(content=""), done=True)

        return stream_parts()


@pytest.mark.asyncio
async def test_generate_analysis_appends_population_hint():
    service = ProvisionService.__new__(ProvisionService)
    service.llm_client = FakeStreamingLlm()
    chunks = [
        chunk
        async for chunk in service._generate_analysis(
            "model", "запрос", "контекст", 1.0, trailing_note=POPULATION_HINT
        )
    ]
    # LLM chunks must not be terminal when a trailing note follows
    assert all(not chunk["content"]["done"] for chunk in chunks[:-1])
    assert chunks[-1]["content"]["text"] == POPULATION_HINT
    assert chunks[-1]["content"]["done"] is True


@pytest.mark.asyncio
async def test_generate_analysis_without_note_keeps_terminal_chunk():
    service = ProvisionService.__new__(ProvisionService)
    service.llm_client = FakeStreamingLlm()
    chunks = [
        chunk
        async for chunk in service._generate_analysis(
            "model", "запрос", "контекст", 1.0
        )
    ]
    assert chunks[-1]["content"]["done"] is True
    assert all(chunk["content"]["text"] != POPULATION_HINT for chunk in chunks)
