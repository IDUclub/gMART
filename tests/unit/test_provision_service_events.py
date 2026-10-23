"""Unit tests for provision pipeline event/part helpers (no I/O)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.api_clients.chat_storage_client.request_models import (
    TablePartRequest,
)
from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.schema.provision_response import ProvisionResponse
from src.agents.services.pipeline_state import PipelineStatus, PipelineStep
from src.agents.services.provision.provision_context import ProvisionContextBuilder
from src.agents.services.provision.provision_tool_executor import ProvisionToolExecutor
from src.agents.services.provision.provsion_service import (
    BASE_SCENARIO_EFFECTS_MESSAGE,
    POPULATION_HINT,
    ProvisionService,
)
from src.agents.services.service_entities.provision_plan import ProvisionPlan

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

    async def chat(self, model, messages, options=None, stream=True, **kwargs):
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


@pytest.mark.parametrize(
    "base_id,checkpoint_present",
    [
        (772, False),
        (771, False),
        (None, False),
        (RuntimeError("lookup failed"), False),
        (772, True),
    ],
)
async def test_single_effects_base_scenario_guard(
    state_store, base_id, checkpoint_present
):
    service = ProvisionService.__new__(ProvisionService)
    service.state_store = state_store
    service.tool_executor = ProvisionToolExecutor()
    service.context_builder = ProvisionContextBuilder()
    service.llm_client = FakeStreamingLlm()
    lookup = AsyncMock(return_value=base_id)
    if isinstance(base_id, Exception):
        lookup.side_effect = base_id
    service.urban_api_client = SimpleNamespace(get_project_base_scenario_id=lookup)
    idu_mcp = SimpleNamespace(execute_tool=AsyncMock(return_value=22))
    effects_mcp = SimpleNamespace(calculate_object_effects=AsyncMock(return_value={}))
    await state_store.create(
        "single-effects",
        chat_id=None,
        user_query="compare",
        scenario_id=772,
        model="m",
        temperature=0,
    )
    checkpoint = {}
    if checkpoint_present:
        checkpoint = {
            PipelineStep.GET_SERVICE_ID: {"service_type_id": 22},
            PipelineStep.CALCULATE_EFFECTS: {},
        }
    events = [
        event
        async for event in service._run_effects(
            request_id="single-effects",
            idu_mcp_client=idu_mcp,
            effects_mcp_client=effects_mcp,
            token_ref=["t"],
            checkpoint=checkpoint,
            plan=ProvisionPlan(mode="effects", service_name="Школы"),
            model="m",
            temperature=0,
            user_query="compare",
            scenario_id=772,
            llm_history=[],
        )
    ]
    if checkpoint_present:
        lookup.assert_not_awaited()
        idu_mcp.execute_tool.assert_not_awaited()
        effects_mcp.calculate_object_effects.assert_not_awaited()
    elif base_id == 772:
        lookup.assert_awaited_once_with("t", 772)
        idu_mcp.execute_tool.assert_not_awaited()
        effects_mcp.calculate_object_effects.assert_not_awaited()
        assert events == [
            service._status("effects_calculation", BASE_SCENARIO_EFFECTS_MESSAGE),
            service._chunk(BASE_SCENARIO_EFFECTS_MESSAGE, done=True),
        ]
    else:
        idu_mcp.execute_tool.assert_awaited_once()
        effects_mcp.calculate_object_effects.assert_awaited_once()
    assert (await state_store.get_state("single-effects"))[
        "status"
    ] == PipelineStatus.DONE


async def test_single_effects_base_lookup_token_expiration_propagates(state_store):
    service = ProvisionService.__new__(ProvisionService)
    service.state_store = state_store
    service.urban_api_client = SimpleNamespace(
        get_project_base_scenario_id=AsyncMock(side_effect=TokenExpiredError("expired"))
    )
    with pytest.raises(TokenExpiredError):
        async for _ in service._run_effects(
            request_id="expired",
            idu_mcp_client=object(),
            effects_mcp_client=object(),
            token_ref=["t"],
            checkpoint={},
            plan=ProvisionPlan(mode="effects", service_name="Школы"),
            model="m",
            temperature=0,
            user_query="compare",
            scenario_id=772,
            llm_history=[],
        ):
            pytest.fail("No events expected before token expiration propagates")
