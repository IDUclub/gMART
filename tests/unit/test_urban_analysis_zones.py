"""Zone discovery honours API constraints and validates inferred intent."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.dto.pzz_request_dto import PzzInputs
from src.agents.services.urban_analysis.urban_analysis_zones import resolve_zones


async def select(rows, supplied=None, inferred=None):
    api = SimpleNamespace(
        get_scenario_functional_zone_sources=AsyncMock(return_value=rows)
    )
    pzz = SimpleNamespace(
        resolve_inputs=AsyncMock(
            return_value=PzzInputs.model_validate(
                {"mode": "scenario", **(inferred or supplied or {})}
            )
        )
    )
    return await resolve_zones(
        api,
        pzz,
        token="t",
        scenario_id=17,
        model="m",
        user_query="Проведи анализ ПЗЗ",
        pzz_inputs=supplied,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "supplied, expected",
    [
        ({}, (2024, "PZZ")),
        ({"year": 2025}, (2025, "User")),
        ({"source": "OSM"}, (2026, "OSM")),
    ],
)
async def test_priority_latest_year_and_partial_constraints(supplied, expected):
    selected = await select(
        [
            {"source": "pzz", "year": 2023},
            {"name": "PzZ", "year": 2024},
            {"name": "USER", "year": 2025},
            {"source": "osm", "year": 2026},
            {"source": "unknown", "year": 2099},
            {"source": "PZZ", "year": "invalid"},
        ],
        supplied,
    )
    assert (selected.inputs.year, selected.inputs.source) == expected
    assert selected.inputs.mode == "scenario"


@pytest.mark.asyncio
async def test_false_pzz_mention_falls_back_with_note():
    selected = await select(
        [{"name": "User", "year": 2025}], inferred={"source": "PZZ"}
    )
    assert selected.inputs.source == "User"
    assert selected.notes and "автовыбор" in selected.notes[0]


@pytest.mark.asyncio
async def test_inferred_pair_is_used_only_when_available():
    selected = await select(
        [
            {"name": "PZZ", "year": 2026},
            {"name": "User", "year": 2024},
        ],
        inferred={"source": "User", "year": 2024},
    )
    assert (selected.inputs.source, selected.inputs.year) == ("User", 2024)


@pytest.mark.asyncio
async def test_inferred_value_does_not_override_trusted_constraint():
    selected = await select(
        [
            {"name": "PZZ", "year": 2026},
            {"name": "User", "year": 2025},
        ],
        {"year": 2025},
        {"source": "PZZ", "year": 2025},
    )
    assert (selected.inputs.source, selected.inputs.year) == ("User", 2025)


@pytest.mark.asyncio
async def test_complete_api_pair_is_trusted_and_file_fields_are_ignored():
    selected = await select(
        [],
        {
            "year": 1999,
            "source": "OSM",
            "mode": "pzz_check",
            "buildings_upload_id": "ignored",
        },
    )
    assert selected.inputs.mode == "scenario"
    assert selected.inputs.year == 1999
    assert selected.inputs.buildings_upload_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_pair", [False, True])
async def test_scenario_options_survive_zone_selection(complete_pair):
    options = {
        "physical_object_type_id": 7,
        "group_by": "object",
        "priority": 8,
        "force_recompute": True,
    }
    supplied = {
        **options,
        "mode": "pzz_check",
        "buildings_upload_id": "ignored",
        "pzz_zones_upload_id": "ignored",
        "cadastral_geojson": {"type": "FeatureCollection", "features": []},
    }
    if complete_pair:
        supplied.update(year=2025, source="PZZ")
    api = SimpleNamespace(
        get_scenario_functional_zone_sources=AsyncMock(
            return_value=[{"source": "PZZ", "year": 2025}]
        )
    )
    pzz = SimpleNamespace(resolve_inputs=AsyncMock(return_value=PzzInputs()))
    selected = await resolve_zones(
        api,
        pzz,
        token="t",
        scenario_id=17,
        model="m",
        user_query="q",
        pzz_inputs=supplied,
    )
    expected = PzzInputs(mode="scenario", year=2025, source="PZZ", **options)
    assert selected.inputs == expected
    if complete_pair:
        api.get_scenario_functional_zone_sources.assert_not_awaited()
        pzz.resolve_inputs.assert_not_awaited()
    else:
        pzz.resolve_inputs.assert_awaited_once_with(
            "m", "q", 17, PzzInputs(mode="scenario", **options)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rows,supplied", [([], {}), ([{"source": "OSM", "year": 2020}], {"year": 2021})]
)
async def test_no_suitable_zones_requires_upload(rows, supplied):
    selected = await select(rows, supplied)
    assert selected.status == "needs_clarification"
    assert "Загрузите" in selected.reason
    assert selected.inputs is None


@pytest.mark.asyncio
async def test_api_failure_is_explicit():
    api = SimpleNamespace(
        get_scenario_functional_zone_sources=AsyncMock(side_effect=RuntimeError)
    )
    selected = await resolve_zones(
        api, None, token="t", scenario_id=17, model="m", user_query="q"
    )
    assert selected.status == "failed"
    assert "Urban API" in selected.reason


@pytest.mark.asyncio
async def test_client_endpoint_and_token():
    handler = SimpleNamespace(get=AsyncMock(return_value=[]))
    await UrbanApiClient(handler).get_scenario_functional_zone_sources("token", 17)
    handler.get.assert_awaited_once_with(
        "/v1/scenarios/17/functional_zone_sources", auth_token="token"
    )
