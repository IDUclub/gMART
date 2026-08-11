"""Tests for the per-scenario Urban API response cache used by benchmark runs.

Covers:
  * the cache being off unless ``SCENARIO_CACHE_SIZE`` is set;
  * LRU eviction once more scenarios than the limit are touched;
  * TTL expiry;
  * ``UrbanApiTool.get_entity_by_names`` serving a repeat call without hitting
    the Urban API client again, while still returning independent models.
"""

from __future__ import annotations

import pytest

from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum
from src.idu_mcp.tools_services.scenario_cache import ScenarioCache
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


def _fc() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
                "properties": {},
            }
        ],
    }


class FakeUrbanClient:
    """Counts calls so a cache hit is observable."""

    def __init__(self):
        self.services_calls = 0
        self.name_id_calls = 0

    async def get_service_name_id(self, names, token):
        self.name_id_calls += 1
        return {name: i for i, name in enumerate(names)}

    async def get_services(self, scenario_id, ids, token):
        self.services_calls += 1
        return [_fc() for _ in ids]


def test_cache_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SCENARIO_CACHE_SIZE", raising=False)
    cache = ScenarioCache()
    assert not cache.enabled
    cache.set(1, ("k",), "v")
    assert cache.get(1, ("k",)) is None


def test_lru_evicts_the_least_recently_used_scenario():
    cache = ScenarioCache(max_scenarios=2, ttl=0)
    cache.set(1, ("k",), "one")
    cache.set(2, ("k",), "two")
    assert cache.get(1, ("k",)) == "one"  # scenario 1 is now the most recent
    cache.set(3, ("k",), "three")  # evicts scenario 2
    assert cache.get(2, ("k",)) is None
    assert cache.get(1, ("k",)) == "one"
    assert cache.get(3, ("k",)) == "three"


def test_entry_expires_after_ttl(monkeypatch):
    cache = ScenarioCache(max_scenarios=2, ttl=10)
    now = [1000.0]
    monkeypatch.setattr(
        "src.idu_mcp.tools_services.scenario_cache.time.monotonic", lambda: now[0]
    )
    cache.set(1, ("k",), "one")
    now[0] += 5
    assert cache.get(1, ("k",)) == "one"
    now[0] += 6
    assert cache.get(1, ("k",)) is None


def test_size_from_env(monkeypatch):
    monkeypatch.setenv("SCENARIO_CACHE_SIZE", "10")
    assert ScenarioCache().max_scenarios == 10
    monkeypatch.setenv("SCENARIO_CACHE_SIZE", "not-a-number")
    assert not ScenarioCache().enabled


@pytest.mark.asyncio
async def test_repeat_query_is_served_from_cache(monkeypatch):
    monkeypatch.setenv("SCENARIO_CACHE_SIZE", "10")
    client = FakeUrbanClient()
    tool = UrbanApiTool(client)

    first = await tool.get_entity_by_names(
        846, ["Аптека"], ObjectTypeEnum.SERVICE, "token"
    )
    second = await tool.get_entity_by_names(
        846, ["Аптека"], ObjectTypeEnum.SERVICE, "token"
    )

    assert client.services_calls == 1
    assert client.name_id_calls == 1
    assert list(first) == list(second) == ["Аптека"]
    # models are rebuilt per call, so callers cannot mutate each other's data
    assert first["Аптека"] is not second["Аптека"]

    # a different scenario is a different key
    await tool.get_entity_by_names(5575, ["Аптека"], ObjectTypeEnum.SERVICE, "token")
    assert client.services_calls == 2


@pytest.mark.asyncio
async def test_cache_off_means_every_call_hits_the_api(monkeypatch):
    monkeypatch.delenv("SCENARIO_CACHE_SIZE", raising=False)
    client = FakeUrbanClient()
    tool = UrbanApiTool(client)
    for _ in range(3):
        await tool.get_entity_by_names(
            846, ["Аптека"], ObjectTypeEnum.SERVICE, "token"
        )
    assert client.services_calls == 3
