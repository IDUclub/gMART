"""Tests for the per-scenario Urban API response cache used by benchmark runs.

The cache lives on disk: a scenario's objects run to tens of megabytes as Python
objects, and keeping several scenarios resident starved the host of memory.

Covers:
  * the cache being off unless ``SCENARIO_CACHE_SIZE`` is set;
  * LRU eviction once more scenarios than the limit are touched;
  * TTL expiry;
  * entries stored compressed, readable by a fresh instance, and a damaged entry
    counting as a miss rather than an error;
  * ``UrbanApiTool.get_entity_by_names`` serving a repeat call without hitting
    the Urban API client again, while still returning independent models.
"""

from __future__ import annotations

import gzip
import json
import os

import pytest

from src.idu_mcp.tools_services.entites.object_type_enum import ObjectTypeEnum
from src.idu_mcp.tools_services.scenario_cache import ScenarioCache
from src.idu_mcp.tools_services.urb_api_tools import UrbanApiTool


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Never let a test write into the working tree."""

    monkeypatch.setenv("SCENARIO_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


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


def test_entry_expires_after_ttl(cache_dir):
    """Age is the entry file's own mtime, so it outlives the process."""

    cache = ScenarioCache(max_scenarios=2, ttl=10)
    cache.set(1, ("k",), "one")
    assert cache.get(1, ("k",)) == "one"

    entry = next((cache_dir / "1").glob("*.json.gz"))
    stale = os.stat(entry).st_mtime - 11
    os.utime(entry, (stale, stale))

    assert cache.get(1, ("k",)) is None
    assert not entry.exists()


def test_entries_are_stored_compressed_and_outlive_the_instance(cache_dir):
    payload = {"features": [{"name": "жилой дом"} for _ in range(500)]}
    ScenarioCache(max_scenarios=2, ttl=0).set(846, ("k",), payload)

    entry = next((cache_dir / "846").glob("*.json.gz"))
    assert entry.stat().st_size < len(json.dumps(payload, ensure_ascii=False)) / 5
    with gzip.open(entry, "rt", encoding="utf-8") as fh:
        assert json.load(fh) == payload

    # a fresh instance — a restarted container — reads the same entry
    assert ScenarioCache(max_scenarios=2, ttl=0).get(846, ("k",)) == payload


def test_a_damaged_entry_is_a_miss(cache_dir):
    cache = ScenarioCache(max_scenarios=2, ttl=0)
    cache.set(1, ("k",), "one")
    entry = next((cache_dir / "1").glob("*.json.gz"))
    entry.write_bytes(b"not gzip at all")

    assert cache.get(1, ("k",)) is None
    assert not entry.exists()


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
        await tool.get_entity_by_names(846, ["Аптека"], ObjectTypeEnum.SERVICE, "token")
    assert client.services_calls == 3
