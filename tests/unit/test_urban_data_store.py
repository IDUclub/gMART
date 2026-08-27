"""Unit tests for the offline Urban API data store.

The store exists so an experiment run survives a dropped VPN and stays
reproducible across arms. Two properties carry that and are tested here:

* an entry is keyed per single request, so any *combination* of entity names a
  later query asks for is served from pieces already on disk — the thing
  ``ScenarioCache`` (keyed on the whole tuple of names one call asked for)
  cannot do;
* a miss in ``replay`` raises instead of falling through to the network or to an
  empty result, so a data gap can never be scored as a model failure.
"""

from __future__ import annotations

import gzip
import json

import pytest

from src.idu_mcp.common.api_handlers.json_api_handler import JsonApiHandler
from src.idu_mcp.common.api_handlers.urban_data_store import (
    LIVE,
    RECORD,
    REPLAY,
    UrbanDataStore,
    UrbanDataUnavailable,
    store_mode,
)


class FakeResponse:
    def __init__(self, status: int, json_body=None) -> None:
        self.status = status
        self._json = json_body
        self.content_type = "application/json"

    async def json(self, content_type=None):
        return self._json

    async def text(self):
        return ""


class FakeReqCtx:
    def __init__(self, outcome) -> None:
        self._outcome = outcome

    async def __aenter__(self):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Records every request so a test can assert the network was not used."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[tuple] = []

    def get(self, url=None, headers=None, params=None):
        self.requests.append((url, dict(params or {})))
        return FakeReqCtx(self._outcomes.pop(0))


def _handler(tmp_path, mode: str) -> JsonApiHandler:
    return JsonApiHandler(
        "http://urban",
        max_retries=1,
        backoff_base=0,
        store=UrbanDataStore(mode=mode, root=tmp_path),
    )


# --------------------------------------------------------------------------- #
# modes
# --------------------------------------------------------------------------- #
def test_live_mode_neither_reads_nor_writes(tmp_path):
    store = UrbanDataStore(mode=LIVE, root=tmp_path)
    store.set("http://urban", "v1/service_types", {"name": "школа"}, [{"id": 1}])

    assert not store.enabled
    assert list(tmp_path.rglob("*.json.gz")) == []
    with pytest.raises(KeyError):
        store.get("http://urban", "v1/service_types", {"name": "школа"})


def test_replay_mode_does_not_write(tmp_path):
    store = UrbanDataStore(mode=REPLAY, root=tmp_path)
    store.set("http://urban", "v1/service_types", {"name": "школа"}, [{"id": 1}])

    assert list(tmp_path.rglob("*.json.gz")) == []


def test_unknown_mode_falls_back_to_live(monkeypatch):
    monkeypatch.setenv("URBAN_DATA_MODE", "nonsense")
    assert store_mode() == LIVE


def test_mode_defaults_to_live_when_unset(monkeypatch):
    monkeypatch.delenv("URBAN_DATA_MODE", raising=False)
    assert store_mode() == LIVE
    assert not UrbanDataStore().enabled


# --------------------------------------------------------------------------- #
# keying
# --------------------------------------------------------------------------- #
def test_param_order_and_types_do_not_change_the_key(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set(
        "http://urban",
        "v1/scenarios/7/services_with_geometry",
        {"service_type_id": 3, "page": 1},
        {"features": []},
    )

    # same request, dict built in the other order and the id as a string
    assert store.get(
        "http://urban",
        "v1/scenarios/7/services_with_geometry",
        {"page": "1", "service_type_id": "3"},
    ) == {"features": []}


def test_trailing_slash_in_endpoint_does_not_change_the_key(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set("http://urban/", "/v1/service_types", {"name": "школа"}, [{"id": 1}])

    assert store.get("http://urban", "v1/service_types", {"name": "школа"}) == [
        {"id": 1}
    ]


def test_scenario_requests_are_filed_by_scenario(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set("http://urban", "v1/scenarios/42/service_types", None, ["школа"])
    store.set("http://urban", "v1/service_types", {"name": "школа"}, [{"id": 1}])

    assert store.scenarios() == [42]
    assert (tmp_path / "42").is_dir()
    assert (tmp_path / "_global").is_dir()


def test_stored_entry_records_the_request_it_answers(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set("http://urban", "v1/service_types", {"name": "школа"}, [{"id": 1}])

    entry_file = next(tmp_path.rglob("*.json.gz"))
    with gzip.open(entry_file, "rt", encoding="utf-8") as handle:
        entry = json.load(handle)
    assert entry["endpoint"] == "/v1/service_types"
    assert entry["params"] == {"name": "школа"}
    assert entry["response"] == [{"id": 1}]


def test_null_response_is_distinguishable_from_a_miss(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set("http://urban", "v1/service_types", {"name": "нет"}, None)

    assert store.get("http://urban", "v1/service_types", {"name": "нет"}) is None
    with pytest.raises(KeyError):
        store.get("http://urban", "v1/service_types", {"name": "другое"})


def test_truncated_entry_is_a_miss_and_is_dropped(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set("http://urban", "v1/service_types", {"name": "школа"}, [{"id": 1}])
    entry_file = next(tmp_path.rglob("*.json.gz"))
    entry_file.write_bytes(b"not gzip at all")

    with pytest.raises(KeyError):
        store.get("http://urban", "v1/service_types", {"name": "школа"})
    assert not entry_file.exists()


# --------------------------------------------------------------------------- #
# handler integration
# --------------------------------------------------------------------------- #
async def test_record_writes_and_replay_serves_without_the_network(tmp_path):
    recorder = _handler(tmp_path, RECORD)
    session = FakeSession([FakeResponse(200, [{"service_type_id": 3}])])

    recorded = await recorder.get(
        "v1/service_types", params={"name": "школа"}, session=session
    )
    assert recorded == [{"service_type_id": 3}]
    assert len(session.requests) == 1

    replayer = _handler(tmp_path, REPLAY)
    offline_session = FakeSession([])  # any use of it would IndexError
    replayed = await replayer.get(
        "v1/service_types", params={"name": "школа"}, session=offline_session
    )

    assert replayed == [{"service_type_id": 3}]
    assert offline_session.requests == []


async def test_replay_miss_raises_rather_than_returning_empty(tmp_path):
    replayer = _handler(tmp_path, REPLAY)

    with pytest.raises(UrbanDataUnavailable) as excinfo:
        await replayer.get(
            "v1/scenarios/7/services_with_geometry",
            params={"service_type_id": 3},
            session=FakeSession([]),
        )

    # The runner records which data was missing, so it carries the request.
    assert excinfo.value.endpoint == "v1/scenarios/7/services_with_geometry"
    assert excinfo.value.params == {"service_type_id": 3}


async def test_record_serves_a_second_identical_request_from_disk(tmp_path):
    recorder = _handler(tmp_path, RECORD)
    session = FakeSession([FakeResponse(200, {"features": []})])

    for _ in range(2):
        await recorder.get(
            "v1/scenarios/7/services_with_geometry",
            params={"service_type_id": 3},
            session=session,
        )

    assert len(session.requests) == 1
    assert recorder.store.hits == 1
    assert recorder.store.writes == 1


async def test_any_combination_of_names_is_served_from_per_entity_entries(tmp_path):
    """The property ScenarioCache's tuple-of-names key cannot provide.

    A run that fetched "школа" and "парк" separately must answer a later query
    asking for both without touching the network.
    """

    recorder = _handler(tmp_path, RECORD)
    session = FakeSession(
        [
            FakeResponse(200, [{"service_type_id": 1}]),
            FakeResponse(200, [{"service_type_id": 2}]),
        ]
    )
    await recorder.get("v1/service_types", params={"name": "Школа"}, session=session)
    await recorder.get("v1/service_types", params={"name": "Парк"}, session=session)

    replayer = _handler(tmp_path, REPLAY)
    offline = FakeSession([])
    both = [
        await replayer.get("v1/service_types", params={"name": name}, session=offline)
        for name in ("Школа", "Парк")
    ]

    assert both == [[{"service_type_id": 1}], [{"service_type_id": 2}]]
    assert offline.requests == []


async def test_live_handler_always_hits_the_network(tmp_path):
    handler = _handler(tmp_path, LIVE)
    session = FakeSession([FakeResponse(200, [1]), FakeResponse(200, [1])])

    for _ in range(2):
        await handler.get("v1/service_types", params={"name": "школа"}, session=session)

    assert len(session.requests) == 2
    assert list(tmp_path.rglob("*.json.gz")) == []


async def test_boolean_params_key_the_same_in_record_and_replay(tmp_path):
    """``_check_request_params`` rewrites bools in place; the key must not move."""

    recorder = _handler(tmp_path, RECORD)
    session = FakeSession([FakeResponse(200, {"ok": True})])
    await recorder.get(
        "v1/scenarios/7/services_with_geometry",
        params={"service_type_id": 3, "centers_only": True},
        session=session,
    )

    replayer = _handler(tmp_path, REPLAY)
    replayed = await replayer.get(
        "v1/scenarios/7/services_with_geometry",
        params={"service_type_id": 3, "centers_only": True},
        session=FakeSession([]),
    )

    assert replayed == {"ok": True}


def test_stats_report_the_corpus(tmp_path):
    store = UrbanDataStore(mode=RECORD, root=tmp_path)
    store.set("http://urban", "v1/scenarios/7/service_types", None, ["школа"])
    store.set("http://urban", "v1/scenarios/8/service_types", None, ["парк"])

    stats = store.stats()
    assert stats["entries"] == 2
    assert stats["scenarios"] == 2
    assert stats["mode"] == RECORD
