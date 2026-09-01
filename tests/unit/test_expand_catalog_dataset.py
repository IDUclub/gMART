"""Unit tests for the catalog-grounded dataset expansion.

Everything here is the part that decides *what* is asked; the generation call
itself is the model's business and is not exercised. The properties that matter
are that a question is never built on an entity the run cannot fetch, that the
two entity roles stay distinct, and that a generated line is only accepted when
it actually carries the triple it was generated for — an accepted line becomes
ground truth, so a loose check here silently corrupts the extended gold.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "harness"))

import expand_catalog_dataset as expand  # noqa: E402


def _write_entry(root: Path, scenario_id: int, endpoint: str, response: object) -> None:
    directory = root / str(scenario_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{abs(hash(endpoint)):x}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"endpoint": endpoint, "params": {}, "response": response}, handle)


@pytest.fixture()
def store(tmp_path: Path) -> Path:
    root = tmp_path / "urban_data"
    _write_entry(
        root,
        558,
        "/v1/scenarios/558/service_types",
        [{"name": "Школа"}, {"name": "Детская площадка"}],
    )
    _write_entry(
        root,
        558,
        "/v1/scenarios/558/physical_object_types",
        [{"name": "Жилой дом"}, {"name": "Местная дорога"}],
    )
    _write_entry(
        root,
        558,
        "/v1/scenarios/558/services_with_geometry",
        {"type": "FeatureCollection", "features": []},
    )
    return root


def test_catalog_from_store_splits_the_two_catalogs(store: Path):
    catalogs = expand.catalog_from_store(store, 558)

    assert catalogs[expand.SERVICE] == ["Школа", "Детская площадка"]
    assert catalogs[expand.PHYSICAL] == ["Жилой дом", "Местная дорога"]


def test_catalog_from_store_ignores_layer_entries(store: Path):
    """A layer response is a FeatureCollection, not a catalog; it must not leak in."""

    catalogs = expand.catalog_from_store(store, 558)

    assert "features" not in catalogs[expand.SERVICE]
    assert len(catalogs[expand.SERVICE]) + len(catalogs[expand.PHYSICAL]) == 4


def test_catalog_from_store_missing_scenario_is_empty(store: Path):
    assert expand.catalog_from_store(store, 999) == {
        expand.SERVICE: [],
        expand.PHYSICAL: [],
    }


def test_broken_entities_reads_names_and_whole_scenarios(tmp_path: Path):
    report = tmp_path / "gaps.json"
    report.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": 1738,
                        "failed": [
                            {"what": "GetPhysicalObjects", "name": "Местная дорога"}
                        ],
                    },
                    {"scenario_id": 420, "failed": [{"what": "catalog"}]},
                    {"scenario_id": 124, "failed": []},
                ]
            }
        ),
        encoding="utf-8",
    )

    broken = expand.broken_entities([report])

    assert broken[1738] == {"местная дорога"}
    # a catalog-level failure blocks the scenario outright
    assert broken[420] == {"*"}
    assert 124 not in broken


def test_draw_triples_excludes_blocked_entities(store: Path):
    catalogs = expand.catalog_from_store(store, 558)

    triples = expand.draw_triples(catalogs, {"местная дорога"}, 9, random.Random(1))

    assert triples
    names = {t["buffered_name"] for t in triples} | {t["counted_name"] for t in triples}
    assert "Местная дорога" not in names


def test_draw_triples_never_pairs_an_entity_with_itself(store: Path):
    catalogs = expand.catalog_from_store(store, 558)

    triples = expand.draw_triples(catalogs, set(), 9, random.Random(2))

    assert triples
    assert all(t["buffered_name"] != t["counted_name"] for t in triples)


def test_draw_triples_is_deterministic_for_a_seed(store: Path):
    catalogs = expand.catalog_from_store(store, 558)

    first = expand.draw_triples(catalogs, set(), 9, random.Random(7))
    second = expand.draw_triples(catalogs, set(), 9, random.Random(7))

    assert first == second


def test_draw_triples_needs_two_entities(tmp_path: Path):
    catalogs = {expand.SERVICE: ["Школа"], expand.PHYSICAL: []}

    assert expand.draw_triples(catalogs, set(), 9, random.Random(3)) == []


TRIPLE = {
    "buffered_name": "Школа",
    "buffered_kind": expand.SERVICE,
    "counted_name": "Жилой дом",
    "counted_kind": expand.PHYSICAL,
    "distance_m": 200,
}


def test_valid_accepts_an_inflected_mention():
    question = "Сколько жилых домов расположено в радиусе 200 м от школ?"

    assert expand.valid(question, TRIPLE)


def test_valid_rejects_a_polarity_inversion():
    question = "Сколько жилых домов находится за пределами 200 м от школ?"

    assert not expand.valid(question, TRIPLE)


def test_valid_rejects_a_changed_distance():
    question = "Сколько жилых домов расположено в радиусе 300 м от школ?"

    assert not expand.valid(question, TRIPLE)


def test_valid_rejects_a_dropped_entity():
    question = "Сколько объектов расположено в радиусе 200 м от школ?"

    assert not expand.valid(question, TRIPLE)


def _base_row() -> pd.Series:
    row = pd.Series(
        {
            expand.COL_PROJ: "Проект ИТМО Хайп-Парк",
            expand.COL_PROJ_ID: 558,
            expand.COL_SCEN_NAME: "Исходный",
            expand.COL_SID: 558,
            expand.COL_Q: "исходный вопрос",
        }
    )
    row.name = 3
    return row


def test_make_row_files_each_entity_under_its_own_catalog_column():
    row = expand.make_row(_base_row(), TRIPLE, "вопрос", variant=1)

    assert row[expand.COL_SN] == "Школа"
    assert row[expand.COL_PN] == "Жилой дом"


def test_make_row_carries_the_triple_as_ground_truth():
    row = expand.make_row(_base_row(), TRIPLE, "вопрос", variant=1)

    assert row["gen_buffered_entity"] == "Школа"
    assert row["gen_counted_entity"] == "Жилой дом"
    assert row["gen_distance_m"] == 200
    assert row["base_index"] == 3


def test_answer_text_invents_no_object_count():
    """The count is unknown for a generated question; a number here would be a

    fabricated ground truth, and gold_parser reads counts out of this field."""

    text = expand.answer_text("Проект", TRIPLE)

    import re

    numbers = set(re.findall(r"\d+", text))
    assert numbers == {"200"}


def test_valid_rejects_a_longer_catalog_name_that_also_matches():
    """«малых рек» matches «Река» on the stem but names «Малая река».

    Accepting it would file the row under the wrong buffered entity — a wrong
    number rather than a visible gap, which is the worse of the two failures."""

    triple = dict(TRIPLE, buffered_name="Река", buffered_kind=expand.PHYSICAL)
    question = "Сколько жилых домов расположено в радиусе 200 м от малых рек?"

    assert not expand.valid(question, triple, ["Река", "Малая река", "Жилой дом"])


def test_valid_keeps_an_unrelated_entity_mentioned_in_passing():
    triple = dict(TRIPLE, buffered_name="Река", buffered_kind=expand.PHYSICAL)
    question = "Сколько жилых домов у реки в радиусе 200 м, включая садовые дома?"

    assert expand.valid(question, triple, ["Река", "Садовый дом", "Жилой дом"])


# --------------------------------------------------------------------------- #
# Top-up: raising --n-questions must add, not replace
# --------------------------------------------------------------------------- #
def test_draw_triples_skips_triples_already_used_in_the_scenario(store: Path):
    """Base records share scenarios, so the used-set is what stops duplicates."""

    catalogs = expand.catalog_from_store(store, 558)
    used: set = set()

    first = expand.draw_triples(catalogs, set(), 5, random.Random(11), used)
    second = expand.draw_triples(catalogs, set(), 5, random.Random(11), used)

    keys = lambda ts: {  # noqa: E731
        (
            expand.norm(t["buffered_name"]),
            expand.norm(t["counted_name"]),
            t["distance_m"],
        )
        for t in ts
    }
    assert keys(first) & keys(second) == set()


def test_draw_triples_records_what_it_took_into_the_used_set(store: Path):
    catalogs = expand.catalog_from_store(store, 558)
    used: set = set()

    triples = expand.draw_triples(catalogs, set(), 4, random.Random(3), used)

    assert len(used) == len(triples)


def test_the_same_pair_at_another_radius_is_a_new_triple(store: Path):
    """Pairs alone run out on a small catalog; pairs x radii do not.

    Scenario 846 offers 132 ordered pairs and carries 8 base records, so at 18
    questions each the pair supply is exhausted while the triple supply is not."""

    catalogs = {expand.SERVICE: ["Школа"], expand.PHYSICAL: ["Жилой дом"]}
    used: set = set()

    triples = expand.draw_triples(catalogs, set(), 6, random.Random(5), used)

    # one ordered pair each way, six radii -> far more than the two pairs alone
    assert len(triples) > 2
    assert len({t["distance_m"] for t in triples}) > 1


def test_draw_triples_runs_out_gracefully(store: Path):
    """Asking for more than exists returns what exists, not a repeat."""

    catalogs = {expand.SERVICE: ["Школа"], expand.PHYSICAL: ["Жилой дом"]}
    used: set = set()

    triples = expand.draw_triples(catalogs, set(), 999, random.Random(5), used)

    keys = {
        (
            expand.norm(t["buffered_name"]),
            expand.norm(t["counted_name"]),
            t["distance_m"],
        )
        for t in triples
    }
    assert len(keys) == len(triples)
    assert len(triples) == 2 * len(expand.DISTANCES_M)
