"""Counts come from Python, not from a sample the model has to eyeball.

Regression guard for the reported failure: 924 physical objects were fetched, the model was
shown eight of them plus "… ещё 916", and answered that the types were unknown.
"""

from __future__ import annotations

from src.agents.services.scenario_data_aggregate import (
    aggregate_records,
    aggregate_result,
    extract_records,
)


def _objects(counts: dict[str, int]) -> list[dict]:
    rows = []
    index = 0
    for type_name, count in counts.items():
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "physical_object_id": index,
                    "name": f"объект {index}",
                    "physical_object_type": {"id": 1, "name": type_name},
                }
            )
    return rows


class TestAggregateRecords:
    def test_counts_every_record_not_a_sample(self):
        result = aggregate_records(_objects({"Жилой дом": 900, "Банк": 20, "Школа": 4}))

        assert result["total_records"] == 924
        counts = result["breakdown"]["physical_object_type.name"]["counts"]
        assert counts == {"Жилой дом": 900, "Банк": 20, "Школа": 4}

    def test_identifiers_are_not_a_breakdown(self):
        result = aggregate_records(_objects({"Жилой дом": 30}))

        assert "physical_object_id" not in result["breakdown"]
        assert "physical_object_type.id" not in result["breakdown"]

    def test_free_text_fields_are_skipped(self):
        """A per-row unique name is an identifier in disguise."""
        result = aggregate_records(_objects({"Жилой дом": 30}))

        assert "name" not in result["breakdown"]

    def test_long_tails_are_folded_into_other_values(self):
        # 60 types over 300 rows: a real catalogue, not a per-row unique value.
        rows = _objects({f"тип {i}": 5 for i in range(60)})

        entry = aggregate_records(rows)["breakdown"]["physical_object_type.name"]

        assert entry["distinct_values"] == 60
        assert len(entry["counts"]) == 30
        assert entry["other_values"] == 150  # the 30 types not listed, 5 rows each

    def test_a_value_unique_per_row_is_treated_as_an_identifier(self):
        """60 types across 60 rows is indistinguishable from an id, and is dropped."""
        rows = _objects({f"тип {i}": 1 for i in range(60)})

        assert aggregate_records(rows) is None

    def test_returns_none_when_nothing_is_categorical(self):
        rows = [{"id": i, "name": f"n{i}"} for i in range(20)]

        assert aggregate_records(rows) is None

    def test_empty_input(self):
        assert aggregate_records([]) is None


class TestExtractRecords:
    def test_plain_list(self):
        assert extract_records([{"a": 1}]) == [{"a": 1}]

    def test_wrapped_in_result(self):
        assert extract_records({"result": [{"a": 1}]}) == [{"a": 1}]

    def test_geojson_features_use_their_properties(self):
        collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {"physical_object_type": {"name": "Банк"}},
                }
            ],
        }

        assert extract_records(collection) == [
            {"physical_object_type": {"name": "Банк"}}
        ]

    def test_non_record_payloads_are_declined(self):
        assert extract_records({"total": 5}) is None
        assert extract_records([1, 2, 3]) is None
        assert extract_records([]) is None


class TestAggregateResult:
    def test_geometry_never_reaches_the_breakdown(self):
        collection = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
                    "properties": {"physical_object_type": {"name": "Банк"}},
                }
                for _ in range(5)
            ],
        }

        result = aggregate_result(collection)

        assert result["total_records"] == 5
        assert result["breakdown"]["physical_object_type.name"]["counts"] == {"Банк": 5}
        assert not any("geometry" in key for key in result["breakdown"])
