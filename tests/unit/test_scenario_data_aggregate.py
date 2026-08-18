"""Counts come from Python, not from a sample the model has to eyeball.

Regression guard for the reported failure: 924 physical objects were fetched, the model was
shown eight of them plus "… ещё 916", and answered that the types were unknown.
"""

from __future__ import annotations

from src.agents.services.scenario_data_aggregate import (
    aggregate_records,
    aggregate_result,
    answer_records,
    bounded_observation_context,
    extract_records,
    unresolved_references,
)


def _objects(counts: dict[str, int]) -> list[dict]:
    rows = []
    index = 0
    for type_id, (type_name, count) in enumerate(counts.items(), start=1):
        for _ in range(count):
            index += 1
            rows.append(
                {
                    "physical_object_id": index,
                    "name": f"объект {index}",
                    "physical_object_type": {"id": type_id, "name": type_name},
                }
            )
    return rows


class TestAggregateRecords:
    def test_counts_every_record_not_a_sample(self):
        result = aggregate_records(_objects({"Жилой дом": 900, "Банк": 20, "Школа": 4}))

        assert result["total_records"] == 924
        counts = result["breakdown"]["physical_object_type.name"]["counts"]
        assert counts == {"Жилой дом": 900, "Банк": 20, "Школа": 4}

    def test_row_identifiers_are_not_a_breakdown(self):
        """One value per row carries no distribution."""
        result = aggregate_records(_objects({"Жилой дом": 30}))

        assert "physical_object_id" not in result["breakdown"]

    def test_a_foreign_key_is_kept_as_a_join_hint(self):
        """`service_type_id`-style keys are the join keys the planner needs to resolve.

        Dropping them wholesale is what hid the very field a "which types" question turns on.
        """
        result = aggregate_records(_objects({"Жилой дом": 20, "Банк": 10}))

        assert result["breakdown"]["physical_object_type.id"]["distinct_values"] == 2

    def test_a_category_outranks_a_near_constant_field(self):
        """Ranking by fewest-distinct alone let booleans crowd the type out of the list."""
        rows = [
            dict(row, is_capacity_real=True, some_flag=False, other_flag=True)
            for row in _objects({f"тип {i}": 3 for i in range(10)})
        ]

        breakdown = aggregate_records(rows)["breakdown"]

        assert "physical_object_type.name" in breakdown
        names = list(breakdown)
        assert names.index("physical_object_type.name") < names.index(
            "is_capacity_real"
        )

    def test_free_text_fields_are_skipped(self):
        """A per-row unique name is an identifier in disguise."""
        result = aggregate_records(_objects({"Жилой дом": 30}))

        assert "name" not in result["breakdown"]

    def test_long_tails_keep_every_available_type(self):
        # 60 types over 300 rows: a real catalogue, not a per-row unique value.
        rows = _objects({f"тип {i}": 5 for i in range(60)})

        entry = aggregate_records(rows)["breakdown"]["physical_object_type.name"]

        assert entry["distinct_values"] == 60
        assert len(entry["counts"]) == 60
        assert "other_values" not in entry

    def test_a_type_value_unique_per_row_is_still_a_real_category(self):
        rows = _objects({f"тип {i}": 1 for i in range(60)})

        breakdown = (aggregate_records(rows) or {}).get("breakdown", {})

        assert len(breakdown["physical_object_type.name"]["counts"]) == 60

    def test_duplicate_entity_rows_are_counted_once(self):
        rows = _objects({"Школа": 2, "Банк": 1})

        result = aggregate_records([rows[0], rows[0], rows[1], rows[2]])

        assert result["total_records"] == 3
        assert result["breakdown"]["physical_object_type.name"]["counts"] == {
            "Школа": 2,
            "Банк": 1,
        }

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


def test_small_type_catalogue_is_preserved_for_the_final_answer():
    rows = [
        {"service_type_id": index, "name": f"Тип {index}", "capacity": 100}
        for index in range(1, 25)
    ]

    assert answer_records(rows) == [
        {"service_type_id": index, "name": f"Тип {index}"} for index in range(1, 25)
    ]


def test_large_entity_collection_is_not_copied_into_model_context():
    rows = [
        {"physical_object_id": index, "name": f"Объект {index}"} for index in range(101)
    ]

    assert answer_records(rows) is None


def test_bounded_context_keeps_recent_evidence_and_valid_json():
    context = bounded_observation_context(
        [
            {"mapping": {"result": "x" * 50000, "source_tool": "old"}},
            {
                "tool": "projects.GetScenarioServiceTypes",
                "answer_records": [{"service_type_id": 22, "name": "Школа"}],
            },
        ],
        max_chars=1000,
    )

    assert '"name": "Школа"' in context
    assert "x" * 100 not in context
    assert context.startswith('{"order": "most_recent_first"')


def test_bounded_context_marks_a_table_already_sent_to_the_client():
    context = bounded_observation_context(
        [{"table_count": 1, "aggregate": {"total_records": 70}}], max_chars=1000
    )

    assert '"table_count": 1' in context


class TestUnresolvedReferences:
    def test_a_bare_foreign_key_needs_a_lookup(self):
        rows = [{"service_id": i, "service_type_id": i % 12} for i in range(900)]

        aggregate = aggregate_records(rows)

        assert unresolved_references(aggregate) == ["service_type_id"]

    def test_a_key_with_its_name_alongside_needs_nothing(self):
        rows = [
            {"service_id": i, "service_type": {"id": i % 3, "name": f"тип {i % 3}"}}
            for i in range(90)
        ]

        aggregate = aggregate_records(rows)

        assert unresolved_references(aggregate) == []

    def test_no_aggregate_means_nothing_pending(self):
        assert unresolved_references(None) == []
