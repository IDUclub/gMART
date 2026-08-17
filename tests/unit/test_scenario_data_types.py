from src.agents.services.scenario_data_types import (
    ScenarioEntityKind,
    build_type_distribution,
    classify_type_query,
    distribution_answer,
    distribution_table,
)


class TestClassifyTypeQuery:
    def test_a_bare_objects_question_needs_clarification(self):
        intent = classify_type_query(
            "Какие объекты есть в сценарии и сколько их по типам?"
        )

        assert intent is not None
        assert intent.kinds == ()
        assert "физические объекты" in intent.clarification
        assert "сервисы" in intent.clarification

    def test_physical_objects_are_explicit(self):
        intent = classify_type_query("Сколько физических объектов в сценарии по типам?")

        assert intent.kinds == (ScenarioEntityKind.PHYSICAL_OBJECT,)

    def test_services_are_explicit(self):
        intent = classify_type_query("Сколько услуг каждого типа есть в сценарии?")

        assert intent.kinds == (ScenarioEntityKind.SERVICE,)

    def test_objects_and_services_request_both_tables(self):
        intent = classify_type_query(
            "Сколько объектов и сервисов есть в сценарии по типам?"
        )

        assert intent.kinds == (
            ScenarioEntityKind.PHYSICAL_OBJECT,
            ScenarioEntityKind.SERVICE,
        )

    def test_a_short_answer_resolves_the_previous_clarification(self):
        history = [
            {
                "role": "user",
                "content": "Какие объекты есть в сценарии и сколько их по типам?",
            },
            {
                "role": "assistant",
                "content": "Физические объекты, сервисы или оба набора?",
            },
        ]

        intent = classify_type_query("И те и те", history)

        assert intent.kinds == (
            ScenarioEntityKind.PHYSICAL_OBJECT,
            ScenarioEntityKind.SERVICE,
        )

    def test_both_sets_resolves_the_previous_clarification(self):
        history = [
            {
                "role": "user",
                "content": "Какие объекты есть в сценарии и сколько их по типам?",
            }
        ]

        intent = classify_type_query("Оба набора", history)

        assert intent.kinds == (
            ScenarioEntityKind.PHYSICAL_OBJECT,
            ScenarioEntityKind.SERVICE,
        )

    def test_an_unrelated_question_stays_on_the_general_agent_path(self):
        assert classify_type_query("Покажи показатели населения") is None


def _physical_type(type_id: int, name: str, function_id: int = 1) -> dict:
    return {
        "physical_object_type_id": type_id,
        "name": name,
        "physical_object_function": {"id": function_id, "name": "Здание"},
    }


def _physical_object(object_id: int, type_id: int, name: str) -> dict:
    return {
        "physical_object_id": object_id,
        "physical_object_type": _physical_type(type_id, name),
    }


class TestBuildTypeDistribution:
    def test_counts_unique_entities_and_omits_zero_types(self):
        entities = [
            _physical_object(1, 5, "Нежилое здание"),
            _physical_object(1, 5, "Нежилое здание"),
            _physical_object(2, 5, "Нежилое здание"),
        ]
        catalog = [
            _physical_type(5, "Нежилое здание"),
            _physical_type(6, "Жилое здание"),
        ]

        result = build_type_distribution(
            entities, catalog, ScenarioEntityKind.PHYSICAL_OBJECT
        )

        assert result.total_unique == 2
        assert result.available_types == 1
        assert result.present_types == 1
        assert result.rows == [
            {
                "type_id": 5,
                "type_name": "Нежилое здание",
                "count": 2,
                "status": "точное соответствие",
                "possible_types": "—",
            },
        ]

    def test_fallback_dictionary_resolves_an_id_exactly(self):
        result = build_type_distribution(
            [_physical_object(1, 7, "Площадка")],
            [_physical_type(5, "Нежилое здание")],
            ScenarioEntityKind.PHYSICAL_OBJECT,
            fallback_catalog_result=[_physical_type(7, "Площадка")],
        )

        row = next(row for row in result.rows if row["type_id"] == 7)
        assert row["type_name"] == "Площадка"
        assert row["status"] == "точное соответствие"

    def test_all_related_dictionary_candidates_are_shown(self):
        unknown = _physical_object(1, 999, "Неизвестный")
        catalog = [
            _physical_type(5, "Нежилое здание"),
            _physical_type(6, "Жилое здание"),
        ]

        result = build_type_distribution(
            [unknown], catalog, ScenarioEntityKind.PHYSICAL_OBJECT
        )

        row = next(row for row in result.rows if row["type_id"] == 999)
        assert row["status"] == "предположение"
        assert row["possible_types"] == "Жилое здание, Нежилое здание"


def test_distribution_presentation_is_text_plus_full_table():
    distribution = build_type_distribution(
        [_physical_object(1, 5, "Нежилое здание")],
        [_physical_type(5, "Нежилое здание"), _physical_type(6, "Жилое здание")],
        ScenarioEntityKind.PHYSICAL_OBJECT,
    )

    answer = distribution_answer(772, [distribution])
    table = distribution_table(distribution)

    assert "сценария 772" in answer
    assert "1 уникальный физический объект" in answer
    assert "представлено 1 тип" in answer
    assert len(table["rows"]) == 1
