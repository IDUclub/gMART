"""Unit tests for the gold-set filter.

The filter deletes expert records, so the tests that matter are the ones about
*not* deleting the wrong ones. Two failure modes are pinned in both directions:
an inflected or misspelt phrase must still reach its catalog lemma, and two
different catalog entries that merely look alike («парк» / «парковка») must not
be merged — the first would drop valid rows, the second would keep a row under
the wrong ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "eval"))

import restrictions_filter as rf  # noqa: E402


class _Record:
    """The handful of GoldRecord fields the classifier reads."""

    def __init__(
        self,
        question: str,
        *,
        intent: str = "restrictions",
        source_entity: str | None = "Жилой дом",
        target_entity: str | None = "Школа",
        distance_m: int | None = 200,
    ) -> None:
        self.question = question
        self.intent = intent
        self.source_entity = source_entity
        self.target_entity = target_entity
        self.distance_m = distance_m


CATALOG = ["Жилой дом", "Школа", "Парк", "Парковка", "Остановка наземного транспорта"]


# --------------------------------------------------------------- resolution --
def test_resolve_matches_a_catalog_entry_literally():
    assert rf.resolve("Школа", CATALOG) == "Школа"


def test_resolve_reaches_a_synonym_through_the_alias_table():
    """«жилыми зданиями» is a synonym of «жилой дом», not an inflection of it."""

    assert rf.resolve("жилыми зданиями", CATALOG) == "Жилой дом"


def test_resolve_survives_the_experts_typo():
    assert (
        rf.resolve("отсановками наземного транспорта", CATALOG)
        == "Остановка наземного транспорта"
    )


def test_resolve_strips_layer_spec_debris():
    assert rf.resolve("- остановок", CATALOG) == "Остановка наземного транспорта"
    assert rf.resolve("со школами", CATALOG) == "Школа"


def test_resolve_keeps_park_and_parking_apart():
    """The pair that breaks every loose matcher: 0.67 similarity, both real."""

    assert rf.resolve("парк", CATALOG) == "Парк"
    assert rf.resolve("парковок", CATALOG) == "Парковка"


def test_resolve_returns_none_when_the_entity_is_absent():
    assert rf.resolve("Кладбище", CATALOG) is None
    assert rf.resolve("водный объект", CATALOG) is None


def test_resolve_ignores_an_empty_name():
    assert rf.resolve(None, CATALOG) is None
    assert rf.resolve("", CATALOG) is None


def test_aliases_hold_no_identity_entries():
    """An identity alias collides with its neighbours under the coarse signature.

    «парковка» -> «парковка» would make «парк» resolve to «Парковка»."""

    assert not [key for key, value in rf.ALIASES.items() if key == value]


# ------------------------------------------------------------ classification --
def test_a_plain_restrictions_question_is_kept():
    record = _Record("Сколько жилых домов в радиусе 200 м от школ?")

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.KEEP


def test_a_share_question_is_kept():
    """Phrasing is not the operation: this is still buffer-and-contain."""

    record = _Record("Покажи процент жилых домов в радиусе 200 м от школ.")

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.KEEP


def test_the_complement_is_dropped():
    record = _Record("Сколько жилых домов вне радиуса 200 м от школ?")

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.COMPLEMENT


def test_a_negated_provision_question_is_the_complement_too():
    record = _Record("Какие дома не оснащены школами в радиусе 200 м?")

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.COMPLEMENT


def test_a_travel_time_radius_is_dropped():
    record = _Record("Сколько жилых домов в 10 минутах ходьбы от школ?")

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.TIME_DISTANCE


def test_a_missing_radius_is_dropped():
    record = _Record("Сколько жилых домов рядом со школами?", distance_m=None)

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.NO_DISTANCE


def test_an_absent_entity_is_dropped_and_named():
    record = _Record(
        "Сколько жилых домов в радиусе 200 м от кладбищ?", target_entity="Кладбище"
    )

    reason, detail = rf.classify(record, CATALOG, set())

    assert reason == rf.ENTITY_ABSENT
    assert "buffered" in detail


def test_an_unfetchable_entity_is_data_unavailable_not_a_model_failure():
    record = _Record("Сколько жилых домов в радиусе 200 м от школ?")

    reason, _ = rf.classify(record, CATALOG, {"школа"})

    assert reason == rf.DATA_UNAVAILABLE


def test_an_empty_catalog_means_the_scenario_is_unreachable():
    record = _Record("Сколько жилых домов в радиусе 200 м от школ?")

    reason, _ = rf.classify(record, [], set())

    assert reason == rf.DATA_UNAVAILABLE


def test_a_gold_clarification_row_is_not_a_restrictions_task():
    record = _Record(
        "Проверь, попадают ли здания в зоны ограничений.",
        intent="needs_clarification",
    )

    reason, _ = rf.classify(record, CATALOG, set())

    assert reason == rf.NO_DISTANCE


@pytest.mark.parametrize(
    "path",
    [
        Path("runtime/urban_data/gaps_token.json"),
        Path("runtime/urban_data/does-not-exist.json"),
    ],
)
def test_blocked_entities_tolerates_a_missing_report(path: Path):
    """A gap report is optional; its absence must not silently drop everything."""

    assert isinstance(rf.blocked_entities([path]), dict)
