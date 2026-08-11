"""Regression tests for gold entity extraction.

Run with:  .venv/bin/python3 -m pytest benchmarks/eval/test_gold_parser.py -q

`benchmarks/` is untracked, so these live here rather than in tests/unit — the
repo's CI must not depend on them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from gold_parser import load_gold, parse_record  # noqa: E402

GOLD = "benchmarks/data/gold/exp_data.csv"
LAYERS_COL = (
    "Слои (ответ) (*здесь краткое описание о спецификации содержимого в слое "
    "+файлы geojson (СК epsg:4326)"
)


def _row(spec: str, services: str = "", phys: str = "") -> pd.Series:
    return pd.Series(
        {
            "scenario_id": 1,
            "Наименование проекта": "test",
            "Промт (вопрос)": "тестовый вопрос",
            "Промт (ответ)": "в радиусе 500 метров находится 3 объекта",
            LAYERS_COL: spec,
            "service_names": services,
            "phys_names": phys,
        }
    )


def test_buffer_clause_first_does_not_collapse_both_roles():
    """The spec often puts the buffered entity first; source must still come
    from the other clause, not from clauses[0]."""

    rec = parse_record(
        _row(
            "1 слой с буферной зоной 500 метров от школы,"
            "1 слой со спортивными площадками",
            services="Школа, Спортивная площадка",
        ),
        LAYERS_COL,
    )

    assert rec.target_entity == "Школа"           # the buffered entity
    assert rec.source_entity == "Спортивная площадка"
    assert rec.source_entity != rec.target_entity


def test_buffer_clause_second_keeps_working():
    rec = parse_record(
        _row(
            "1 - слой с жилыми домами, 2 - слой с водными объектами, "
            "учитывающий буфер 200 метров",
            phys="Жилой дом, Водный объект",
        ),
        LAYERS_COL,
    )

    assert rec.source_entity == "Жилой дом"
    assert rec.target_entity == "Водный объект"


def test_unresolvable_entity_is_flagged_not_silently_trusted():
    """A clause that matches neither catalog nor lexicon still yields a string,
    but must not be marked as catalog-resolved — the evaluator skips those."""

    rec = parse_record(
        _row("1 слой с отсановками наземного транспорта и них, 1 слой с чем-то"),
        LAYERS_COL,
    )

    assert not rec.conf["source_in_catalog"] or not rec.conf["target_in_catalog"]


@pytest.mark.skipif(not Path(GOLD).exists(), reason="gold CSV not present")
def test_real_gold_no_longer_collapses_roles():
    gold = load_gold(GOLD)
    collapsed = [g for g in gold if g.source_entity and g.source_entity == g.target_entity]
    # was 89/202 before the fix; a couple of genuinely self-referential specs remain
    assert len(collapsed) <= 5, f"{len(collapsed)} records still collapse both roles"
