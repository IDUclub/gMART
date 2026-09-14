"""A resolvable source from the wrong clause must never pass live acceptance."""

from copy import deepcopy

import pytest

from tests.integration.local_stack.source_contract import verify_source_records


def records():
    return {
        "documents": [
            {
                "id": "clause",
                "doc_id": "doc",
                "name": "LOCAL SDK TEST",
                "version": "2026",
                "version_id": "version",
                "numbering": "1.1",
                "text": "Расстояние от здания школы до открытой автомобильной стоянки должно быть не менее 50 метров.",
            }
        ],
        "norms": [
            {
                "id": "restriction",
                "subject": "здание школы",
                "object": "открытая автомобильная стоянка",
                "value": {"number": 50, "operator": ">=", "unit": "м"},
                "provenance": {
                    "doc_id": "doc",
                    "clause_node_id": "clause",
                    "name": "LOCAL SDK TEST",
                    "version": "2026",
                    "version_id": "version",
                },
            }
        ],
    }


def test_source_oracle_accepts_matching_original_clause_and_restriction():
    verify_source_records(records())


@pytest.mark.parametrize(
    "path,value",
    [
        (("provenance", "clause_node_id"), "other-clause"),
        (("provenance", "doc_id"), "other-document"),
        (("provenance", "version_id"), "older-version"),
        (("value", "number"), 500),
        (("value", "operator"), "<="),
        (("value", "unit"), "км"),
        (("subject",), "больница"),
        (("object",), "магазин"),
    ],
)
def test_source_oracle_rejects_resolvable_but_wrong_restriction(path, value):
    sources = deepcopy(records())
    target = sources["norms"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        verify_source_records(sources)
