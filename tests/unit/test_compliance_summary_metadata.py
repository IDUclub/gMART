"""The downloadable summary describes territory documents in plain language."""

import pytest

from src.agents.services.compilance.compliance_archive import build_summary_report


@pytest.mark.parametrize(
    "territory,expected",
    [
        (
            {"documents_in_force": 15, "documents_out_of_force": 2},
            "Документы на территории: действуют — 15; не действуют — 2.",
        ),
        (
            {"documents_in_force": 0, "documents_out_of_force": 0},
            "Документы на территории: действуют — 0; не действуют — 0.",
        ),
        ({}, "Документы на территории: сведения отсутствуют."),
        (None, "Документы на территории: сведения отсутствуют."),
    ],
)
def test_territory_documents_are_human_readable(territory, expected):
    report = build_summary_report({"territory": territory, "results": []})

    assert expected in report
    assert "documents_in_force" not in report
    assert "documents_out_of_force" not in report
