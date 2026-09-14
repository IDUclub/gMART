"""Acceptance must reject missing deliverables, even when numbers are correct."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.integration.industrial.acceptance import verify_result
from tests.integration.industrial.control import entities
from tests.unit.test_harness_source_oracle import records
from tests.unit.test_industrial_acceptance import sample


def add(context, identifier, kind, content):
    context["artifacts"].append(
        {
            "id": identifier,
            "request_id": "r",
            "step": 1,
            "confirmed": True,
            "kind": kind,
            "content": content,
        }
    )


def test_source_map_without_the_requested_table_is_not_accepted():
    final, context = sample()
    layer = entities(91001, "physical_object", 8)
    add(
        context,
        "park",
        "feature_collection",
        {"name": "Парк", "feature_collection": layer},
    )
    contract = {
        "source_layers": [
            {"scenario_id": 91001, "domain": "physical_object", "type_id": 8}
        ]
    }
    assert not verify_result(final, context, contract)["passed"]
    add(
        context,
        "park-table",
        "table",
        {
            "rows": [deepcopy(f["properties"]) for f in layer["features"]],
            "complete": True,
        },
    )
    assert verify_result(final, context, contract)["passed"]
    context["artifacts"][-1]["content"]["rows"][0]["source_version"] = "stale"
    assert not verify_result(final, context, contract)["passed"]


def test_correct_compliance_count_requires_the_full_result_layer():
    final, context = sample()
    for system, sources in records().items():
        add(context, system, "source_evidence", {"system": system, "sources": sources})
    result = {
        "restriction_id": "restriction",
        "verification_status": "complete",
        "compliance_status": "violated",
        "coverage": {
            "applicable_objects": 1,
            "checked_objects": 1,
            "unchecked_objects": 0,
        },
        "summary": {"violated_objects": 1, "passed_objects": 0},
    }
    add(context, "compliance", "compliance_result", result)
    contract = {
        "sources": True,
        "compliance": [{"scenario_id": 91001, "violations": 1}],
    }
    assert not verify_result(final, context, contract)["passed"]
    layer = entities(91001, "physical_object", 7)
    for feature in layer["features"]:
        feature["properties"].update(
            restriction_id="restriction", compliance_status="violated"
        )
    result.update(
        violated_features=layer,
        passed_features={"type": "FeatureCollection", "features": []},
    )
    assert verify_result(final, context, contract)["passed"]
    layer["features"][0]["geometry"] = entities(91001, "physical_object", 8)[
        "features"
    ][0]["geometry"]
    assert not verify_result(final, context, contract)["passed"]


def test_outer_harness_cannot_overwrite_an_existing_series(tmp_path, monkeypatch):
    import sys

    from tests.integration.local_stack import harness

    previous = tmp_path / "live-20.log"
    previous.write_text("original failure", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--mode",
            "live",
            "--env-file",
            "unused",
            "--output",
            str(tmp_path),
        ],
    )
    monkeypatch.setattr(
        harness.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    with pytest.raises(FileExistsError):
        harness.main()
    assert previous.read_text(encoding="utf-8") == "original failure"
