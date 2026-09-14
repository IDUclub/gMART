"""Captured ChatStorage typed contracts, exercised against actual emitted events."""

import json
from pathlib import Path

import jsonschema
import pytest

from src.agents.services.orchestrator.analysis import artifact_parts
from src.agents.services.orchestrator.analysis_context import AnalysisContext

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.parametrize(
    "kind",
    ["check_plan", "requirement_resolution", "compliance_result", "compliance_summary"],
)
def test_confirmed_compliance_artifact_satisfies_chatstorage_contract(kind):
    events = json.loads(
        (FIXTURES / "compliance_storage_events.json").read_text(encoding="utf-8")
    )
    contracts = json.loads(
        (FIXTURES / "chatstorage_compliance_contract.json").read_text(encoding="utf-8")
    )
    event = next(e for e in events if e["type"] == kind)
    context = AnalysisContext()
    aid = context.add_artifact(event, 1, "compliance-request")
    context.finish(
        1,
        "Проверить норму",
        772,
        "completed",
        "Проверка завершена",
        "compliance-request",
    )
    parts = artifact_parts(context)
    assert len(parts) == 1
    wire = parts[0].model_dump(mode="json", exclude_none=True)
    assert wire["kind"] == kind
    jsonschema.validate(wire["payload"], contracts[kind])
    assert context.get(aid)["content"] == event["content"]
