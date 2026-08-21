import json
from pathlib import Path

import pytest

from src.agents.services.compliance_registry import DEFAULT_COMPLIANCE_REGISTRY

CASES = json.loads(
    (Path(__file__).parents[1] / "contract" / "check_plan_cases.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda item: item["name"])
def test_shared_check_plan_contract(case):
    if case["valid"]:
        DEFAULT_COMPLIANCE_REGISTRY.validate_plan(case["plan"])
    else:
        with pytest.raises(Exception):
            DEFAULT_COMPLIANCE_REGISTRY.validate_plan(case["plan"])
