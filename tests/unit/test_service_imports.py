"""Services must load with the repository root on sys.path, as in deployment."""

import subprocess
import sys
from pathlib import Path


def test_service_packages_import_from_repository_root(tmp_path):
    # Isolate from pytest's import cache, PYTHONPATH and developer path overrides.
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import importlib
import sys

sys.path.insert(0, sys.argv[1])
for module in (
    "compilance.compliance_executor",
    "dvd.dvd_rag_service",
    "normgraph.normgraph_rag_service",
    "orchestrator.orchestrator_service",
    "provision.provsion_service",
    "restriction.restriction_parser_service",
    "scenario_data.scenario_data_service",
    "synapse.synapse_gateway_service",
):
    importlib.import_module("src.agents.services." + module)

assert not any(name == "agents" or name.startswith("agents.") for name in sys.modules)
""",
            str(repo_root),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
