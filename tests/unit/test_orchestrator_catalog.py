"""Unit tests for the orchestrator agent catalogue filtering."""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.services.orchestrator_catalog import available_agents
from src.agents.services.service_entities.orchestrator_plan import OrchestratorAgent


def config(
    dvd: str | None = "http://dvd",
    norms: str | None = "http://norms",
    urban: str | None = "http://urban-mcp",
):
    return SimpleNamespace(
        DVD_MCP_URL=dvd, NORM_GRAPH_MCP_URL=norms, URBAN_MCP_URL=urban
    )


def keys(agents) -> set[OrchestratorAgent]:
    return {entry.key for entry in agents}


def test_all_agents_available_with_scenario_and_urls():
    assert keys(available_agents(config(), scenario_id=772)) == {
        OrchestratorAgent.RESTRICTION,
        OrchestratorAgent.PROVISION,
        OrchestratorAgent.SCENARIO_DATA,
        OrchestratorAgent.DOCUMENTS,
        OrchestratorAgent.NORMS,
    }


def test_scenario_bound_agents_excluded_without_scenario():
    assert keys(available_agents(config(), scenario_id=None)) == {
        OrchestratorAgent.SCENARIO_DATA,
        OrchestratorAgent.DOCUMENTS,
        OrchestratorAgent.NORMS,
    }


def test_documents_excluded_without_dvd_url():
    assert OrchestratorAgent.DOCUMENTS not in keys(
        available_agents(config(dvd=None), scenario_id=772)
    )


def test_norms_excluded_without_normgraph_url():
    assert OrchestratorAgent.NORMS not in keys(
        available_agents(config(norms=None), scenario_id=772)
    )


def test_scenario_data_excluded_without_urban_mcp_url():
    assert OrchestratorAgent.SCENARIO_DATA not in keys(
        available_agents(config(urban=None), scenario_id=772)
    )


def test_no_agents_without_scenario_and_urls():
    assert (
        available_agents(config(dvd=None, norms=None, urban=None), scenario_id=None)
        == []
    )
