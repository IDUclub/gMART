import pytest

from src.agents.a2a.a2a_format import synapse_prepopulated_params_agent_card
from src.agents.a2a.agent import RestrictionA2AAgent
from src.agents.a2a.dvd_agent import DocumentQaA2AAgent
from src.agents.a2a.normgraph_agent import NormGraphA2AAgent
from src.agents.a2a.provision_agent import ProvisionA2AAgent
from src.agents.a2a.scenario_data_agent import ScenarioDataA2AAgent


@pytest.mark.parametrize(
    "agent_type",
    [
        RestrictionA2AAgent,
        ProvisionA2AAgent,
        DocumentQaA2AAgent,
        NormGraphA2AAgent,
        ScenarioDataA2AAgent,
    ],
)
def test_card_capabilities_match_synapse_strict_schema(agent_type) -> None:
    card = agent_type().get_agent_card("https://gmart.example")

    assert set(card["capabilities"]) <= {
        "streaming",
        "pushNotifications",
        "stateTransitionHistory",
        "extensions",
    }
    assert card["protocolVersion"] == "0.3.0"
    assert card["url"].startswith("https://gmart.example/")


@pytest.mark.parametrize("agent_type", [RestrictionA2AAgent, ProvisionA2AAgent])
def test_prepopulated_params_card_relaxes_only_extension_activation(agent_type) -> None:
    canonical = agent_type().get_agent_card("https://gmart.example")
    compatible = synapse_prepopulated_params_agent_card(canonical)

    canonical_extension = canonical["capabilities"]["extensions"][0]
    compatible_extension = compatible["capabilities"]["extensions"][0]

    assert canonical_extension["required"] is True
    assert compatible_extension["required"] is False
    assert compatible_extension["params"]["required"] == ["scenario_id"]
    assert compatible["url"] == canonical["url"]
