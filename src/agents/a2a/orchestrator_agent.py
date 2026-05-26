from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION


class OrchestratorA2AAgent(A2AServer):
    """
    A2A entity for the orchestrator agent.
    Routes user queries to restriction-creation-agent and/or provision-effects-agent
    based on LLM intent classification.
    Attributes:
        agent_card (AgentCard): python-a2a agent card used for discovery.
    """

    def __init__(self) -> None:
        """OrchestratorA2AAgent initialization function."""
        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        """
        Return A2A agent card for the given server base URL.
        Args:
            base_url (str): Public server base URL.
        Returns:
            dict[str, Any]: Serialized A2A agent card.
        """
        return self._build_agent_card(base_url).to_dict()

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        """
        Build a python-a2a AgentCard instance.
        Args:
            base_url (str): Public server base URL.
        Returns:
            AgentCard: A2A agent card instance.
        """
        url = (
            f"{base_url.rstrip('/')}/orchestrator/a2a"
            if base_url
            else "/orchestrator/a2a"
        )
        return AgentCard(
            name="orchestrator-agent",
            description=(
                "Routes geospatial queries to restriction-creation-agent and/or "
                "provision-effects-agent based on LLM intent classification. "
                "Requires scenario_id in message metadata. "
                "Returns merged status updates and GeoJSON artifacts from all invoked sub-agents."
            ),
            url=url,
            version=APP_VERSION,
            protocol_version="0.3.0",
            preferred_transport="JSONRPC",
            capabilities={
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
                "google_a2a_compatible": True,
                "parts_array_format": True,
            },
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=[
                "text/plain",
                "application/vnd.geo+json",
                "application/geo+json",
            ],
            skills=[
                AgentSkill(
                    id="orchestrate-geospatial-query",
                    name="Orchestrate geospatial analysis",
                    description=(
                        "Classifies user intent via LLM and delegates to "
                        "restriction-creation-agent and/or provision-effects-agent. "
                        "Streams status updates and merged GeoJSON artifacts."
                    ),
                    tags=[
                        "orchestrator",
                        "restrictions",
                        "provision",
                        "geospatial",
                        "routing",
                    ],
                    examples=[
                        "Построй зону ограничения вокруг школ 200 метров",
                        "Рассчитай эффекты обеспеченности для детских садов",
                        "Ограничения вокруг больниц и их влияние на обеспеченность",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=[
                        "text/plain",
                        "application/vnd.geo+json",
                        "application/geo+json",
                    ],
                )
            ],
        )
