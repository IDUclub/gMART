from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_DESCRIPTION, APP_VERSION
from src.agents.a2a.a2a_format import scenario_context_extension


class RestrictionA2AAgent(A2AServer):
    """
    A2A entity for restriction creation agent.
    Attributes:
        agent_card (AgentCard): python-a2a agent card used for discovery.
    """

    def __init__(self) -> None:
        """
        RestrictionA2AAgent initialization function.
        """

        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        """
        Function returns A2A agent card for current server base url.
        Args:
            base_url (str): Public server base url.
        Returns:
            dict[str, Any]: A2A agent card representation.
        """

        return self._build_agent_card(base_url).to_dict()

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        """
        Function creates python-a2a AgentCard instance.
        Args:
            base_url (str): Public server base url.
        Returns:
            AgentCard: A2A agent card instance.
        """

        url = f"{base_url.rstrip('/')}/a2a" if base_url else "/a2a"
        return AgentCard(
            name="restriction-creation-agent",
            description=APP_DESCRIPTION.strip(),
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
                "extensions": [scenario_context_extension()],
            },
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=[
                "text/plain",
                "application/vnd.geo+json",
                "application/geo+json",
            ],
            skills=[
                AgentSkill(
                    id="create-geospatial-restrictions",
                    name="Create geospatial construction restrictions",
                    description=(
                        "Runs the restriction creation pipeline and returns status updates, "
                        "text and GeoJSON layers as A2A artifacts."
                    ),
                    tags=["restrictions", "geospatial", "geojson"],
                    examples=[
                        "Построй зону ограничения вокруг школ 200 метров",
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
