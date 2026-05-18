from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION


class ProvisionA2AAgent(A2AServer):
    """A2A agent card for the provision effects pipeline."""

    def __init__(self) -> None:
        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        return self._build_agent_card(base_url).to_dict()

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        url = f"{base_url.rstrip('/')}/provision/a2a" if base_url else "/provision/a2a"
        return AgentCard(
            name="provision-effects-agent",
            description=(
                "Calculates service provision effects for a project scenario and returns "
                "status updates, GeoJSON layers, and an LLM-generated analysis."
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
                    id="calculate-provision-effects",
                    name="Calculate service provision effects",
                    description=(
                        "Runs the provision effects pipeline: resolves service by name, "
                        "calls CalculateObjectEffects, and returns layers with an analysis."
                    ),
                    tags=["provision", "effects", "geospatial", "geojson"],
                    examples=[
                        "Рассчитай эффекты обеспеченности для школ",
                        "Покажи влияние проекта на обеспеченность детскими садами",
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
