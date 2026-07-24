from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION


class UrbanDataA2AAgent(A2AServer):
    """A2A agent card for the urban-data QA (external, grouped Urban MCP) agent."""

    def __init__(self) -> None:
        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        return self._build_agent_card(base_url).to_dict()

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        url = (
            f"{base_url.rstrip('/')}/urban-data/a2a" if base_url else "/urban-data/a2a"
        )
        return AgentCard(
            name="urban-data-qa-agent",
            description=(
                "Answers questions about urban data via an external, grouped Urban MCP "
                "server: territories, projects/scenarios, physical objects, services, "
                "indicators, social groups/values, and reference dictionaries. Picks and "
                "calls the relevant read-only tools itself and returns GeoJSON layers "
                "when the data is spatial."
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
                "application/json",
            ],
            skills=[
                AgentSkill(
                    id="answer-urban-data-questions",
                    name="Answer questions on urban data",
                    description=(
                        "Discovers the available Urban MCP tools at runtime and calls "
                        "them to answer questions about territories, projects/scenarios, "
                        "physical objects, services, indicators, social groups/values, "
                        "and dictionaries; strictly read-only (never creates or modifies "
                        "data). Returns a grounded text answer and, when the tool results "
                        "carry spatial data, GeoJSON layer artifacts."
                    ),
                    tags=[
                        "urban-data",
                        "territories",
                        "projects",
                        "indicators",
                        "geospatial",
                        "geojson",
                    ],
                    examples=[
                        "Какие территории входят в проект?",
                        "Покажи физические объекты на этой территории",
                        "Какие индикаторы есть для этого сценария?",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=[
                        "text/plain",
                        "application/vnd.geo+json",
                        "application/geo+json",
                        "application/json",
                    ],
                )
            ],
        )
