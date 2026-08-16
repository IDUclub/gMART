from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION
from src.agents.a2a.a2a_format import scenario_context_extension


class ScenarioDataA2AAgent(A2AServer):
    """A2A agent card for read-only Urban API data questions."""

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
            f"{base_url.rstrip('/')}/scenario-data/a2a"
            if base_url
            else "/scenario-data/a2a"
        )
        output_modes = [
            "text/plain",
            "application/vnd.geo+json",
            "application/geo+json",
            "application/json",
        ]
        return AgentCard(
            name="scenario-data-agent",
            description=(
                "Answers read-only questions over the grouped Urban MCP. It can query "
                "shared dictionaries and urban data without scenario context, and uses "
                "an optional scenario_id for scenario-scoped tools. Returns text, "
                "GeoJSON layers, and strict data tables."
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
                "extensions": [scenario_context_extension(required=False)],
            },
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=output_modes,
            skills=[
                AgentSkill(
                    id="query-scenario-and-urban-data",
                    name="Query scenario and urban data",
                    description=(
                        "Discovers and calls read-only Urban MCP tools for projects, "
                        "territories, physical objects, services, dictionaries, "
                        "indicators, and social groups. scenario_id is optional; when "
                        "it is absent, scenario-scoped tools are not exposed."
                    ),
                    tags=[
                        "urban-api",
                        "scenario-data",
                        "geojson",
                        "tables",
                        "read-only",
                    ],
                    examples=[
                        "Какие типы городских сервисов доступны?",
                        "Какие объекты есть в сценарии и сколько их по типам?",
                        "Покажи физические объекты сценария на карте",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=output_modes,
                )
            ],
        )
