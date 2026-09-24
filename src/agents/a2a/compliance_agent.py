from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION
from src.agents.a2a.a2a_format import (
    scenario_context_extension,
    synapse_compatible_agent_card,
)


class ComplianceA2AAgent(A2AServer):
    """A2A agent card for the normative compliance check of scenario objects."""

    def __init__(self) -> None:
        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        return synapse_compatible_agent_card(self._build_agent_card(base_url).to_dict())

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        url = (
            f"{base_url.rstrip('/')}/compliance/a2a" if base_url else "/compliance/a2a"
        )
        return AgentCard(
            name="compliance-agent",
            description=(
                "Checks the objects of a scenario against executable normative "
                "restrictions from NormGraph: each norm's validated CheckPlan runs as a "
                "deterministic geometry check, with coverage and evidence per norm. The "
                "check can be narrowed to topics (schools, housing) and to normative "
                "documents; when a document cannot be identified unambiguously the task "
                "enters input-required with a numbered choice, answered by the next "
                "message in the same context."
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
                "extensions": [scenario_context_extension()],
            },
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=["text/plain", "application/json"],
            skills=[
                AgentSkill(
                    id="check-normative-compliance",
                    name="Check scenario objects against normative restrictions",
                    description=(
                        "Selects the executable norms matching the request (topic "
                        "entities and documents), checks the scenario's layers against "
                        "each of them and returns the verdict text, violation layers as "
                        "GeoJSON artifacts and a machine-readable summary. Requires "
                        "scenario_id in a DataPart or message/params metadata."
                    ),
                    tags=["compliance", "restrictions", "normgraph", "geojson"],
                    examples=[
                        "Проверь нормы по школам",
                        "Проверь соответствие нормам СП 42.13330 касательно жилой "
                        "застройки",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=["text/plain", "application/json"],
                )
            ],
        )
