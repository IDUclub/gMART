from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION


class NormGraphA2AAgent(A2AServer):
    """A2A agent card for the normative-restrictions QA (NormGraph graph-RAG) agent."""

    def __init__(self) -> None:
        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        return self._build_agent_card(base_url).to_dict()

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        url = f"{base_url.rstrip('/')}/norms/a2a" if base_url else "/norms/a2a"
        return AgentCard(
            name="norms-qa-agent",
            description=(
                "Answers questions about normative construction restrictions (СП/СНиП/ГОСТ/"
                "СанПиН) via a graph-RAG over NormGraph. Plans which NormGraph tool to call "
                "(free-text search vs. applicable-to-object lookup), optionally checks for "
                "contradicting restrictions, iteratively drafts an answer and self-reviews it "
                "for grounding and citations before returning the final response."
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
            default_output_modes=["text/plain"],
            skills=[
                AgentSkill(
                    id="answer-normative-restriction-questions",
                    name="Answer questions on normative restrictions",
                    description=(
                        "Queries the NormGraph restriction graph (search or applicable-to-object "
                        "lookup, optional graph-neighbourhood expansion and conflict check), "
                        "drafts an answer grounded in and citing the retrieved restrictions, "
                        "critically reviews it, and returns a source-grounded response as A2A "
                        "artifacts."
                    ),
                    tags=["rag", "graph", "restrictions", "compliance", "search"],
                    examples=[
                        "Какие ограничения действуют на объекты пищевой промышленности в "
                        "санитарно-защитной зоне?",
                        "Есть ли противоречия между нормами по минимальной ширине проезда для "
                        "пожарной техники?",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=["text/plain"],
                )
            ],
        )
