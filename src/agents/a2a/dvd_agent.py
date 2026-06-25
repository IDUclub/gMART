from __future__ import annotations

from typing import Any

from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION


class DocumentQaA2AAgent(A2AServer):
    """A2A agent card for the regulatory-documents QA (RAG) agent."""

    def __init__(self) -> None:
        super().__init__(
            agent_card=self._build_agent_card(""),
            google_a2a_compatible=True,
        )

    def get_agent_card(self, base_url: str) -> dict[str, Any]:
        return self._build_agent_card(base_url).to_dict()

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        url = f"{base_url.rstrip('/')}/documents/a2a" if base_url else "/documents/a2a"
        return AgentCard(
            name="document-qa-agent",
            description=(
                "Answers questions about regulatory urban-planning documents (RAG over the "
                "IDU_DVD vector database). Iteratively drafts an answer and self-reviews it "
                "against retrieved fragments before returning the final response."
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
                    id="answer-normative-questions",
                    name="Answer questions on regulatory documents",
                    description=(
                        "Retrieves fragments from IDU_DVD, drafts an answer, critically "
                        "reviews it, and returns a source-grounded response as A2A artifacts."
                    ),
                    tags=["rag", "documents", "regulatory", "search"],
                    examples=[
                        "Какие требования к озеленению дворовых территорий?",
                        "Какова минимальная ширина проезда для пожарной техники?",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=["text/plain"],
                )
            ],
        )
