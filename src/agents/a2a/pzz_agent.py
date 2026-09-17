from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION
from src.agents.a2a.a2a_format import (
    scenario_context_extension,
    synapse_compatible_agent_card,
)


class PzzA2AAgent(A2AServer):
    def __init__(self):
        super().__init__(
            agent_card=self._build_agent_card(""), google_a2a_compatible=True
        )

    def get_agent_card(self, base_url):
        return synapse_compatible_agent_card(self._build_agent_card(base_url).to_dict())

    @staticmethod
    def _build_agent_card(base_url):
        return AgentCard(
            name="pzz-agent",
            url=f"{base_url.rstrip('/')}/pzz/a2a",
            version=APP_VERSION,
            description="Проверяет ВРИ участков и размещение зданий по ПЗЗ: определяет колонки, запускает проверку, получает отчёт и формирует ответ по его данным.",
            protocol_version="0.3.0",
            preferred_transport="JSONRPC",
            capabilities={
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
                "extensions": [scenario_context_extension(required=False)],
            },
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=["text/plain", "application/json"],
            skills=[
                AgentSkill(
                    id="pzz-auto-check",
                    name="Проверка ПЗЗ",
                    description="Режимы pzz_check, classify_only, building_pzz_check и scenario. Передайте inputs в DataPart или metadata. Для сценария нужны scenario_id, year, source; для файлов — GeoJSON или upload_id. confirmed_zone_map задаётся только после подтверждения пользователя.",
                    tags=["pzz", "vri", "zoning", "compliance"],
                    examples=[
                        "Проверь ВРИ участков по ПЗЗ",
                        "Проверь здания сценария по зонам PZZ за 2026 год",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=["text/plain", "application/json"],
                )
            ],
        )
