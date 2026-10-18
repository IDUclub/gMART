from python_a2a.models.agent import AgentCard, AgentSkill
from python_a2a.server.a2a_server import A2AServer

from src.agents.__version__ import APP_VERSION
from src.agents.a2a.a2a_format import synapse_compatible_agent_card


class ProjectA2AAgent(A2AServer):
    """A2A agent card for creating an Urban API project from the user's files."""

    def __init__(self) -> None:
        super().__init__(
            agent_card=self._build_agent_card(""), google_a2a_compatible=True
        )

    def get_agent_card(self, base_url: str):
        return synapse_compatible_agent_card(self._build_agent_card(base_url).to_dict())

    @staticmethod
    def _build_agent_card(base_url: str) -> AgentCard:
        url = f"{base_url.rstrip('/')}/projects/a2a" if base_url else "/projects/a2a"
        return AgentCard(
            name="project-files-agent",
            url=url,
            version=APP_VERSION,
            description=(
                "Создаёт проект и сценарий в Urban API из файлов пользователя "
                "(GeoJSON-слои проекта застройки со смешанными объектами): "
                "сопоставляет объекты с типами Urban API, определяет регион и профиль "
                "сценария, загружает зоны, здания и сервисы. Неясное уточняет "
                "вопросом (input-required); ответ — следующее сообщение того же "
                "контекста."
            ),
            protocol_version="0.3.0",
            preferred_transport="JSONRPC",
            capabilities={
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": True,
            },
            default_input_modes=["text/plain", "application/json"],
            default_output_modes=["text/plain", "application/json"],
            skills=[
                AgentSkill(
                    id="create-project-from-files",
                    name="Создание проекта из файлов",
                    description=(
                        "Передайте upload_ids (POST /compliance/uploads) или files — "
                        "список {name, role, geojson} — в DataPart или metadata; "
                        "необязательные inputs: project_name, scenario_name, "
                        "territory_id, functional_zone_type_id, existing_objects "
                        "(replace_same | delete_all | keep). Результат — "
                        "project_id и scenario_id в артефакте project_summary."
                    ),
                    tags=["urban-api", "project", "scenario", "geojson", "upload"],
                    examples=[
                        "Создай проект из загруженных файлов",
                        "Создай проект «Город-сад» и сценарий многоэтажной застройки",
                    ],
                    input_modes=["text/plain", "application/json"],
                    output_modes=["text/plain", "application/json"],
                )
            ],
        )
