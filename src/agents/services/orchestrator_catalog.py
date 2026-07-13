"""
Static catalogue of the gMART agents the orchestrator can route a request to.

Descriptions are written in Russian (the domain language) — they are embedded
verbatim into the orchestrator planner's system prompt, so the LLM routes user
requests based on these texts. Keep them accurate and example-rich.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.agents.services.service_entities.orchestrator_plan import OrchestratorAgent

if TYPE_CHECKING:
    from src.agents.common.config.app_config import AgentsAppConfig


@dataclass(frozen=True)
class AgentCatalogEntry:
    """
    A single routable agent as the planner sees it.
    Attributes:
        key (OrchestratorAgent): Stable agent key used in plan steps.
        title (str): Human-readable Russian title (used in SSE events and digests).
        description (str): What the agent does and which requests it fits (Russian).
        examples (tuple[str, ...]): Example user requests the agent handles.
        requires_scenario_id (bool): True when the agent cannot run without a
            scenario selected in Urban API.
    """

    key: OrchestratorAgent
    title: str
    description: str
    examples: tuple[str, ...]
    requires_scenario_id: bool


AGENT_CATALOG: dict[OrchestratorAgent, AgentCatalogEntry] = {
    OrchestratorAgent.RESTRICTION: AgentCatalogEntry(
        key=OrchestratorAgent.RESTRICTION,
        title="Агент градостроительных ограничений",
        description=(
            "Извлекает из запроса градостроительные ограничения для выбранного "
            "сценария, строит буферные зоны вокруг объектов и формирует слои "
            "ограничений застройки (GeoJSON). Подходит для запросов про зоны "
            "ограничений, санитарные/охранные буферы и территории, где нельзя строить."
        ),
        examples=(
            "Построй ограничения застройки от рек и дорог",
            "Сформируй буферные зоны 100 метров вокруг промышленных объектов",
        ),
        requires_scenario_id=True,
    ),
    OrchestratorAgent.PROVISION: AgentCatalogEntry(
        key=OrchestratorAgent.PROVISION,
        title="Агент обеспеченности сервисами",
        description=(
            "Анализирует обеспеченность территории городскими сервисами для "
            "выбранного сценария: список доступных сервисов, сводка "
            "дефицита/профицита по каталогу, расчёт текущей обеспеченности или "
            "эффектов проекта по конкретному сервису. Возвращает слои, таблицы "
            "и аналитический разбор."
        ),
        examples=(
            "Какая обеспеченность школами?",
            "Как проект повлияет на обеспеченность детскими садами?",
            "Дай сводку по обеспеченности сервисами",
        ),
        requires_scenario_id=True,
    ),
    OrchestratorAgent.DOCUMENTS: AgentCatalogEntry(
        key=OrchestratorAgent.DOCUMENTS,
        title="Агент вопросов по нормативной документации",
        description=(
            "Отвечает на вопросы по текстам нормативных документов "
            "градостроительной сферы (RAG-поиск по базе IDU_DVD): требования, "
            "нормы, определения, формулировки из СП, СанПиН, региональных "
            "нормативов и т.п."
        ),
        examples=(
            "Какие требования к инсоляции жилых помещений?",
            "Что говорит СП о ширине пешеходных дорожек?",
        ),
        requires_scenario_id=False,
    ),
    OrchestratorAgent.NORMS: AgentCatalogEntry(
        key=OrchestratorAgent.NORMS,
        title="Агент графа нормативных ограничений",
        description=(
            "Отвечает на вопросы о нормативных ограничениях как о связанных "
            "правилах (граф NormGraph): какие ограничения действуют на объект "
            "или вид деятельности, их субъекты, значения и конфликты между "
            "нормами."
        ),
        examples=(
            "Какие нормативные ограничения действуют на строительство школ?",
            "Есть ли противоречия в нормах о санитарных зонах?",
        ),
        requires_scenario_id=False,
    ),
}


def available_agents(
    app_config: "AgentsAppConfig",
    scenario_id: int | None,
) -> list[AgentCatalogEntry]:
    """
    Function filters the catalogue down to the agents that can actually run.
    Args:
        app_config (AgentsAppConfig): Application config (optional MCP URLs gate
            the documents/norms agents).
        scenario_id (int | None): Scenario ID from the request; agents requiring
            a scenario are excluded when it is absent.
    Returns:
        list[AgentCatalogEntry]: Agents available for the current request.
    """

    agents: list[AgentCatalogEntry] = []
    for entry in AGENT_CATALOG.values():
        if entry.requires_scenario_id and scenario_id is None:
            continue
        if entry.key == OrchestratorAgent.DOCUMENTS and not app_config.DVD_MCP_URL:
            continue
        if entry.key == OrchestratorAgent.NORMS and not app_config.NORM_GRAPH_MCP_URL:
            continue
        agents.append(entry)
    return agents
