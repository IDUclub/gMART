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
            "Строит геометрические зоны ограничений и буферы вокруг объектов "
            "выбранного сценария по расстояниям из запроса. Возвращает GeoJSON зон. "
            "Подходит для «построй буфер», «создай слой отступов». Для проверки "
            "нарушений у конкретных объектов и для вопроса, какие нормативные "
            "ограничения действуют на территории сценария, используй compliance."
        ),
        examples=(
            "Построй ограничения застройки от рек и дорог",
            "Сформируй буферные зоны 100 метров вокруг промышленных объектов",
            "Создай слой отступов 50 метров от дорог",
        ),
        requires_scenario_id=True,
    ),
    OrchestratorAgent.COMPLIANCE: AgentCatalogEntry(
        key=OrchestratorAgent.COMPLIANCE,
        title="Агент проверки нормативного соответствия",
        description=(
            "Проверяет объекты выбранного сценария на нарушения: применяет "
            "канонические правила NormGraph и временные условия пользователя. "
            "Возвращает затронутые объекты, причины, доказательства и покрытие "
            "проверки. Отличает нарушение от невозможности проверки из-за "
            "недостатка данных. Подходит для «проверь соответствие», «какие дома "
            "нарушают отступ», «найди объекты, пересекающие зону». Проверку можно "
            "сузить темой (школы, жилая застройка) и нормативным документом; если "
            "документ не определён однозначно, агент предложит выбрать его. На "
            "вопрос «какие ограничения есть на территории проекта» вместо проверки "
            "показывает на карте зоны действия норм: буферы вокруг школ и других "
            "источников, функциональные зоны и территорию проекта с порогами. "
            "Применяет только нормы документов, действующих на территории "
            "сценария. В task сохраняй дословно названные пользователем темы и "
            "документы и сам вопрос."
        ),
        examples=(
            "Проверь дома по правилу: отступ от дороги не менее 50 м",
            "Какие школы нарушают нормативные ограничения в сценарии?",
            "Проверь объекты по исполнимым правилам NormGraph",
            "Проверь нормы по школам из СП 42.13330",
            "Какие ограничения есть на территории проекта?",
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
    OrchestratorAgent.SCENARIO_DATA: AgentCatalogEntry(
        key=OrchestratorAgent.SCENARIO_DATA,
        title="Агент данных сценария",
        description=(
            "Отвечает на произвольные фактические вопросы по Urban API: проекты, "
            "территории, физические объекты, сервисы, справочники, показатели и "
            "социальные группы. Может работать без выбранного сценария с общими "
            "данными; для сценарных данных попросит выбрать сценарий. При "
            "необходимости возвращает GeoJSON-слои и таблицы. Читает данные, "
            "не создаёт, не изменяет и не удаляет проекты или объекты. "
            "Считает количество объектов («сколько школ»), читает сохранённые "
            "показатели и сравнивает сценарии, в том числе выбранный сценарий с "
            "базовым сценарием проекта — базовый сценарий агент находит сам. "
            "Расчёт дефицита обеспеченности выполняет provision."
        ),
        examples=(
            "Какие объекты есть в сценарии и сколько их по типам?",
            "Покажи на карте физические объекты сценария",
            "Какие значения показателей рассчитаны для сценария?",
            "Сравни показатели сценария с базовым сценарием",
        ),
        requires_scenario_id=False,
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
    OrchestratorAgent.PZZ: AgentCatalogEntry(
        key=OrchestratorAgent.PZZ,
        title="Агент проверки ПЗЗ",
        description=(
            "Проверяет виды разрешённого использования участков и размещение зданий "
            "по правилам землепользования и застройки (ПЗЗ), соответствие функциональным "
            "зонам. Поддерживает кадастровые слои, здания и сценарии; определяет колонки, "
            "запускает PZZ-классификацию, получает отчёт и объясняет результат. "
            "Для сценария нужны год и источник зон; если они отсутствуют, запросит уточнение."
        ),
        examples=(
            "Проверь участки на соответствие ПЗЗ",
            "Проверь здания сценария по зонам PZZ за 2026 год",
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
            "нормами. Не показывает зоны на карте сценария: вопрос об "
            "ограничениях на территории проекта — для compliance."
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
        if (
            entry.key == OrchestratorAgent.SCENARIO_DATA
            and not app_config.URBAN_MCP_URL
        ):
            continue
        if entry.key == OrchestratorAgent.PZZ and not getattr(
            app_config, "PZZ_MCP_URL", None
        ):
            continue
        agents.append(entry)
    return agents
