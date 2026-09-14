"""REST and MCP views over identical inputs; no planner or calculation doubles."""

import json
import os
from contextlib import AsyncExitStack, asynccontextmanager

from fastmcp import FastMCP

from . import control
from .sources import create_app, lookup

groups = {
    name: FastMCP("Control Urban " + name)
    for name in (
        "projects",
        "territories",
        "physical_objects",
        "dictionaries",
        "indicators",
        "soc_groups",
    )
}


def data(path, params=None):
    return lookup(app, path, params or {})


@groups["projects"].tool(name="GetScenarioById", tags={"read"})
def get_scenario(scenario_id: int) -> dict:
    """Карточка подготовленного сценария: проект, версия и свойства."""
    return data(f"/api/v1/scenarios/{scenario_id}")


@groups["projects"].tool(name="GetProjectById", tags={"read"})
def get_project(project_id: int) -> dict:
    """Проект, базовый сценарий и контекст территории."""
    return data(f"/api/v1/projects/{project_id}")


@groups["projects"].tool(name="GetProjectScenarios", tags={"read"})
def get_scenarios(project_id: int) -> list[dict]:
    """Все подготовленные версии проекта."""
    return data(f"/api/v1/projects/{project_id}/scenarios")


def register_entities(noun, domain):
    def types(scenario_id: int) -> list[dict]:
        return data(f"/api/v1/scenarios/{scenario_id}/{domain}_types")

    def services(
        scenario_id: int, service_type_id: int | None = None, for_context: bool = False
    ) -> list[dict]:
        prefix = "context/" if for_context else ""
        return data(
            f"/api/v1/scenarios/{scenario_id}/{prefix}services",
            {"service_type_id": service_type_id},
        )

    def service_geometry(
        scenario_id: int, service_type_id: int | None = None, for_context: bool = False
    ) -> dict:
        prefix = "context/" if for_context else ""
        return data(
            f"/api/v1/scenarios/{scenario_id}/{prefix}services_with_geometry",
            {"service_type_id": service_type_id},
        )

    def objects(
        scenario_id: int,
        physical_object_type_id: int | None = None,
        for_context: bool = False,
    ) -> list[dict]:
        prefix = "context/" if for_context else ""
        return data(
            f"/api/v1/scenarios/{scenario_id}/{prefix}physical_objects",
            {"physical_object_type_id": physical_object_type_id},
        )

    def object_geometry(
        scenario_id: int,
        physical_object_type_id: int | None = None,
        for_context: bool = False,
    ) -> dict:
        prefix = "context/" if for_context else ""
        return data(
            f"/api/v1/scenarios/{scenario_id}/{prefix}physical_objects_with_geometry",
            {"physical_object_type_id": physical_object_type_id},
        )

    for suffix, fn in [
        ("Types", types),
        ("s", services if domain == "service" else objects),
        ("sWithGeometry", service_geometry if domain == "service" else object_geometry),
    ]:
        groups["projects"].tool(
            name=f"GetScenario{noun}{suffix}",
            description=f"Чтение {domain} выбранного сценария. Полная выборка; фильтр по ID типа. Геометрия в WGS84 для WithGeometry.",
            tags={"read"},
        )(fn)


register_entities("Service", "service")
register_entities("PhysicalObject", "physical_object")


@groups["dictionaries"].tool(name="GetServiceTypes", tags={"read"})
def service_types(name: str | None = None) -> list[dict]:
    """Справочник типов услуг по имени."""
    return data("/api/v1/service_types", {"name": name})


@groups["dictionaries"].tool(name="GetPhysicalObjectTypes", tags={"read"})
def physical_types(name: str | None = None) -> list[dict]:
    """Справочник типов физических объектов по имени."""
    return data("/api/v1/physical_object_types", {"name": name})


@groups["indicators"].tool(name="GetScenarioIndicatorsValues", tags={"read"})
def get_indicators(
    scenario_id: int, indicator_ids: list[int] | None = None
) -> list[dict]:
    """Численность населения и площадь жилых помещений в подготовленном сценарии."""
    rows = data(f"/api/v1/scenarios/{scenario_id}/indicators_values")
    return [
        r
        for r in rows
        if not indicator_ids or r["indicator"]["indicator_id"] in indicator_ids
    ]


@groups["projects"].tool(name="GetScenarioFunctionalZoneSources", tags={"read"})
def zone_sources(scenario_id: int) -> list[dict]:
    """Доступные источник и год функционального зонирования."""
    return data(f"/api/v1/scenarios/{scenario_id}/functional_zone_sources")


@groups["projects"].tool(name="GetScenarioFunctionalZones", tags={"read"})
def zones(scenario_id: int, source: str, year: int) -> dict:
    """Слой функциональных зон подготовленного сценария по источнику и году."""
    return data(
        f"/api/v1/scenarios/{scenario_id}/functional_zones",
        {"source": source, "year": year},
    )


@groups["territories"].tool(name="GetTerritoryNormatives", tags={"read"})
def territory_normatives(territory_id: int) -> list[dict]:
    """Исходные нормативы обеспеченности и доступности; синтетические тестовые данные."""
    return data(f"/api/v1/territory/{territory_id}/normatives")


mcp_apps = {
    name: group.http_app(path="/", stateless_http=True)
    for name, group in groups.items()
}


@asynccontextmanager
async def lifespan(app):
    async with AsyncExitStack() as stack:
        for mcp_app in mcp_apps.values():
            await stack.enter_async_context(mcp_app.lifespan(mcp_app))
        yield


app = create_app(faults=json.loads(os.getenv("INDUSTRIAL_SOURCE_FAULTS", "[]")))
app.router.lifespan_context = lifespan
for name, mcp_app in mcp_apps.items():
    app.mount(f"/mcp/{name}", mcp_app)
