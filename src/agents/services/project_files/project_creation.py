"""Create the project, its scenario and the files' objects in Urban API.

The steps follow the frontend (urban-assistant-client): a project with the boundary
(Urban API adds the base scenario), a scenario copied from the base one with the
chosen profile, then the functional zones, the physical objects with their
buildings and services. Every created id is written to the run state as soon as it
is known, so a retry creates only what has not been saved — never a second project
or a duplicate object.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from shapely.geometry import mapping, shape
from shapely.geometry.base import BaseGeometry

from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsBaseException,
    AgentsNotFound,
)
from src.agents.services.compilance.compliance_entity_mapping import (
    EntityTarget,
    _number,
    feature_tags,
    normalized_properties,
    signature_of,
)
from src.agents.services.compilance.compliance_uploads import UserFiles
from src.agents.services.project_files.project_params import (
    ExistingPolicy,
    boundary_centre,
)
from src.agents.services.project_files.project_types import CreationSpec

_POLYGONS = {"Polygon", "MultiPolygon"}
_HOUSE = EntityTarget("Жилой дом", "physical_object")
_BATCH = 8
# Urban API may answer 404 for a service on a just-created object for a moment.
SERVICE_RETRY_DELAYS = (0.25, 0.5, 1.0, 2.0, 4.0)
READINESS_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


class _NotReady(Exception):
    """A successful read whose data is not complete yet."""


def _retryable_read(exc: Exception) -> bool:
    if isinstance(exc, DownstreamServiceError):
        return exc.downstream_status in (None, 408, 429) or (
            exc.downstream_status is not None and exc.downstream_status >= 500
        )
    return isinstance(exc, (AgentsNotFound, _NotReady))


@dataclass
class CreationItem:
    """One object of the files as it is created in the scenario."""

    key: str
    layer: str
    source_id: str
    label: str
    spec: CreationSpec
    geometry: dict[str, Any]
    name: str | None
    properties: dict[str, Any]
    building: dict[str, Any] = field(default_factory=dict)
    capacity: int | None = None


def _valid(geometry: dict[str, Any]) -> BaseGeometry | None:
    try:
        value = shape(geometry)
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    if value.is_empty:
        return None
    return value if value.is_valid else value.buffer(0)


def build_items(
    files: UserFiles, specs: dict[str, CreationSpec]
) -> tuple[list[CreationItem], list[CreationItem], dict[str, int]]:
    """Zones and objects of the project layers; counts of what cannot be created."""
    zones: list[CreationItem] = []
    objects: list[CreationItem] = []
    rejected: dict[str, int] = {}
    for layer in files.by_role("project"):
        for index, feature in enumerate(layer.features):
            properties = feature.get("properties") or {}
            tags = feature_tags(properties)
            spec = specs.get(signature_of(tags))
            if spec is None:
                continue
            geometry = _valid(feature.get("geometry") or {})
            if geometry is None or (
                spec.kind == "zone" and geometry.geom_type not in _POLYGONS
            ):
                reason = "некорректная геометрия"
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            source_id = feature.get("id", properties.get("id"))
            source_id = str(index if source_id in (None, "") else source_id)
            normal = normalized_properties(properties, _HOUSE)
            item = CreationItem(
                key=f"{layer.name}/{index}",
                layer=layer.name,
                source_id=source_id,
                label=signature_of(tags).replace("|", ", ") or "без тегов",
                spec=spec,
                geometry=mapping(geometry),
                name=str(properties["name"]) if properties.get("name") else None,
                properties={
                    **properties,
                    "source": "gMART",
                    "source_layer": layer.name,
                    "source_id": source_id,
                },
                building=_building(normal),
                capacity=_capacity(properties),
            )
            (zones if spec.kind == "zone" else objects).append(item)
    return zones, objects, rejected


def _building(normal: dict[str, Any]) -> dict[str, Any]:
    building = normal.get("building") or {}
    payload: dict[str, Any] = {}
    if building.get("floors"):
        payload["floors"] = int(building["floors"])
    if building.get("total_area"):
        payload["building_area_modeled"] = round(float(building["total_area"]), 2)
    extra = {
        key: building[key]
        for key in ("height", "living_area", "footprint_area")
        if building.get(key) is not None
    }
    if normal.get("population") is not None:
        extra["population"] = normal["population"]
    if extra:
        payload["properties"] = extra
    return payload


def _capacity(properties: dict[str, Any]) -> int | None:
    value = _number(properties.get("capacity"))
    return int(value) if value is not None and value >= 0 else None


def describe_error(exc: BaseException) -> str:
    """A short reason for the report: the status and Urban API's own message."""
    if isinstance(exc, AgentsBaseException):
        detail = exc.error_input
        if isinstance(detail, (dict, list)):
            detail = json.dumps(detail, ensure_ascii=False)
        status = (
            exc.downstream_status
            if isinstance(exc, DownstreamServiceError)
            else getattr(exc, "status_code", None)
        )
        text = " ".join(str(detail or exc.message).split())[:300]
        return f"{status}: {text}" if status else text
    return f"{type(exc).__name__}: {exc}"[:300]


def _created_ids(response: Any) -> tuple[int | None, int | None, int | None]:
    """Physical object, its geometry and its building ids from the response."""
    response = response if isinstance(response, dict) else {}
    physical = response.get("physical_object") or {}
    geometry = response.get("object_geometry") or {}
    building = physical.get("building") or {}

    def as_id(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return (
        as_id(physical.get("physical_object_id") or response.get("physical_object_id")),
        as_id(geometry.get("object_geometry_id") or response.get("object_geometry_id")),
        as_id(building.get("id")),
    )


class ProjectCreator:
    """Runs the Urban API calls of a creation; resumable from its run state."""

    def __init__(self, urban_api, sleep=asyncio.sleep) -> None:
        self.urban_api = urban_api
        self._sleep = sleep

    async def create(
        self,
        token: str,
        run: dict[str, Any],
        zones: list[CreationItem],
        objects: list[CreationItem],
        save: Callable[[], Awaitable[None]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield status events; ``run`` holds the plan and every created id."""
        if not run.get("project_id"):
            yield _status("project", "Создаю проект в Urban API…")
            project = await self.urban_api.create_project(
                token,
                {
                    "name": run["project_name"],
                    "territory_id": run["region_id"],
                    "public": False,
                    "is_city": False,
                    "description": "Создан gMART из файлов: "
                    + ", ".join(run.get("files") or []),
                    "properties": {"source": "gMART"},
                    "territory": {
                        "geometry": run["boundary"],
                        "centre_point": boundary_centre(run["boundary"]),
                        "properties": {},
                    },
                },
            )
            run["project_id"] = int(project["project_id"])
            base = project.get("base_scenario") or {}
            run["base_scenario_id"] = base.get("id") or base.get("scenario_id")
            await save()
            yield {
                "type": "project_created",
                "content": {
                    "project_id": run["project_id"],
                    "name": run["project_name"],
                    "region": run.get("region_name"),
                    "base_scenario_id": run["base_scenario_id"],
                },
            }
        if not run.get("scenario_id"):

            async def project_readable():
                await self.urban_api.get_project(token, run["project_id"])
                if not run.get("base_scenario_id"):
                    run["base_scenario_id"] = await self._base_scenario(token, run)
                    await save()

            async for event in self._wait(project_readable, "доступность проекта"):
                yield event
            yield _status("scenario", "Создаю сценарий проекта…")
            scenario = await self.urban_api.copy_scenario(
                token,
                int(run["base_scenario_id"]),
                {
                    "project_id": run["project_id"],
                    "functional_zone_type_id": run["profile_id"],
                    "name": run["scenario_name"],
                    "properties": {"source": "gMART"},
                },
            )
            run["scenario_id"] = int(scenario["scenario_id"])
            await save()
            yield {
                "type": "scenario_created",
                "content": {
                    "project_id": run["project_id"],
                    "scenario_id": run["scenario_id"],
                    "name": run["scenario_name"],
                    "profile": run.get("profile_name"),
                },
            }
        if not run.get("scenario_visible"):

            async def scenario_visible():
                await self._scenario_visible(token, run)

            async for event in self._wait(scenario_visible, "доступность сценария"):
                yield event
            run["scenario_visible"] = True
            await save()
        cleanup = run.setdefault("cleanup", {})
        if not cleanup.get("done"):
            yield _status("cleanup", "Убираю существующие объекты внутри границы…")
            await self._cleanup(token, run, zones, objects)
            await save()
        zone_state = run.setdefault("zones", {})
        if zones and not zone_state.get("done"):
            run.pop("ready", None)
            yield _status("zones", f"Загружаю функциональные зоны: {len(zones)}…")
            try:
                await self.urban_api.add_scenario_functional_zones(
                    token,
                    run["scenario_id"],
                    [
                        {
                            "geometry": item.geometry,
                            "functional_zone_type_id": item.spec.type_id,
                            "name": item.name,
                            "year": run["year"],
                            "source": "User",
                            "properties": item.properties,
                        }
                        for item in zones
                    ],
                )
                zone_state.update(done=True, count=len(zones), error=None)
            except Exception as exc:
                logger.warning(f"Project zones were not saved: {exc!r}")
                zone_state.update(done=False, error=describe_error(exc))
            await save()
        states = run.setdefault("objects", {})
        todo = [item for item in objects if not _object_done(item, states)]
        if todo:
            run.pop("ready", None)
        for start in range(0, len(todo), _BATCH):
            batch = todo[start : start + _BATCH]
            await asyncio.gather(
                *(self._create_object(token, run, item, states) for item in batch)
            )
            await save()
            saved = sum(_object_done(item, states) for item in objects)
            yield _status(
                "objects", f"Сохранено объектов: {saved} из {len(objects)}", saved=saved
            )

        if not (run.get("ready") or {}).get("done"):
            ready = run["ready"] = {"done": False, "attempts": 0, "error": None}

            async def confirm():
                ready["attempts"] += 1
                await self.urban_api.get_project(token, run["project_id"])
                territory = await self.urban_api.get_project_territory(
                    token, run["project_id"]
                )
                if not territory or not territory.get("geometry"):
                    raise _NotReady("Геометрия территории проекта ещё недоступна")
                await self._scenario_visible(token, run)
                collection = await self.urban_api.get_scenario_physical_objects(
                    token, run["scenario_id"]
                )
                expected = {
                    state["physical_object_id"]
                    for state in states.values()
                    if state.get("physical_object_id") is not None
                }
                visible = {
                    (feature.get("properties") or {}).get("physical_object_id")
                    for feature in (collection or {}).get("features") or []
                }
                missing = expected - visible
                if missing:
                    raise _NotReady(f"Объекты сценария ещё недоступны: {len(missing)}")
                ready["objects_confirmed"] = len(expected)

            try:
                async for event in self._wait(confirm, "готовность проекта"):
                    yield event
                ready["done"] = True
            except Exception as exc:
                ready["error"] = describe_error(exc)
            ready["checked_at"] = datetime.now(timezone.utc).isoformat()
            await save()
            if ready["done"]:
                yield {
                    "type": "project_ready",
                    "content": {
                        "project_id": run["project_id"],
                        "scenario_id": run["scenario_id"],
                        "base_scenario_id": run["base_scenario_id"],
                        "objects_confirmed": ready["objects_confirmed"],
                    },
                }

    async def _wait(self, check, label: str) -> AsyncGenerator[dict[str, Any], None]:
        """Retry reads only; mutations must never be repeated here."""
        for attempt in range(len(READINESS_RETRY_DELAYS) + 1):
            yield _status("waiting", f"Проверяю {label}…", attempt=attempt + 1)
            try:
                await check()
                return
            except Exception as exc:
                if not _retryable_read(exc) or attempt == len(READINESS_RETRY_DELAYS):
                    raise
                await self._sleep(READINESS_RETRY_DELAYS[attempt])

    async def _scenario_visible(self, token: str, run: dict[str, Any]) -> None:
        scenarios = await self.urban_api.get_project_scenarios(token, run["project_id"])
        if not any(row.get("scenario_id") == run["scenario_id"] for row in scenarios):
            raise _NotReady("Созданный сценарий ещё отсутствует в списке проекта")

    async def _base_scenario(self, token: str, run: dict[str, Any]) -> int:
        """The base scenario when the project response did not name it."""
        for scenario in await self.urban_api.get_project_scenarios(
            token, run["project_id"]
        ):
            if scenario.get("is_based"):
                return int(scenario["scenario_id"])
        raise _NotReady("Базовый сценарий проекта ещё недоступен")

    async def _cleanup(
        self,
        token: str,
        run: dict[str, Any],
        zones: list[CreationItem],
        objects: list[CreationItem],
    ) -> None:
        policy: ExistingPolicy = run.get("existing_policy") or "keep"
        cleanup = run["cleanup"]
        cleanup.setdefault("deleted", 0)
        cleanup.setdefault("locked", 0)
        cleanup.setdefault("errors", [])
        if policy == "keep":
            cleanup["done"] = True
            return
        replaced = {item.spec.type_id for item in objects}
        border = shape(run["boundary"])
        collection = await self.urban_api.get_scenario_physical_objects(
            token, run["scenario_id"]
        )
        seen: set[int] = set()
        for feature in (collection or {}).get("features") or []:
            properties = feature.get("properties") or {}
            object_id = properties.get("physical_object_id")
            type_id = (properties.get("physical_object_type") or {}).get(
                "physical_object_type_id"
            )
            if object_id is None or object_id in seen:
                continue
            geometry = _valid(feature.get("geometry") or {})
            if geometry is None or not border.contains(geometry.representative_point()):
                continue
            if policy == "replace_same" and type_id not in replaced:
                continue
            seen.add(object_id)
            if properties.get("is_locked"):
                # Urban API answers 200 to deleting a locked object and keeps it.
                cleanup["locked"] += 1
                continue
            try:
                await self.urban_api.delete_scenario_physical_object(
                    token,
                    run["scenario_id"],
                    int(object_id),
                    bool(properties.get("is_scenario_physical_object")),
                )
                cleanup["deleted"] += 1
            except Exception as exc:
                cleanup["errors"].append(describe_error(exc))
        if zones or policy == "delete_all":
            try:
                await self.urban_api.delete_scenario_functional_zones(
                    token, run["scenario_id"]
                )
                cleanup["zones_deleted"] = True
            except Exception as exc:
                cleanup["errors"].append(describe_error(exc))
        cleanup["done"] = True

    async def _create_object(
        self,
        token: str,
        run: dict[str, Any],
        item: CreationItem,
        states: dict[str, dict[str, Any]],
    ) -> None:
        state = states.setdefault(item.key, {})
        state["error"] = None
        scenario_id = run["scenario_id"]
        try:
            if not state.get("physical_object_id"):
                response = await self.urban_api.add_scenario_physical_object(
                    token,
                    scenario_id,
                    {
                        "geometry": item.geometry,
                        "territory_id": run["region_id"],
                        "physical_object_type_id": item.spec.type_id,
                        "name": item.name,
                        "properties": item.properties,
                    },
                )
                object_id, geometry_id, building_id = _created_ids(response)
                if object_id is None or geometry_id is None:
                    raise ValueError("Urban API не вернул идентификаторы объекта")
                state.update(
                    physical_object_id=object_id,
                    object_geometry_id=geometry_id,
                    building_id=building_id,
                )
            if _needs_building(item) and state.get("building") != "done":
                payload = item.building
                if state.get("building_id"):
                    await self.urban_api.patch_scenario_building(
                        token, scenario_id, state["building_id"], payload
                    )
                else:
                    await self.urban_api.add_scenario_building(
                        token,
                        scenario_id,
                        {
                            **payload,
                            "physical_object_id": state["physical_object_id"],
                            "is_scenario_object": True,
                        },
                    )
                state["building"] = "done"
            if item.spec.service_type_id and state.get("service") != "done":
                await self._add_service(token, scenario_id, item, state)
                state["service"] = "done"
        except Exception as exc:
            logger.warning(f"Project object {item.key} was not saved: {exc!r}")
            state["error"] = describe_error(exc)

    async def _add_service(
        self,
        token: str,
        scenario_id: int,
        item: CreationItem,
        state: dict[str, Any],
    ) -> None:
        payload = {
            "physical_object_id": state["physical_object_id"],
            "object_geometry_id": state["object_geometry_id"],
            "is_scenario_physical_object": True,
            "is_scenario_geometry": True,
            "service_type_id": item.spec.service_type_id,
            "capacity": item.capacity,
            "name": item.name,
        }
        for attempt in range(len(SERVICE_RETRY_DELAYS) + 1):
            try:
                await self.urban_api.add_scenario_service(token, scenario_id, payload)
                return
            except AgentsNotFound:
                if attempt == len(SERVICE_RETRY_DELAYS):
                    raise
                await self._sleep(SERVICE_RETRY_DELAYS[attempt])


def _needs_building(item: CreationItem) -> bool:
    return item.spec.building and bool(item.building)


def _object_done(item: CreationItem, states: dict[str, dict[str, Any]]) -> bool:
    state = states.get(item.key) or {}
    return bool(
        state.get("physical_object_id")
        and (not _needs_building(item) or state.get("building") == "done")
        and (not item.spec.service_type_id or state.get("service") == "done")
    )


def failed_objects(
    objects: list[CreationItem], run: dict[str, Any]
) -> list[tuple[CreationItem, str]]:
    states = run.get("objects") or {}
    return [
        (item, (states.get(item.key) or {}).get("error") or "не сохранён")
        for item in objects
        if not _object_done(item, states)
    ]


def creation_summary(
    run: dict[str, Any],
    zones: list[CreationItem],
    objects: list[CreationItem],
) -> dict[str, Any]:
    states = run.get("objects") or {}
    by_type: dict[str, dict[str, int]] = {}
    for item in objects:
        row = by_type.setdefault(
            item.spec.type_name, {"total": 0, "saved": 0, "services": 0}
        )
        row["total"] += 1
        if _object_done(item, states):
            row["saved"] += 1
            row["services"] += bool(item.spec.service_type_id)
    services: dict[str, int] = {}
    for item in objects:
        if item.spec.service_name and _object_done(item, states):
            services[item.spec.service_name] = (
                services.get(item.spec.service_name, 0) + 1
            )
    zone_state = run.get("zones") or {}
    failures = failed_objects(objects, run)
    return {
        "project_id": run.get("project_id"),
        "ready": bool((run.get("ready") or {}).get("done")),
        "ready_error": (run.get("ready") or {}).get("error"),
        "project_name": run.get("project_name"),
        "base_scenario_id": run.get("base_scenario_id"),
        "scenario_id": run.get("scenario_id"),
        "scenario_name": run.get("scenario_name"),
        "region": {
            "territory_id": run.get("region_id"),
            "name": run.get("region_name"),
        },
        "profile": run.get("profile_name"),
        "existing_policy": run.get("existing_policy"),
        "deleted_existing": (run.get("cleanup") or {}).get("deleted", 0),
        "locked_existing": (run.get("cleanup") or {}).get("locked", 0),
        "cleanup_errors": (run.get("cleanup") or {}).get("errors") or [],
        "zones": {
            "total": len(zones),
            "saved": len(zones) if zone_state.get("done") else 0,
            "error": zone_state.get("error"),
        },
        "objects": {
            "total": len(objects),
            "saved": len(objects) - len(failures),
            "by_type": by_type,
            "services": services,
        },
        "failed": [
            {
                "key": item.key,
                "label": item.label,
                "type": item.spec.type_name,
                "error": error,
            }
            for item, error in failures[:50]
        ],
        "failed_total": len(failures),
    }


def _status(stage: str, text: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "status",
        "content": {"status": f"project_{stage}", "text": text, **extra},
    }
