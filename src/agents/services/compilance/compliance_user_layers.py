"""Layers of a CheckPlan built from the user's files instead of the scenario.

The executor resolves a plan's entities against the Urban API catalogue and then
asks a layer source for one FeatureCollection per entity. For a check of user files
that source is ``UserLayerSource``:

- the roles a norm checks (its targets) get only the design's objects;
- the other roles (sources, neighbours, zones) get the design's objects plus the
  context — the files' surroundings and, over a scenario, the scenario's objects;
- over a scenario, scenario objects of an entity the design has are replaced inside
  the design's boundary and stay outside it as context.

``missing`` tells before execution which norms the data cannot check at all, so
they are reported as «не проверено» instead of being run on nothing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import mapping, shape

from src.agents.services.compilance.compliance_entity_mapping import (
    EntityLayers,
    EntityTarget,
)
from src.agents.services.restriction.restriction_catalog import normalize_name
from src.agents.services.service_entities.compliance import (
    CheckPlan,
    DeclaredRequirements,
)

ALL_ZONES = "functional_zones"
_POINT_TYPES = {"Point", "MultiPoint"}
# ObjectEffects computes provision from the scenario in Urban API; the files'
# objects never reach it.
SCENARIO_ONLY_TEMPLATES = {"service_provision"}

ScenarioRetriever = Callable[
    [DeclaredRequirements], Awaitable[tuple[dict[str, dict], list[dict]]]
]


def target_roles(plan: CheckPlan, params: Any) -> list[str]:
    """The roles whose objects a template checks."""
    if plan.template in {"distance_from_source", "distance_table"}:
        return list(params.targets)
    if plan.template in {
        "presence_within",
        "zonal_attribute_threshold",
        "object_attribute_threshold",
        "accessibility_within",
    }:
        return [params.objects_layer]
    if plan.template == "zonal_ratio":
        return [params.zones_layer]
    return []


@dataclass
class UserLayerSource:
    layers: EntityLayers
    # Fetches scenario layers when the files overlay a scenario; ``None`` without one.
    scenario: ScenarioRetriever | None = None
    revision: str = "user"
    _scenario_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _scenario_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @property
    def overlay(self) -> bool:
        return self.scenario is not None

    def missing(
        self, plan: CheckPlan, params: Any, requirements: DeclaredRequirements
    ) -> list[str]:
        """Why the files cannot check this norm; empty when they can."""
        if plan.template in SCENARIO_ONLY_TEMPLATES:
            return ["user_files:scenario_only_template"]
        targets = set(target_roles(plan, params))
        missing: list[str] = []
        has_target = False
        for requirement in requirements.layers:
            present = bool(self._project(requirement))
            if requirement.role in targets:
                has_target = has_target or present
                continue
            if not requirement.required or self.overlay:
                continue  # over a scenario, Urban API supplies the rest
            if not present and not self._context(requirement):
                missing.append(f"user_files:no_objects:{requirement.entity}")
        if targets and not has_target:
            names = sorted(
                {
                    requirement.entity
                    for requirement in requirements.layers
                    if requirement.role in targets
                }
            )
            missing.insert(0, "user_files:no_project_objects:" + ", ".join(names))
        return missing

    async def retrieve(
        self, plan: CheckPlan, params: Any, requirements: DeclaredRequirements
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        """One FeatureCollection per entity of the plan, keyed by the entity name."""
        targets = set(target_roles(plan, params))
        scenario_layers: dict[str, dict[str, Any]] = {}
        calls: list[dict[str, Any]] = []
        if self.overlay:
            scenario_layers, calls = await self._scenario_layers(requirements)
        result: dict[str, dict[str, Any]] = {}
        for requirement in requirements.layers:
            if requirement.entity in result and requirement.role not in targets:
                continue  # a target already took this entity: it is checked
            project = self._project(requirement)
            if requirement.role in targets:
                features = list(project)
            else:
                features = [*project, *self._context(requirement)]
                scenario = scenario_layers.get(normalize_name(requirement.entity))
                if scenario is not None:
                    features += self._scenario_context(
                        scenario.get("features") or [], replaced=bool(project)
                    )
            result[requirement.entity] = {
                "type": "FeatureCollection",
                "features": _fit_geometry(features, requirement.geometry_types),
                "meta": {
                    "complete": True,
                    "truncated": False,
                    "revision": (
                        f"{self.revision}:{requirement.entity}:"
                        f"{'target' if requirement.role in targets else 'context'}"
                    ),
                },
            }
        return result, calls

    def _project(self, requirement) -> list[dict[str, Any]]:
        return _matching(self.layers.project, requirement)

    def _context(self, requirement) -> list[dict[str, Any]]:
        return _matching(self.layers.context, requirement)

    async def _scenario_layers(self, requirements: DeclaredRequirements):
        wanted = [
            requirement
            for requirement in requirements.layers
            if normalize_name(requirement.entity) not in self._scenario_cache
        ]
        if wanted:
            layers, calls = await self.scenario(
                DeclaredRequirements(layers=wanted, attributes=[])
            )
            for name, collection in (layers or {}).items():
                self._scenario_cache[normalize_name(name)] = collection
            for requirement in wanted:
                key = normalize_name(requirement.entity)
                self._scenario_cache.setdefault(key, {"features": []})
                self._scenario_calls[key] = [
                    call for call in calls if _call_mentions(call, requirement.entity)
                ]
        keys = [normalize_name(item.entity) for item in requirements.layers]
        calls = [
            call
            for key in dict.fromkeys(keys)
            for call in self._scenario_calls.get(key, [])
        ]
        unique_calls = list({repr(call): call for call in calls}.values())
        return {key: self._scenario_cache[key] for key in keys}, unique_calls

    def _scenario_context(
        self, features: list[dict[str, Any]], *, replaced: bool
    ) -> list[dict[str, Any]]:
        """Scenario objects outside the design when the design replaces them."""
        if not replaced:
            return list(features)
        border = self.layers.border
        if border is None:
            return []  # without a boundary the design replaces the whole layer
        kept = []
        for feature in features:
            try:
                geometry = shape(feature["geometry"])
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
            if not geometry.intersects(border):
                kept.append(feature)
        return kept


def _matching(
    by_entity: dict[EntityTarget, list[dict[str, Any]]], requirement
) -> list[dict[str, Any]]:
    entity_type = requirement.entity_type.value
    if entity_type == "functional_zone" and requirement.entity == ALL_ZONES:
        return [
            feature
            for target, features in by_entity.items()
            if target.entity_type == "functional_zone"
            for feature in features
        ]
    wanted = normalize_name(requirement.entity)
    return [
        feature
        for target, features in by_entity.items()
        if target.entity_type == entity_type and normalize_name(target.name) == wanted
        for feature in features
    ]


def _fit_geometry(
    features: list[dict[str, Any]], geometry_types: list[str]
) -> list[dict[str, Any]]:
    """A point-only requirement gets the representative point of a contour.

    Urban API stores many services as points; a drawn school is a polygon. A norm
    that measures from points is measured from a point inside the contour.
    """
    allowed = set(geometry_types or [])
    if not allowed or not allowed <= _POINT_TYPES:
        return features
    fitted = []
    for feature in features:
        geometry = feature.get("geometry") or {}
        if geometry.get("type") in allowed:
            fitted.append(feature)
            continue
        try:
            point = shape(geometry).representative_point()
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        fitted.append(
            {
                **feature,
                "geometry": mapping(point),
                "properties": {
                    **(feature.get("properties") or {}),
                    "geometry_fitted": "representative_point",
                },
            }
        )
    return fitted


def _call_mentions(call: dict[str, Any], entity: str) -> bool:
    arguments = (call.get("function") or {}).get("arguments") or {}
    wanted = normalize_name(entity)
    names = [
        *(arguments.get("services_names") or []),
        *(arguments.get("physical_objects_names") or []),
        *(arguments.get("zone_type_names") or []),
    ]
    if (call.get("function") or {}).get("name") == "GetFunctionalZones" and (
        arguments.get("zone_type_names") is None
    ):
        return entity == ALL_ZONES
    return any(normalize_name(name) == wanted for name in names)
