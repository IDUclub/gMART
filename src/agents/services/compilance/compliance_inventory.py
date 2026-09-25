"""The areas executable norms govern on a scenario («какие ограничения есть»).

An inventory run reads the same CheckPlans as a compliance check, but instead of
judging the checked objects it draws the area each norm acts on: the buffer
around its sources (schools, enterprises), the scenario's functional zones it
limits, or the whole project territory for a norm on every zone. Only the layers
the area is built from are required; targets and their attributes are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from src.agents.services.compilance.compliance_executor import _has_features
from src.agents.services.compilance.compliance_registry import (
    UnsupportedSchemaError,
    UnsupportedTemplateError,
)
from src.agents.services.compilance.compliance_sources import source_reference
from src.agents.services.service_entities.compliance import (
    CheckPlan,
    DeclaredRequirements,
)

if TYPE_CHECKING:
    from src.agents.services.compilance.compliance_executor import (
        ComplianceTemplateExecutor,
    )

ZoneKind = Literal["restriction", "required"]
ZoneStatus = Literal["shown", "no_objects", "unverifiable", "unsupported"]

PROJECT_TERRITORY_LAYER = "project_territory"
# GetFunctionalZones answers under this key; replay feeds it to the zone builder.
FUNCTIONAL_ZONES_LAYER = "functional_zones"
ALL_ZONES_ENTITY = "functional_zones"
ZONAL_TEMPLATES = frozenset({"zonal_attribute_threshold", "zonal_ratio"})
ZONE_TOOL = "CreateRestrictionZones"
_MAX_DESCRIPTION = 1000

ZONE_KIND_TITLES = {
    "restriction": "Зона ограничения",
    "required": "Зона требуемого размещения",
}


@dataclass
class RestrictionZone:
    """One norm's area, or why it is not on the map."""

    restriction_id: str
    template: str
    template_version: int
    status: ZoneStatus
    zone_kind: ZoneKind = "restriction"
    source: dict[str, Any] = field(default_factory=dict)
    description: dict[str, Any] = field(default_factory=dict)
    zones: dict[str, Any] | None = None
    missing_requirements: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def zone_count(self) -> int:
        return len((self.zones or {}).get("features") or [])

    def payload(self) -> dict[str, Any]:
        """Metadata of the zone; geometry travels only as feature_collection."""
        return {
            "restriction_id": self.restriction_id,
            "template": self.template,
            "template_version": self.template_version,
            "status": self.status,
            "zone_kind": self.zone_kind,
            "zone_count": self.zone_count,
            "skipped_objects": int(
                ((self.zones or {}).get("meta") or {}).get("skipped_objects") or 0
            ),
            "description": self.description,
            "missing_requirements": self.missing_requirements,
            "source": self.source,
        }


def zone_kind(plan: CheckPlan, params) -> ZoneKind:
    """Where targets must not be (``restriction``) or must be (``required``)."""
    if plan.template == "presence_within":
        return "required"
    if plan.template in {"distance_from_source", "distance_table"}:
        return "required" if params.violation_when == "not_matched" else "restriction"
    return "restriction"


def zone_layer_name(zone: RestrictionZone) -> str:
    return f"{ZONE_KIND_TITLES[zone.zone_kind]} — {source_reference(zone.source)}"


class RestrictionZoneBuilder:
    def __init__(self, executor: "ComplianceTemplateExecutor") -> None:
        self.executor = executor

    async def project_territory(
        self, mcp_client, scenario_id: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """The project boundary layer and its retrieval call (``None`` if absent)."""
        arguments = {"scenario_id": scenario_id}
        response = await self.executor.tools.execute_named_tool(
            mcp_client, "GetProjectTerritory", arguments
        )
        layer = (response or {}).get(PROJECT_TERRITORY_LAYER)
        if not _has_features(layer):
            return None, None
        return layer, {
            "function": {"name": "GetProjectTerritory", "arguments": arguments}
        }

    async def build(
        self,
        mcp_client,
        raw_plan: dict[str, Any],
        scenario_id: int,
        project_territory: dict[str, Any] | None = None,
    ) -> RestrictionZone:
        registry = self.executor.registry
        source = dict(raw_plan.get("source") or {})
        restriction_id = str(source.get("restriction_id") or "unknown")
        try:
            plan, params = registry.validate_plan(raw_plan)
        except (
            UnsupportedSchemaError,
            UnsupportedTemplateError,
            ValidationError,
        ) as exc:
            return RestrictionZone(
                restriction_id=restriction_id,
                template=str(raw_plan.get("template") or "unknown"),
                template_version=int(raw_plan.get("template_version") or 1),
                status="unsupported",
                source=source,
                missing_requirements=[str(exc)],
            )
        zone = RestrictionZone(
            restriction_id=plan.source.restriction_id,
            template=plan.template,
            template_version=plan.template_version,
            status="unverifiable",
            zone_kind=zone_kind(plan, params),
            source={**source, **plan.source.model_dump(mode="json")},
        )
        if plan.planner_status == "unsupported":
            zone.status = "unsupported"
            zone.missing_requirements = ["planner_status:unsupported"]
            return zone

        effective = registry.effective_requirements(plan)
        requirements = DeclaredRequirements(
            layers=effective["layers"], attributes=effective["attributes"]
        )
        entities = {item.role: item.entity for item in requirements.layers}
        zone.description = _describe(plan, params, entities)
        roles = _zone_roles(plan, params)
        whole_territory = _covers_every_zone(plan, params, entities)
        if whole_territory:
            roles = set()
        # Targets the area is not drawn from may stay undeclared.
        missing_roles = sorted(
            (roles - set(entities)) | (set(effective["missing_registry_roles"]) & roles)
        )
        if missing_roles:
            zone.missing_requirements = [
                f"declared_requirement:{role}" for role in missing_roles
            ]
            return zone
        requirements = DeclaredRequirements(
            layers=[item for item in requirements.layers if item.role in roles],
            attributes=[item for item in requirements.attributes if item.on in roles],
        )

        if whole_territory:
            if project_territory is None:
                zone.missing_requirements = ["layer:project_territory"]
                return zone
            return await self._draw(
                mcp_client,
                zone,
                {
                    "geometry_mode": "geometry",
                    "source_layer": PROJECT_TERRITORY_LAYER,
                },
                {PROJECT_TERRITORY_LAYER: project_territory},
                calls=[],
            )

        catalog = await self.executor.catalog_resolver.resolve(
            mcp_client, scenario_id, requirements
        )
        if not catalog.executable:
            zone.missing_requirements = catalog.missing
            return zone
        requirements = catalog.requirements
        layers, calls = await self.executor._retrieve_layers(
            mcp_client, requirements, scenario_id
        )
        resolution = self.executor.data_gate.resolve(plan, layers, requirements)
        if not resolution.executable:
            zone.missing_requirements = resolution.missing
            zone.tool_calls = calls
            return zone
        source_role = next(iter(roles))
        profile = resolution.profiles.get(source_role) or {}
        if profile.get("object_count") == 0:
            # The complete scenario layer has nothing the norm draws around.
            zone.status = "no_objects"
            return zone

        source_layer = resolution.role_layers[source_role]
        arguments: dict[str, Any] = {
            "geometry_mode": "geometry",
            "source_layer": source_layer,
        }
        zone_layers = {source_layer: resolution.layers[source_layer]}
        if plan.template in {"distance_from_source", "presence_within"}:
            if getattr(params, "geometry_mode", "buffered") == "buffered":
                arguments.update(geometry_mode="buffer", distance_m=params.distance_m)
        elif plan.template == "distance_table":
            arguments.update(
                geometry_mode="attribute_buffer",
                attribute_field=resolution.selected_fields[params.attribute_role],
                bands=[band.model_dump(mode="json") for band in params.bands],
            )
        elif plan.template in ZONAL_TEMPLATES:
            arguments["source_layer"] = FUNCTIONAL_ZONES_LAYER
            zone_layers = {FUNCTIONAL_ZONES_LAYER: resolution.layers[source_layer]}
            threshold = getattr(params, "threshold_source", None)
            if threshold is not None and threshold.kind == "attribute_role":
                arguments["threshold_field"] = resolution.selected_fields[
                    threshold.role
                ]
            if project_territory is not None:
                arguments["clip_layer"] = PROJECT_TERRITORY_LAYER
                zone_layers[PROJECT_TERRITORY_LAYER] = project_territory
        return await self._draw(mcp_client, zone, arguments, zone_layers, calls)

    async def _draw(
        self,
        mcp_client,
        zone: RestrictionZone,
        arguments: dict[str, Any],
        layers: dict[str, dict[str, Any]],
        calls: list[dict[str, Any]],
    ) -> RestrictionZone:
        name = zone_layer_name(zone)
        arguments = {
            "layer_name": name,
            **arguments,
            "properties": _zone_properties(zone, name),
        }
        response = await self.executor.tools.execute_named_tool(
            mcp_client, ZONE_TOOL, {**arguments, "layers": layers}
        )
        zone.zones = (response or {}).get(name)
        zone.tool_calls = list(calls)
        if zone.zone_count:
            zone.status = "shown"
            # Replay rebuilds the zone from the recorded retrieval calls.
            zone.tool_calls.append(
                {"function": {"name": ZONE_TOOL, "arguments": arguments}}
            )
        else:
            zone.status = "no_objects"
        return zone


def _zone_roles(plan: CheckPlan, params) -> set[str]:
    """The layer roles a norm's area is drawn from."""
    if plan.template in {"distance_from_source", "distance_table"}:
        return {params.source_layer}
    if plan.template == "presence_within":
        return {params.objects_layer}
    return {params.zones_layer}


def _covers_every_zone(plan: CheckPlan, params, entities: dict[str, str]) -> bool:
    """A zonal norm with one threshold for every zone acts on the whole project."""
    if plan.template not in ZONAL_TEMPLATES:
        return False
    if entities.get(params.zones_layer) != ALL_ZONES_ENTITY:
        return False
    threshold = getattr(params, "threshold_source", None)
    return threshold is None or threshold.kind == "constant"


def _describe(plan: CheckPlan, params, entities: dict[str, str]) -> dict[str, Any]:
    """What the area is drawn around and what the norm limits there."""

    def names(roles) -> list[str]:
        return [entities.get(role, role) for role in roles]

    if plan.template == "distance_from_source":
        return {
            "around": entities.get(params.source_layer, params.source_layer),
            "distance_m": params.distance_m,
            "applies_to": names(params.targets),
        }
    if plan.template == "distance_table":
        return {
            "around": entities.get(params.source_layer, params.source_layer),
            "distance_by_attribute": True,
            "applies_to": names(params.targets),
        }
    if plan.template == "presence_within":
        return {
            "around": entities.get(params.objects_layer, params.objects_layer),
            "distance_m": params.distance_m,
            "applies_to": names(params.required_neighbor_layers),
            "minimum": params.minimum_neighbors,
        }
    zones = entities.get(params.zones_layer, params.zones_layer)
    description: dict[str, Any] = {
        "zones": None if zones == ALL_ZONES_ENTITY else zones,
        "operator": params.operator,
    }
    if plan.template == "zonal_ratio":
        description.update(
            applies_to=names([params.numerator.layer]),
            threshold=params.threshold,
            unit="%",
        )
        return description
    description["applies_to"] = names([params.objects_layer])
    threshold = params.threshold_source
    if threshold.kind == "constant":
        description.update(threshold=threshold.value, unit=threshold.unit)
    return description


def _zone_properties(zone: RestrictionZone, title: str) -> dict[str, Any]:
    text = " ".join((zone.source.get("extraction_text") or "").split())
    properties: dict[str, Any] = {
        "restriction_title": title,
        "restriction_description": text[:_MAX_DESCRIPTION] or None,
        "restriction_id": zone.restriction_id,
        "zone_kind": zone.zone_kind,
        "applies_to": zone.description.get("applies_to") or [],
        "provenance": {
            "document_name": zone.source.get("document_name"),
            "clause_number": zone.source.get("clause_number"),
        },
    }
    for key in ("operator", "threshold", "unit"):
        if zone.description.get(key) is not None:
            properties[key] = zone.description[key]
    return properties


_OPERATOR_SIGNS = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "==": "="}


def describe_zone(payload: dict[str, Any]) -> str:
    """«50 м вокруг объектов «Школа»; ограничение для: «Жилой дом»»."""

    description = payload.get("description") or {}
    applies = ", ".join(f"«{name}»" for name in description.get("applies_to") or [])
    if "around" in description:
        around = f"«{description['around']}»"
        if description.get("distance_by_attribute"):
            area = f"буферы индивидуального радиуса вокруг объектов {around}"
        elif description.get("distance_m"):
            area = f"{_number(description['distance_m'])} м вокруг объектов {around}"
        else:
            area = f"территория объектов {around}"
        if payload.get("zone_kind") == "required":
            minimum = int(description.get("minimum") or 1)
            rule = f"здесь должны располагаться {applies}" + (
                f" (не менее {minimum})" if minimum > 1 else ""
            )
        else:
            rule = f"ограничение для {applies}"
        return f"{area}; {rule}" if applies else area
    zones = description.get("zones")
    area = (
        f"функциональные зоны «{zones}» в границах проекта"
        if zones
        else "вся территория проекта"
    )
    sign = _OPERATOR_SIGNS.get(description.get("operator"), description.get("operator"))
    threshold = description.get("threshold")
    limit = (
        f"{sign} {_number(threshold)} {description.get('unit') or ''}".strip()
        if threshold is not None
        else f"{sign} порога, заданного для зоны"
    )
    if payload.get("template") == "zonal_ratio":
        return f"{area}; доля площади {applies} {limit}"
    return f"{area}; для {applies}: {limit}" if applies else f"{area}; {limit}"


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)
