"""Allowlisted executable norm templates and their public manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from src.agents.services.service_entities.compliance import (
    CheckPlan,
    DistanceFromSourceParams,
    DistanceTableParams,
    PresenceWithinParams,
    ZonalAttributeThresholdParams,
    ZonalRatioParams,
)


@dataclass(frozen=True)
class TemplateRegistration:
    template: str
    version: int
    params_model: type[BaseModel]
    required_layer_roles: tuple[str, ...]
    required_attribute_roles: tuple[str, ...]
    executor: str
    tool_names: tuple[str, ...]
    geometry_types: tuple[str, ...]
    max_features: int = 50_000
    max_payload_bytes: int = 64 * 1024 * 1024
    timeout_seconds: int = 120
    evidence_schema_version: str = "1.0"
    enabled: bool = True

    def manifest(self) -> dict[str, Any]:
        return {
            "template": self.template,
            "version": self.version,
            "params_schema": self.params_model.model_json_schema(),
            "required_layer_roles": list(self.required_layer_roles),
            "required_attribute_roles": list(self.required_attribute_roles),
            "tool_names": list(self.tool_names),
            "geometry_types": list(self.geometry_types),
            "limits": {
                "max_features": self.max_features,
                "max_payload_bytes": self.max_payload_bytes,
                "timeout_seconds": self.timeout_seconds,
            },
            "evidence_schema_version": self.evidence_schema_version,
            "enabled": self.enabled,
        }


class UnsupportedSchemaError(ValueError):
    pass


class UnsupportedTemplateError(ValueError):
    pass


class TemplateRegistry:
    def __init__(self, entries: list[TemplateRegistration]) -> None:
        self._entries: dict[tuple[str, int], TemplateRegistration] = {}
        for entry in entries:
            key = (entry.template, entry.version)
            if key in self._entries:
                raise ValueError(f"duplicate template registration: {key}")
            self._entries[key] = entry

    def get(self, template: str, version: int) -> TemplateRegistration:
        entry = self._entries.get((template, version))
        if entry is None or not entry.enabled:
            raise UnsupportedTemplateError(
                f"unsupported or disabled template: {template}@v{version}"
            )
        return entry

    def validate_plan(
        self, raw_plan: dict[str, Any] | CheckPlan
    ) -> tuple[CheckPlan, BaseModel]:
        try:
            plan = (
                raw_plan
                if isinstance(raw_plan, CheckPlan)
                else CheckPlan.model_validate(raw_plan)
            )
        except ValidationError as exc:
            if raw_plan.get("schema_version") != "1.0":
                raise UnsupportedSchemaError("unsupported_schema") from exc
            raise
        entry = self.get(plan.template, plan.template_version)
        return plan, entry.params_model.model_validate(plan.params)

    def public_manifest(self) -> dict[str, Any]:
        templates = [entry.manifest() for _, entry in sorted(self._entries.items())]
        return {"schema_version": "1.0", "templates": templates}

    def effective_requirements(self, plan: CheckPlan) -> dict[str, Any]:
        """Merge registry requirements without allowing a plan to weaken them."""

        entry = self.get(plan.template, plan.template_version)
        params = entry.params_model.model_validate(plan.params).model_dump(mode="json")
        layer_roles = self._resolve_role_selectors(entry.required_layer_roles, params)
        attribute_roles = self._resolve_role_selectors(
            entry.required_attribute_roles, params
        )
        declared = plan.declared_requirements
        layers = {item.role: item for item in (declared.layers if declared else [])}
        attributes = {
            item.role: item for item in (declared.attributes if declared else [])
        }
        for role in layer_roles:
            if role in layers:
                layers[role] = layers[role].model_copy(update={"required": True})
        for role in attribute_roles:
            if role in attributes:
                attributes[role] = attributes[role].model_copy(
                    update={"required": True}
                )
        missing_layers = [role for role in layer_roles if role not in layers]
        missing_attributes = [
            role for role in attribute_roles if role not in attributes
        ]
        return {
            "layers": list(layers.values()),
            "attributes": list(attributes.values()),
            "missing_registry_roles": [*missing_layers, *missing_attributes],
        }

    @staticmethod
    def _resolve_role_selectors(
        selectors: tuple[str, ...], params: dict[str, Any]
    ) -> list[str]:
        roles: list[str] = []
        for selector in selectors:
            current: Any = params
            optional = selector.endswith("?")
            path = selector.removeprefix("$").removesuffix("?")
            for part in path.split("."):
                if not isinstance(current, dict) or part not in current:
                    current = None
                    break
                current = current[part]
            values = current if isinstance(current, list) else [current]
            for value in values:
                if isinstance(value, str) and value not in roles:
                    roles.append(value)
            if current is None and not optional:
                raise ValueError(f"required role selector {selector!r} is unresolved")
        return roles


def build_default_registry(
    disabled_templates: set[str] | None = None,
) -> TemplateRegistry:
    disabled = disabled_templates or set()
    entries = [
        TemplateRegistration(
            "distance_from_source",
            1,
            DistanceFromSourceParams,
            ("$source_layer", "$targets"),
            (),
            "execute_distance_from_source",
            ("CheckDistanceFromSource",),
            (
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            ),
        ),
        TemplateRegistration(
            "distance_table",
            1,
            DistanceTableParams,
            ("$source_layer", "$targets"),
            ("$attribute_role",),
            "execute_distance_table",
            ("CheckDistanceTable",),
            (
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            ),
        ),
        TemplateRegistration(
            "presence_within",
            1,
            PresenceWithinParams,
            ("$objects_layer", "$required_neighbor_layers"),
            (),
            "execute_presence_within",
            ("CheckPresenceWithin",),
            (
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            ),
        ),
        TemplateRegistration(
            "zonal_attribute_threshold",
            1,
            ZonalAttributeThresholdParams,
            ("$objects_layer", "$zones_layer"),
            ("$attribute_role", "$threshold_source.role?"),
            "execute_zonal_attribute_threshold",
            ("CheckZonalAttributeThreshold",),
            (
                "Point",
                "MultiPoint",
                "LineString",
                "MultiLineString",
                "Polygon",
                "MultiPolygon",
            ),
            enabled="zonal_attribute_threshold" not in disabled,
        ),
        TemplateRegistration(
            "zonal_ratio",
            1,
            ZonalRatioParams,
            ("$zones_layer", "$numerator.layer"),
            (),
            "execute_zonal_ratio",
            ("CheckZonalRatio",),
            ("Polygon", "MultiPolygon"),
            max_features=20_000,
            timeout_seconds=180,
            enabled="zonal_ratio" not in disabled,
        ),
    ]
    return TemplateRegistry(entries)


DEFAULT_COMPLIANCE_REGISTRY = build_default_registry()
