"""Scenario-layer profiling and deterministic CheckPlan data gate."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from typing import Any

from src.agents.services.restriction_catalog import normalize_name
from src.agents.services.service_entities.compliance import (
    AttributeCandidate,
    CheckPlan,
    DeclaredRequirements,
    ResolvedRequirement,
)


def _nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for item in path.split("."):
        if not isinstance(current, dict) or item not in current:
            return None
        current = current[item]
    return current


def _normalized_type(values: list[Any]) -> str:
    present = [value for value in values if value is not None]
    if not present:
        return "unknown"
    if all(isinstance(value, bool) for value in present):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in present):
        return "integer"
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in present
    ):
        return "number"
    if all(isinstance(value, str) for value in present):
        return "string"
    return "mixed"


def describe_scenario_layer(
    role: str,
    name: str,
    feature_collection: dict[str, Any],
) -> dict[str, Any]:
    """Return a bounded, non-secret layer profile used by the data gate."""

    features = feature_collection.get("features") or []
    field_names: set[str] = set()

    def collect(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                collect(f"{prefix}.{key}" if prefix else str(key), item)
        elif prefix:
            field_names.add(prefix)

    for feature in features:
        collect("", feature.get("properties") or {})
    fields = []
    warnings: list[str] = []
    for field in sorted(field_names):
        values = [
            _nested_get(feature.get("properties") or {}, field) for feature in features
        ]
        present = [value for value in values if value is not None]
        numeric_values = []
        for value in present:
            if isinstance(value, bool):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                numeric_values.append(numeric)
        kind = _normalized_type(values)
        if kind == "mixed":
            warnings.append(f"mixed_types:{field}")
        examples: list[Any] = []
        for value in present:
            if value not in examples:
                examples.append(value)
            if len(examples) == 3:
                break
        fields.append(
            {
                "name": field,
                "type": kind,
                "null_count": len(values) - len(present),
                "fill_rate": len(present) / len(values) if values else 1.0,
                "numeric_fill_rate": (
                    len(numeric_values) / len(values) if values else 1.0
                ),
                "examples": examples,
            }
        )
    geometry_types = sorted(
        {
            feature.get("geometry", {}).get("type")
            for feature in features
            if feature.get("geometry")
        }
    )
    invalid_geometry_count = sum(
        1
        for feature in features
        if feature.get("geometry") is not None
        and not isinstance(feature.get("geometry"), dict)
    )
    if invalid_geometry_count:
        warnings.append("invalid_geometry")
    source_meta = feature_collection.get("meta") or {}
    canonical = json.dumps(feature_collection, ensure_ascii=False, sort_keys=True)
    return {
        "role": role,
        "name": name,
        "object_count": len(features),
        "complete": bool(
            source_meta.get("complete", not source_meta.get("truncated", False))
        ),
        "truncated": bool(source_meta.get("truncated", False)),
        "fields": fields,
        "geometry_types": geometry_types,
        "crs": feature_collection.get("crs", {})
        .get("properties", {})
        .get("name", "EPSG:4326"),
        "warnings": warnings,
        "revision": source_meta.get("revision")
        or "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


class RequirementsResolution:
    def __init__(
        self,
        *,
        layers: dict[str, dict[str, Any]],
        effective_requirements: DeclaredRequirements,
        resolved: list[ResolvedRequirement],
        missing: list[str],
        profiles: dict[str, dict[str, Any]],
        role_layers: dict[str, str],
        selected_fields: dict[str, str],
    ) -> None:
        self.layers = layers
        self.effective_requirements = effective_requirements
        self.resolved = resolved
        self.missing = missing
        self.profiles = profiles
        self.role_layers = role_layers
        self.selected_fields = selected_fields

    @property
    def executable(self) -> bool:
        return not self.missing


class ComplianceDataGate:
    """Resolve plan roles to concrete layers/fields without LLM guesses."""

    def resolve(
        self,
        plan: CheckPlan,
        layers: dict[str, dict[str, Any]],
        effective_requirements: DeclaredRequirements,
    ) -> RequirementsResolution:
        mutable_layers = deepcopy(layers)
        layer_lookup = {normalize_name(name): name for name in layers}
        resolved: list[ResolvedRequirement] = []
        missing: list[str] = []
        profiles: dict[str, dict[str, Any]] = {}
        role_layers: dict[str, str] = {}
        selected_fields: dict[str, str] = {}
        for requirement in effective_requirements.layers:
            layer_name = layer_lookup.get(normalize_name(requirement.entity))
            if layer_name is None:
                resolved.append(
                    ResolvedRequirement(
                        role=requirement.role,
                        requirement_type="layer",
                        resolved=False,
                        reason="layer_not_found",
                    )
                )
                if requirement.required:
                    missing.append(f"layer:{requirement.role}")
                continue
            profile = describe_scenario_layer(
                requirement.role, layer_name, layers[layer_name]
            )
            profiles[requirement.role] = profile
            role_layers[requirement.role] = layer_name
            geometry_ok = not requirement.geometry_types or set(
                profile["geometry_types"]
            ) <= set(requirement.geometry_types)
            complete = profile["complete"] and not profile["truncated"]
            ok = geometry_ok and complete
            reason = None
            if not geometry_ok:
                reason = "unsupported_geometry_type"
            elif not complete:
                reason = "incomplete_source_layer"
            resolved.append(
                ResolvedRequirement(
                    role=requirement.role,
                    requirement_type="layer",
                    resolved=ok,
                    layer=layer_name,
                    reason=reason,
                )
            )
            if requirement.required and not ok:
                missing.append(f"layer:{requirement.role}:{reason}")

        for requirement in effective_requirements.attributes:
            layer_name = role_layers.get(requirement.on)
            if not layer_name:
                if requirement.required:
                    missing.append(f"attribute:{requirement.role}:layer_unresolved")
                resolved.append(
                    ResolvedRequirement(
                        role=requirement.role,
                        requirement_type="attribute",
                        resolved=False,
                        reason="layer_unresolved",
                    )
                )
                continue
            profile = profiles[requirement.on]
            fields = {item["name"]: item for item in profile["fields"]}
            selection: tuple[AttributeCandidate, dict[str, Any]] | None = None
            for candidate in requirement.accepts:
                field = fields.get(candidate.field)
                if (
                    field
                    and field["numeric_fill_rate"] > 0
                ):
                    selection = (candidate, field)
                    break
            if selection is None:
                resolved.append(
                    ResolvedRequirement(
                        role=requirement.role,
                        requirement_type="attribute",
                        resolved=False,
                        layer=layer_name,
                        reason="attribute_not_found",
                    )
                )
                if requirement.required:
                    missing.append(f"attribute:{requirement.role}")
                continue
            candidate, field = selection
            selected_field = candidate.field
            if candidate.derive == "height_to_floors_v1":
                selected_field = f"__derived__.{requirement.role}"
                self._derive_height_to_floors(
                    mutable_layers[layer_name], candidate.field, selected_field
                )
            fill_rate = float(field["numeric_fill_rate"])
            ok = fill_rate >= requirement.min_fill_rate
            resolved.append(
                ResolvedRequirement(
                    role=requirement.role,
                    requirement_type="attribute",
                    resolved=ok,
                    layer=layer_name,
                    field=selected_field,
                    unit=candidate.unit,
                    quality=candidate.quality,
                    derive=candidate.derive,
                    fill_rate=fill_rate,
                    reason=None if ok else "fill_rate_below_minimum",
                )
            )
            if ok:
                selected_fields[requirement.role] = selected_field
            elif requirement.required:
                missing.append(f"attribute:{requirement.role}:fill_rate")

        return RequirementsResolution(
            layers=mutable_layers,
            effective_requirements=effective_requirements,
            resolved=resolved,
            missing=missing,
            profiles=profiles,
            role_layers=role_layers,
            selected_fields=selected_fields,
        )

    @staticmethod
    def _derive_height_to_floors(
        feature_collection: dict[str, Any], source_field: str, target_field: str
    ) -> None:
        """Registered v1 conversion: 3 metres per floor, rounded down, minimum one."""

        for feature in feature_collection.get("features") or []:
            properties = feature.setdefault("properties", {})
            raw = _nested_get(properties, source_field)
            try:
                value = max(1, int(float(raw) // 3)) if raw is not None else None
            except (TypeError, ValueError):
                value = None
            properties[target_field] = value
