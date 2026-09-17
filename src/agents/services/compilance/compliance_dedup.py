"""Group equivalent checks without discarding their normative sources."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.agents.services.compilance.compliance_catalog import ComplianceCatalogResolver
from src.agents.services.compilance.compliance_registry import (
    DEFAULT_COMPLIANCE_REGISTRY,
)
from src.agents.services.service_entities.compliance import DeclaredRequirements


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(plan, params, requirements) -> str:
    layers = {}
    for item in requirements.layers:
        value = item.model_dump(mode="json", exclude={"role"})
        value["entity"] = " ".join(value["entity"].casefold().split())
        value["geometry_types"] = sorted(value["geometry_types"])
        layers[item.role] = value
    attributes = {}
    for item in requirements.attributes:
        value = item.model_dump(mode="json", exclude={"role"})
        value["on"] = layers[value["on"]]
        # Candidate order is meaningful: it controls attribute selection.
        attributes[item.role] = value
    value = params.model_dump(mode="json")
    for key in ("source_layer", "objects_layer", "zones_layer"):
        if key in value:
            value[key] = layers[value[key]]
    for key in ("targets", "required_neighbor_layers"):
        if key in value:
            value[key] = sorted((layers[role] for role in value[key]), key=_json)
    if "attribute_role" in value:
        value["attribute_role"] = attributes[value["attribute_role"]]
    if "numerator" in value:
        value["numerator"]["layer"] = layers[value["numerator"]["layer"]]
    threshold = value.get("threshold_source")
    if threshold and threshold["kind"] == "attribute_role":
        threshold["role"] = attributes[threshold["role"]]
    # Include unused requirements too: they can still affect the data gate.
    return _json(
        dict(
            schema=plan.schema_version,
            template=plan.template,
            version=plan.template_version,
            params=value,
            layers=sorted(layers.values(), key=_json),
            attributes=sorted(attributes.values(), key=_json),
        )
    )


@dataclass
class CheckGroup:
    plan: dict[str, Any]
    sources: list[dict[str, Any]] = field(default_factory=list)


async def group_checks(
    plans, mcp_client, scenario_id, *, resolver=None
) -> list[CheckGroup]:
    """Resolve names through Urban API before comparing roles and all parameters.

    An unavailable/ambiguous catalogue or incomplete plan disables deduplication
    for that plan. It remains available to the normal executor and its diagnosis.
    """
    resolver = resolver or ComplianceCatalogResolver()
    groups = []
    seen = {}
    for raw in plans:
        key = None
        try:
            plan, params = DEFAULT_COMPLIANCE_REGISTRY.validate_plan(raw)
            effective = DEFAULT_COMPLIANCE_REGISTRY.effective_requirements(plan)
            if not effective["missing_registry_roles"]:
                resolution = await resolver.resolve(
                    mcp_client,
                    scenario_id,
                    DeclaredRequirements(
                        layers=effective["layers"], attributes=effective["attributes"]
                    ),
                )
                if resolution.executable:
                    key = _fingerprint(plan, params, resolution.requirements)
        except (KeyError, TypeError, ValueError):
            # The executor owns validation errors; never hide a malformed plan.
            pass
        source = dict(raw.get("source") or {})
        if key is not None and key in seen:
            group = seen[key]
            if source not in group.sources:
                group.sources.append(source)
        else:
            group = CheckGroup(plan=raw, sources=[source])
            groups.append(group)
            if key is not None:
                seen[key] = group
    return groups
