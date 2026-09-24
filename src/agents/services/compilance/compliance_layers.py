"""One map layer of the objects a compliance run checked and found compliant.

Violations reach the map norm by norm; objects without any violation had no layer
at all, so a run that found nothing left the user with an empty map. This layer
closes that gap: every checked object that passed at least one norm and violated
none, once, with the norms it passed (equivalent norms included).
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.services.compilance.compliance_sources import source_references
from src.agents.services.service_entities.compliance import ComplianceResult

PASSED_OBJECTS_LAYER = "Объекты без нарушений"

# Per-norm verdict fields of a result feature; the merged object has its own.
_PER_NORM_FIELDS = frozenset(
    {
        "compliance_evidence",
        "compliance_status",
        "restriction_evidence",
        "restriction_id",
        "verification_status",
    }
)


def _object_key(feature: dict[str, Any]) -> str:
    properties = feature.get("properties") or {}
    ref = properties.get("object_ref")
    if isinstance(ref, dict) and ref.get("id"):
        return str(ref["id"])
    return json.dumps(feature.get("geometry"), sort_keys=True)


def passed_objects_layer(results: list[ComplianceResult]) -> dict[str, Any] | None:
    """Merge every norm's passed objects minus any object violated anywhere.

    Returns a FeatureCollection, or ``None`` when no checked object is compliant.
    """
    violated = {
        _object_key(feature)
        for result in results
        for feature in (result.violated_features or {}).get("features") or []
    }
    merged: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.compliance_status not in {"passed", "violated"}:
            continue
        references = source_references(result.source)
        for feature in (result.passed_features or {}).get("features") or []:
            key = _object_key(feature)
            if key in violated:
                continue
            if key not in merged:
                properties = {
                    name: value
                    for name, value in (feature.get("properties") or {}).items()
                    if name not in _PER_NORM_FIELDS
                }
                merged[key] = {
                    "type": "Feature",
                    "geometry": feature.get("geometry"),
                    "properties": {
                        **properties,
                        "compliance_status": "passed",
                        "passed_norms": [],
                    },
                }
            norms = merged[key]["properties"]["passed_norms"]
            for reference in references:
                if reference not in norms:
                    norms.append(reference)
    if not merged:
        return None
    return {"type": "FeatureCollection", "features": list(merged.values())}
