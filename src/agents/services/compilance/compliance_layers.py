"""Per-norm output; geometry from Check* is authoritative."""

from copy import deepcopy

from src.agents.services.compilance.compliance_sources import source_reference

STATUS_RU = {
    "passed": "выполнено",
    "violated": "не выполнено",
    "partial": "частично",
    "not_checked": "не проверялось",
}

CHECK_TYPES = {
    "distance_from_source": ("distance", "расстояние"),
    "distance_table": ("distance", "расстояние"),
    "presence_within": ("presence", "наличие"),
    "zonal_attribute_threshold": ("attribute_threshold", "пороговое значение"),
    "object_attribute_threshold": ("attribute_threshold", "пороговое значение"),
    "zonal_ratio": ("area_ratio", "доля площади"),
    "accessibility_within": ("accessibility", "доступность"),
    "service_provision": ("provision", "обеспеченность"),
}


def result_status(result):
    if result.compliance_status == "violated":
        return "violated"
    if result.verification_status == "complete":
        return "passed"
    return "partial"


def norm_layer(execution, zone=None):
    """Consume MCP output; assemble equivalent output for ObjectEffects only."""
    result = execution.result
    layer = deepcopy(execution.norm_layer)
    status = result_status(result)
    reason = None
    if status == "violated":
        reason = f"Нарушений: {result.summary.violated_objects}."
    elif status == "partial":
        reason = (
            "; ".join(result.missing_requirements + result.warnings)
            or f"Не проверено объектов: {result.coverage.unchecked_objects}."
        )
    if layer is None and (
        result.verification_status == "complete" or execution.effects_tool_calls
    ):
        layer = deepcopy(getattr(zone, "norm_layer", None)) or {
            "type": "FeatureCollection",
            "features": [],
            "meta": {},
        }
        if execution.effects_tool_calls:
            zone_features = list(layer["features"])
            zone_refs = [
                item.get("properties", {}).get("zone_ref") for item in zone_features
            ]
            zone_refs = [ref for ref in zone_refs if ref is not None]
            for kind, collection in (
                ("violation", result.violated_features),
                ("object", result.passed_features),
            ):
                for feature in (collection or {}).get("features", []):
                    item = deepcopy(feature)
                    evidence = (item.get("properties") or {}).get(
                        "compliance_evidence"
                    ) or []
                    measured = evidence[0].get("measured_value") if evidence else None
                    required = evidence[0].get("threshold") if evidence else None
                    item.setdefault("properties", {}).update(
                        feature_kind="object",
                        zone_refs=zone_refs,
                        object_name=(item.get("properties") or {}).get("name"),
                        measured_value=measured,
                        required_value=required,
                        deficit=(
                            max(0, required - measured)
                            if measured is not None and required is not None
                            else None
                        ),
                        unit="share",
                        violation_geometry_mode="zone",
                        explanation_ru=f"Обеспеченность: {measured}; требуется ≥ {required}.",
                    )
                    layer["features"].append(item)
                    if kind == "violation":
                        for area in zone_features:
                            layer["features"].append(
                                {
                                    "type": "Feature",
                                    "geometry": area["geometry"],
                                    "properties": {
                                        **item["properties"],
                                        "feature_kind": "violation",
                                        "zone_ref": (area.get("properties") or {}).get(
                                            "zone_ref"
                                        ),
                                    },
                                }
                            )
    if layer is None:
        result.status, result.status_reason = status, reason
        return None
    meta = layer.setdefault("meta", {})
    scope_unknown = any(
        warning.startswith("scope:without_value=")
        and warning != "scope:without_value=0"
        for warning in result.warnings
    )
    if execution.norm_layer is not None:
        status = meta.get("norm_status", status)
        reason = meta.get("norm_status_reason", reason)
        if status == "passed" and result.coverage.unchecked_objects:
            status, reason = (
                "partial",
                f"Не проверено объектов: {result.coverage.unchecked_objects}.",
            )
        if scope_unknown:
            reason = " ".join(
                filter(
                    None,
                    [
                        reason,
                        f"Не проверено объектов с неизвестным условием применения: {result.coverage.unchecked_objects}.",
                    ],
                )
            )
    source = result.source
    first = next(iter(layer.get("features", [])), {}).get("properties", {})
    check_en, check_ru = CHECK_TYPES.get(
        result.template, (result.template, result.template)
    )
    common = {
        "norm_id": result.restriction_id,
        "norm_status": status,
        "norm_status_ru": STATUS_RU[status],
        "norm_status_reason": reason,
        "norm_reference": source_reference(source),
        "document_name": source.get("document_name"),
        "clause_number": source.get("clause_number"),
        "norm_text": source.get("extraction_text"),
        "check_type_en": first.get(
            "check_type_en", meta.get("check_type_en", check_en)
        ),
        "check_type_ru": first.get(
            "check_type_ru", meta.get("check_type_ru", check_ru)
        ),
    }
    result.check_type_en = common["check_type_en"]
    result.check_type_ru = common["check_type_ru"]
    meta.update(
        group_by="norm_id",
        group_properties=list(common),
        check_type_en=result.check_type_en,
        check_type_ru=result.check_type_ru,
        **{
            key: common[key]
            for key in (
                "norm_id",
                "norm_status",
                "norm_status_ru",
                "norm_status_reason",
            )
        },
    )
    features = []

    def append(feature, geometry):
        if not geometry:
            return
        if geometry.get("type") == "GeometryCollection":
            for child in geometry.get("geometries", []):
                append(feature, child)
        else:
            features.append({**feature, "geometry": geometry})

    for feature in layer.get("features", []):
        properties = feature.setdefault("properties", {})
        properties.update(common)
        if execution.norm_layer is None and properties.get("feature_kind") == "zone":
            properties.update(
                status=status, status_ru=STATUS_RU[status], status_reason=reason
            )
        elif (
            scope_unknown
            and properties.get("feature_kind") == "zone"
            and properties.get("status") == "passed"
        ):
            properties.update(
                status="partial", status_ru=STATUS_RU["partial"], status_reason=reason
            )
        append(feature, feature.get("geometry"))
    layer["features"] = features
    result.status, result.status_reason = status, reason
    return layer
