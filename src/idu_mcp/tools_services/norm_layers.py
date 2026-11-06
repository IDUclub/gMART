"""Deterministic per-norm GeoJSON, independent of the agents application."""

from __future__ import annotations

import math
import operator as ops
import re
from copy import deepcopy

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from src.idu_mcp.tools_services.spatial_predicates import exact_spatial_predicate

STATUS_RU = {
    "passed": "выполнено",
    "violated": "не выполнено",
    "partial": "частично",
    "not_checked": "не проверялось",
}
GROUP_PROPERTIES = [
    "norm_id",
    "norm_status",
    "norm_status_ru",
    "norm_status_reason",
    "check_type_en",
    "check_type_ru",
    "norm_reference",
    "document_name",
    "clause_number",
    "norm_text",
]
CHECK_TYPES = {
    "distance_from_source": ("distance", "расстояние"),
    "distance_table": ("distance", "расстояние"),
    "presence_within": ("distance", "расстояние"),
    "accessibility_within": ("accessibility", "доступность"),
    "zonal_ratio": ("area_ratio", "доля площади"),
    "service_provision": ("provision", "обеспеченность"),
}
ATTRIBUTE_TYPES = {
    "floors": ("floors", "этажность"),
    "storeys": ("floors", "этажность"),
    "number_of_floors": ("floors", "этажность"),
    "этажность": ("floors", "этажность"),
    "height": ("height", "высота"),
    "height_m": ("height", "высота"),
    "building_height": ("height", "высота"),
    "высота": ("height", "высота"),
    "area": ("area", "площадь"),
    "area_m2": ("area", "площадь"),
    "площадь": ("area", "площадь"),
}
COMPARE = {"<": ops.lt, "<=": ops.le, ">": ops.gt, ">=": ops.ge, "==": ops.eq}
SYMBOLS = {"<=": "≤", ">=": "≥", "==": "=", "<": "<", ">": ">"}
UNITS = {"m": "м", "count": "шт.", "floors": "эт.", "m2": "м²"}


def source_reference(source: dict) -> str:
    name = " ".join((source.get("document_name") or "").split())
    code = re.search(r"\bСП\s*(\d+(?:\.\d+)+)\b", name, re.IGNORECASE)
    label = f"СП {code.group(1)}" if code else name or "Источник не указан"
    clause = " ".join((source.get("clause_number") or "").split())
    return f"{label}, п. {clause}" if clause else label


def check_type(template: str, attribute: str | None = None) -> tuple[str, str]:
    if template in CHECK_TYPES:
        return CHECK_TYPES[template]
    field = (attribute or "").casefold().split(".")[-1]
    return ATTRIBUTE_TYPES.get(field, ("attribute_threshold", "пороговое значение"))


def geometry_features(geometry, properties: dict) -> list[dict]:
    """Flatten mixed intersections/repairs into drawable features without loss."""
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "GeometryCollection":
        return [
            feature
            for part in geometry.geoms
            for feature in geometry_features(part, properties)
        ]
    return [
        {
            "type": "Feature",
            "geometry": mapping(geometry),
            "properties": deepcopy(properties),
        }
    ]


def build_norm_layer(
    *,
    restriction_id: str,
    template: str,
    provenance: dict | None,
    zone_features: list[dict],
    violation_features: list[dict] | None = None,
    object_features: list[dict] | None = None,
    attribute: str | None = None,
    unchecked_count: int = 0,
    unchecked_reason: str | None = None,
    inventory: bool = False,
) -> dict:
    """Reusable assembler, also suitable for service_provision in a later phase.

    Callers supply zone status/reason and violation measurements; this builder
    computes the norm verdict and repeats the grouping fields on every feature.
    """
    zones = deepcopy(zone_features)
    if inventory:
        for zone in zones:
            zone["properties"].update(status="not_checked", status_reason=None)
    # A mixed geometry may be rendered as several features of the same zone.
    # Count logical zones, retaining the same zone_ref on all their parts.
    statuses = list(
        {
            str((zone["properties"].get("zone_ref") or {}).get("id", position)): zone[
                "properties"
            ]["status"]
            for position, zone in enumerate(zones)
        }.values()
    )
    violated = statuses.count("violated")
    partial = statuses.count("partial")
    if inventory:
        status, reason = "not_checked", None
    elif violated:
        status = "violated"
        reason = f"Нарушено зон: {violated} из {len(statuses)}."
        if partial or unchecked_count:
            reason += (
                f" Частичных зон: {partial}; непроверенных объектов: {unchecked_count}."
            )
    elif partial or unchecked_count:
        status = "partial"
        zone_reasons = list(
            dict.fromkeys(
                zone["properties"].get("status_reason")
                for zone in zones
                if zone["properties"].get("status_reason")
            )
        )
        reason = " ".join(filter(None, [unchecked_reason, *zone_reasons])) or (
            f"Частичных зон: {partial}; непроверенных объектов: {unchecked_count}."
        )
    else:
        status, reason = "passed", None
    source = provenance or {}
    # Accept both the CheckPlan source and the inventory's nested provenance.
    source = {**(source.get("source") or {}), **source}
    en, ru = check_type(template, attribute)
    common = {
        "norm_id": restriction_id,
        "norm_status": status,
        "norm_status_ru": STATUS_RU[status],
        "norm_status_reason": reason,
        "check_type_en": en,
        "check_type_ru": ru,
        "norm_reference": source_reference(source),
        "document_name": source.get("document_name"),
        "clause_number": source.get("clause_number"),
        "norm_text": source.get("extraction_text"),
    }
    features = []
    for feature in [
        *zones,
        *([] if inventory else violation_features or []),
        *([] if inventory else object_features or []),
    ]:
        properties = {**feature.get("properties", {}), **common}
        if properties.get("feature_kind") == "zone":
            properties["status_ru"] = STATUS_RU[properties["status"]]
            if properties["status"] in {"partial", "violated"} and not properties.get(
                "status_reason"
            ):
                raise ValueError(
                    "Для partial/violated обязательна причина статуса зоны"
                )
        features.extend(geometry_features(shape(feature["geometry"]), properties))
    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "group_properties": list(GROUP_PROPERTIES),
            "group_by": "norm_id",
            **{
                key: common[key]
                for key in (
                    "norm_id",
                    "norm_status",
                    "norm_status_ru",
                    "norm_status_reason",
                )
            },
        },
    }


def _number(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _field(properties: dict, field: str | None):
    if field in properties:
        return properties[field]
    value = properties
    for part in (field or "").split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def _ref(feature):
    return feature["properties"].get("object_ref") or {}


def _key(feature):
    return str(_ref(feature).get("id"))


def _name(feature):
    properties = feature["properties"]
    return str(
        _ref(feature).get("name") or properties.get("name") or "Объект без названия"
    )


def _measurement(value, threshold, operator, unit, label):
    deficit = (
        max(0.0, threshold - value)
        if operator in {">", ">="}
        else (
            max(0.0, value - threshold)
            if operator in {"<", "<="}
            else abs(value - threshold)
        )
    )
    suffix = UNITS.get(unit, unit or "")
    text = (
        f"{label}: {value:g} {suffix}, требуется {SYMBOLS[operator]} "
        f"{threshold:g} {suffix}"
    )
    if operator in {">", ">="}:
        text += f", не хватает {deficit:g} {suffix}"
    elif operator in {"<", "<="}:
        text += f", превышение {deficit:g} {suffix}"
    else:
        text += f", отклонение {deficit:g} {suffix}"
    if deficit == 0 and operator in {"<", ">"}:
        text += "; равенство порогу не допускается"
    return {
        "measured_value": value,
        "required_value": threshold,
        "deficit": deficit,
        "unit": unit,
        "explanation_ru": text,
    }


def build_check_layer(
    *,
    template: str,
    restriction_id: str,
    provenance: dict | None,
    zones: dict,
    objects: dict,
    predicate: str = "intersects",
    violation_when: str = "matched",
    source_objects: dict | None = None,
    required_neighbor_layers: list[str] | None = None,
    minimum_neighbors: int = 1,
    distance_m: float | None = None,
    attribute: str | None = None,
    operator: str = "<=",
    threshold: float | None = None,
    zone_threshold_attribute: str | None = None,
    unit: str | None = None,
    extra_unchecked: int = 0,
    extra_reason: str | None = None,
    applies_to: list[str] | None = None,
) -> dict:
    """Evaluate exact local zone semantics, independently of legacy baskets."""
    zone_features = zones.get("features", [])
    candidates = objects.get("features", [])
    candidate_shapes = [shape(feature["geometry"]) for feature in candidates]
    candidate_tree = STRtree(candidate_shapes)
    sources = {_key(f): f for f in (source_objects or {}).get("features", [])}
    all_geometries = [
        shape(f["geometry"]) for f in [*zone_features, *candidates] if f.get("geometry")
    ]
    needs_metric = template == "zonal_ratio" or (
        template in {"distance_from_source", "distance_table"}
        and bool(source_objects)
        and any(f["properties"].get("buffer_size") is not None for f in zone_features)
    )
    if all_geometries and needs_metric:
        crs = gpd.GeoSeries(all_geometries, crs=4326).estimate_utm_crs()
        if crs is None:
            raise ValueError("Не удалось определить метрическую CRS проверки")
        project = Transformer.from_crs(4326, crs, always_xy=True).transform
    else:
        project = lambda x, y, z=None: (x, y)
    output_zones, violations = [], []
    related: dict[str, dict] = {}
    unchecked_refs = set()
    metric_geometries = {}

    def metric_geometry(feature):
        key = id(feature)
        if key not in metric_geometries:
            metric_geometries[key] = transform(project, shape(feature["geometry"]))
        return metric_geometries[key]

    def add_object(feature, zone_ref):
        key = _key(feature)
        if key not in related:
            properties = feature["properties"]
            related[key] = {
                "type": "Feature",
                "geometry": feature["geometry"],
                "properties": {
                    **properties,
                    "feature_kind": "object",
                    "object_ref": _ref(feature),
                    "object_name": _name(feature),
                    "type": properties.get("service_type")
                    or properties.get("physical_object_type")
                    or properties.get("object_type")
                    or properties.get("source_layer"),
                    "zone_refs": [],
                },
            }
        refs = related[key]["properties"]["zone_refs"]
        if zone_ref is not None and zone_ref not in refs:
            refs.append(zone_ref)

    for position, zone in enumerate(zone_features):
        area = shape(zone["geometry"])
        props = zone.get("properties") or {}
        zone_ref = {"id": f"norm/{restriction_id}/zone/{position}"}
        source_ref = props.get("object_ref")
        radius = _number(props.get("_distance_m", props.get("buffer_size", distance_m)))
        inside = [
            candidates[index]
            for index in sorted(candidate_tree.query(area))
            if exact_spatial_predicate(candidate_shapes[index], area, predicate)
        ]
        for feature in inside:
            add_object(feature, zone_ref)
        failed, unknown = [], []
        zone_threshold = threshold
        if zone_threshold_attribute:
            zone_threshold = _number(_field(props, zone_threshold_attribute))

        def violation(feature, mode, measurement):
            geometry = area if mode == "zone" else shape(feature["geometry"])
            if mode == "intersection":
                geometry = geometry.intersection(area)
            violation_props = {
                "feature_kind": "violation",
                "object_ref": _ref(feature),
                "object_name": _name(feature),
                "zone_ref": zone_ref,
                "violation_geometry_mode": mode,
                **measurement,
            }
            violations.extend(geometry_features(geometry, violation_props))
            add_object(feature, zone_ref)

        if template in {"distance_from_source", "distance_table"}:
            zone_threshold = radius
            if violation_when == "matched":
                failed = inside
                mode = "intersection"
            else:
                failed = [] if inside else candidates
                mode = "object"
            missing_target = violation_when == "not_matched" and not inside
            source = sources.get(str((source_ref or {}).get("id")))
            for feature in failed:
                if source is not None and radius is not None:
                    measured = metric_geometry(feature).distance(
                        metric_geometry(source)
                    )
                    measurement = _measurement(
                        measured,
                        radius,
                        ">=" if violation_when == "matched" else "<=",
                        "m",
                        f"Расстояние до объекта «{_name(source)}»",
                    )
                    if predicate != "intersects":
                        measurement[
                            "explanation_ru"
                        ] += f"; предикат объекта относительно зоны: {predicate}"
                else:
                    measurement = {
                        "measured_value": None,
                        "required_value": radius,
                        "deficit": None,
                        "unit": "m",
                        "explanation_ru": (
                            "Объект пересекает запрещённую зону."
                            if mode == "intersection"
                            else "Требуемый объект отсутствует в зоне."
                        ),
                    }
                violation(feature, mode, measurement)
            violated = bool(failed) or missing_target
            reason = (
                "Требуемый объект отсутствует в зоне."
                if missing_target
                else f"Объектов в запрещённой зоне: {len(failed)}."
            )
            required = 0 if violation_when == "matched" else 1
            zone_kind = "source_buffer" if radius is not None else "source_geometry"
        elif template in {"presence_within", "accessibility_within"}:
            zone_threshold = minimum_neighbors
            counts = {
                name: sum(f["properties"].get("source_layer") == name for f in inside)
                for name in required_neighbor_layers or []
            }
            missing = {
                name: count
                for name, count in counts.items()
                if count < minimum_neighbors
            }
            violated = bool(missing)
            source = sources.get(str((source_ref or {}).get("id")))
            if source:
                add_object(source, zone_ref)
            if violated and source:
                count = min(counts.values(), default=0)
                measurement = _measurement(
                    count, minimum_neighbors, ">=", "count", "Соседей в зоне"
                )
                measurement["explanation_ru"] += "; недостаточно в слоях: " + ", ".join(
                    missing
                )
                violation(source, "object", measurement)
            reason = "Недостаточно соседей: " + "; ".join(
                f"{name}: {count}, требуется ≥ {minimum_neighbors}"
                for name, count in missing.items()
            )
            required, zone_kind = minimum_neighbors, (
                "accessibility_buffer"
                if template == "accessibility_within"
                else "neighbor_buffer"
            )
        elif template == "zonal_ratio":
            metric_area = transform(project, area)
            denominator = metric_area.area
            if denominator <= 0:
                unknown = [zone]
                violated = False
            else:
                clipped = [metric_geometry(f).intersection(metric_area) for f in inside]
                value = unary_union(clipped).area / denominator * 100
                violated = not COMPARE[operator](value, float(zone_threshold))
                if violated:
                    violation(
                        zone,
                        "zone",
                        _measurement(
                            value, float(zone_threshold), operator, "%", "Доля площади"
                        ),
                    )
            reason, required, zone_kind = (
                "Доля площади не соответствует нормативному порогу.",
                None,
                "functional_zone",
            )
        else:
            for feature in inside:
                value = _number(_field(feature["properties"], attribute))
                if value is None or zone_threshold is None:
                    unknown.append(feature)
                elif not COMPARE[operator](value, zone_threshold):
                    failed.append(feature)
                    _, label = check_type(template, attribute)
                    violation(
                        feature,
                        (
                            "intersection"
                            if template == "zonal_attribute_threshold"
                            else "object"
                        ),
                        _measurement(
                            value, zone_threshold, operator, unit, label.capitalize()
                        ),
                    )
            violated = bool(failed)
            reason = f"Нарушают порог объектов: {len(failed)} из {len(inside)}."
            required, zone_kind = None, (
                "functional_zone"
                if template == "zonal_attribute_threshold"
                else "project_territory"
            )
        for feature in unknown:
            unchecked_refs.add(_key(feature))
        if violated:
            status = "violated"
            if unknown:
                reason += f" Не проверено объектов: {len(unknown)}."
        elif unknown:
            status = "partial"
            reason = (
                f"Не проверено {len(unknown)} из {len(inside) if template != 'zonal_ratio' else 1} объектов: "
                f"нет числового значения {attribute or 'площади зоны'}, порога или корректной геометрии."
            )
        else:
            status, reason = "passed", None
        zone_operator = operator
        if template in {"presence_within", "accessibility_within"}:
            zone_operator = ">="
        elif template in {"distance_from_source", "distance_table"}:
            zone_operator = (
                (">=" if violation_when == "matched" else "<=")
                if radius is not None
                else violation_when
            )
        output_zones.extend(
            geometry_features(
                area,
                {
                    "feature_kind": "zone",
                    "zone_ref": zone_ref,
                    "status": status,
                    "status_reason": reason,
                    "zone_kind": zone_kind,
                    "applies_to": applies_to
                    or required_neighbor_layers
                    or sorted(
                        {f["properties"].get("source_layer", "") for f in candidates}
                    ),
                    "buffer_size": radius,
                    "operator": zone_operator,
                    "threshold": zone_threshold,
                    "unit": unit or ("m" if radius is not None else None),
                    "source_object_ref": source_ref,
                    "objects_inside": len(inside),
                    "objects_required": required,
                },
            )
        )
    # Objects outside every zone are still useful evidence for absent targets;
    # an attribute check without a territory must never fabricate that territory.
    if template == "object_attribute_threshold" and not zone_features:
        for feature in candidates:
            add_object(feature, None)
        extra_unchecked += len(candidates) or 1
        extra_reason = "Не передана территория проекта; зона проверки не определена."
    return build_norm_layer(
        restriction_id=restriction_id,
        template=template,
        provenance=provenance,
        zone_features=output_zones,
        violation_features=violations,
        object_features=list(related.values()),
        attribute=attribute,
        unchecked_count=len(unchecked_refs) + extra_unchecked,
        unchecked_reason=extra_reason,
    )
