"""Public layer attributes. Calculation inputs and scenario-data stay untouched.

Keep explicit field sets: accepting arbitrary prefixes would expose new upstream
debug fields. Aliases retain their original names for existing map consumers.
"""

from typing import Any, Literal

LayerProfile = Literal["restrictions", "provision", "effects", "pzz"]


def _fields(names: str) -> frozenset[str]:
    return frozenset(
        name.strip().casefold() for name in names.split("|") if name.strip()
    )


_IDENTITY = _fields(
    """id | name | name_full | name_short | address | readable_address | type |
    object_id | object_name | object_type | object_type_id | object_type_name |
    physical_object_id | physical_object_name | physical_object_type |
    physical_object_type_id | physical_object_type_name |
    service_id | service_name | service_type | service_type_id | service_type_name |
    functional_zone_id | functional_zone_type | functional_zone_type_id |
    functional_zone_type_name | object_ref | source_layer |
    Название | Наименование | Адрес | Тип объекта | Тип сервиса"""
)
_RESTRICTIONS = _fields(
    """buffer_size | buffer_type | restriction_title | restriction_name |
    restriction_description | restriction_id | restriction_evidence |
    compliance_status | verification_status | compliance_evidence |
    origin | provenance | violated | generator_ref | generator_refs | passed_norms |
    zone_kind | applies_to | threshold | operator | unit"""
)
_PROVISION = _fields(
    """population | demand | demand_left | capacity | capacity_left | service_load |
    provision_value | distance | avg_dist | min_dist | building_index | service_index |
    supplied_demands_within | supplied_demands_without |
    carried_capacity_within | carried_capacity_without |
    Население (чел) | Спрос (чел) | Неудовлетворённый спрос (чел) |
    Вместимость (чел) | Профицит мест (чел) | Нагрузка на сервис |
    Оценка обеспеченности | Расстояние (м) | Средняя доступность до сервиса (м) |
    Минмиальное расстояне до сервиса (м) | Минимальное расстояние до сервиса (м) |
    ID здания | ID сервиса | Удовлетворённый спрос в нормативной доступности (чел) |
    Удовлетворённый спрос вне нормативной доступности (чел) |
    Обеспечено в радиусе нормативной доступности (чел) |
    Обеспечено вне радиуса нормативной доступности (чел)"""
)
_EFFECTS = (
    _PROVISION
    | _fields(
        """absolute_total | index_total | absolute_scenario_project | index_scenario_project |
    absolute_within | is_project | is_scenario_object |
    Абсолютный эффект (чел) | Индексный эффект |
    Абсолютный эффект на территории проекта | Индексный эффект на территории проекта |
    Абсолютный эффект в нормативной доступности | Проектный объект | Сценарный объект"""
    )
    | frozenset(
        f"{field}_{phase}"
        for field in (
            "supplied_demands_within",
            "supplied_demands_without",
            "us_demands_within",
            "us_demands_without",
        )
        for phase in ("before", "after")
    )
    | frozenset(
        f"{kind} спрос {area} нормативной доступности ({phase}) (чел)".casefold()
        for kind in ("Удовлетворённый", "Неудовлетворённый")
        for area in ("в", "вне")
        for phase in ("до", "после")
    )
)
_PZZ = _fields(
    """cad_num | cadastral_number | cadastral_num | Кадастровый номер | Кадастровый_номер |
    vri | vri_text | vri_name | vri_code | permitted_use_established_by_document |
    zone_code | zone_name | zone_type_id | verdict | fit | reason | resolution_basis |
    matched_vri_name | matched_vri_code | classification_status |
    ВРИ_ЕГРН | Код фактической зоны нахождения кадастра |
    Название фактической зоны нахождения кадастра | Вердикт_ПЗЗ | Причина |
    Подобранный_ВРИ | Код_подобранного_ВРИ | Основание_подбора_ВРИ |
    Топ1_возможный_ВРИ | Статус | Статус_классификации | Категория_объекта |
    PZZ_VRI_VERDICT | PZZ_REASON"""
)
_PROFILES = {
    "restrictions": _IDENTITY | _RESTRICTIONS,
    "provision": _IDENTITY | _PROVISION,
    "effects": _IDENTITY | _EFFECTS,
    "pzz": _IDENTITY | _PZZ,
}

# Compact nested type dictionaries, object references and the substantive part of
# evidence. Do not pass through full upstream records, revisions or prompt traces.
_NESTED = _IDENTITY | _fields("""code | namespace | entity_id | geometry_id | layer |
    title | description | reason | reason_code | restriction_id | origin |
    document_id | document_name | document_version | clause_id | clause_number |
    name | numbering | breadcrumb | extraction_text | provenance |
    source_layer | target_layer | generator_ref | generator_refs | zone_ref | zone_refs |
    distance_m | operation | boundary_policy | measured_value | unit | threshold |
    operator | violated | used_fields | field | quality | warnings |
    neighbor_count | numerator_area_m2 | denominator_area_m2""")


def _compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _compact_value(item)
            for key, item in value.items()
            if key.casefold() in _NESTED
        }
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    return value


def compact_layer(collection: dict, profile: LayerProfile) -> dict:
    """Copy public properties without changing geometry, IDs or calculation data."""
    allowed = _PROFILES[profile]
    features = []
    for feature in collection.get("features", []):
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            features.append(dict(feature))
            continue
        features.append(
            {
                **feature,
                "properties": {
                    key: _compact_value(value)
                    for key, value in properties.items()
                    if key.casefold() in allowed
                },
            }
        )
    return {**collection, "features": features}


def compact_layer_event(event: dict, agent: str) -> dict:
    """Apply the same policy to replayed events, including orchestrator steps."""
    content = event.get("content") or {}
    if event.get("type") == "step_event":
        inner = content.get("event")
        if isinstance(inner, dict):
            return {
                **event,
                "content": {
                    **content,
                    "event": compact_layer_event(inner, content.get("agent", "")),
                },
            }
    if event.get("type") != "feature_collection" or agent not in {
        "restrictions",
        "restriction",
        "compliance",
        "provision",
        "pzz",
    }:
        return event
    collection = content.get("feature_collection")
    if (
        not isinstance(collection, dict)
        or collection.get("type") != "FeatureCollection"
    ):
        return event
    profile = (
        "restrictions"
        if agent in {"restrictions", "restriction", "compliance"}
        else "effects" if agent == "provision" else "pzz"
    )
    return {
        **event,
        "content": {
            **content,
            "feature_collection": compact_layer(collection, profile),
        },
    }
