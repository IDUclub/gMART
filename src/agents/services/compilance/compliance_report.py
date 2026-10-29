"""Shared deterministic descriptions for per-norm reports and follow-up answers."""

from __future__ import annotations

from typing import Any

STRICTEST_NORM_NOTICE = (
    "Часть норм содержит условия применения или разные значения для разных "
    "случаев. Для них применена самая строгая норма ко всем объектам; эти нормы "
    "необходимо проверить на дополнительные условия."
)
TRANSPORT_NOTICE = (
    "Транспортная доступность оценена приближённо: радиусом по прямой при средней "
    "скорости транспорта, без дорожной сети и расписаний. Пограничные результаты "
    "таких норм стоит проверить вручную."
)

_TEMPLATE_TITLES = {
    "distance_from_source": "расстояние от источника",
    "distance_table": "расстояние по таблице диапазонов атрибута",
    "presence_within": "наличие объектов в радиусе",
    "zonal_attribute_threshold": "порог атрибута в зоне",
    "zonal_ratio": "доля площади в зоне",
    "object_attribute_threshold": "порог атрибута объекта",
    "accessibility_within": "доступность (по буферу)",
    "service_provision": "обеспеченность по нормативу",
}
_PARAM_LABELS = {
    "source_layer": "Источник",
    "targets": "Проверяемые объекты",
    "objects_layer": "Проверяемые объекты",
    "zones_layer": "Зоны",
    "required_neighbor_layers": "Обязательные объекты рядом",
    "attribute_role": "Атрибут",
    "geometry_mode": "Геометрия источника",
    "distance_m": "Расстояние, м",
    "predicate": "Пространственное условие",
    "join_predicate": "Связь с зоной",
    "violation_when": "Нарушение",
    "minimum_neighbors": "Минимум объектов рядом",
    "operator": "Оператор",
    "threshold": "Порог",
    "threshold_source": "Порог",
    "bands": "Диапазоны",
    "numerator": "Числитель",
    "neighbors_layer": "Обязательные объекты рядом",
    "services_layer": "Сервис",
    "limit": "Доступность",
    "capacity_per_1000": "Мест на 1000 жителей",
    "residents_per_service": "Жителей на 1 объект",
    "accessibility": "Доступность",
    "min_provision": "Минимальная обеспеченность",
    "mode": "Вид доступности",
    "accessibility_mode": "Вид доступности",
}
_VALUE_LABELS = {
    "buffered": "буфер вокруг источника",
    "source_geometry": "собственная геометрия",
    "intersects": "пересекает",
    "within": "внутри",
    "contains": "содержит",
    "matched": "объект попадает в зону",
    "not_matched": "объект не попадает в зону",
    "walk": "пешеходная",
    "transport": "транспортная (приближённая оценка)",
}


def strictest_applicability(result: dict[str, Any]) -> dict[str, Any] | None:
    """The plan's ``applicability`` when the strictest value of a clause was applied."""

    plan = (result.get("source") or {}).get("check_plan") or {}
    applicability = plan.get("applicability") or {}
    if applicability.get("mode") == "strictest_variant":
        return applicability
    return None


def _strictest_lines(result: dict[str, Any]) -> list[str]:
    applicability = strictest_applicability(result)
    if applicability is None:
        return []
    lines = [
        f"- **Применена самая строгая норма:** {_one_line(applicability.get('applied'))}. "
        "Пункт задаёт условия применения или разные значения для разных случаев; "
        "проверка применила самое строгое значение ко всем объектам. Проверьте, "
        "выполняются ли для объектов сценария дополнительные условия, — по ним "
        "может действовать менее строгое значение."
    ]
    conditions = [_one_line(item) for item in applicability.get("conditions") or []]
    if conditions:
        lines.append(f"  - Условия пункта: {'; '.join(conditions)}")
    variants = [_one_line(item) for item in applicability.get("variants") or []]
    if variants:
        lines.append(f"  - Варианты нормы: {'; '.join(variants)}")
    return lines


def plan_scope(result: dict[str, Any]) -> dict[str, Any] | None:
    """The plan's ``scope``: only objects meeting the clause's condition were checked."""

    plan = (result.get("source") or {}).get("check_plan") or {}
    return plan.get("scope") or None


def transport_accessibility(result: dict[str, Any]) -> bool:
    """Whether the plan estimated a transport accessibility by a radius."""

    params = ((result.get("source") or {}).get("check_plan") or {}).get("params") or {}
    return "transport" in {params.get("mode"), params.get("accessibility_mode")}


def scope_text(scope: dict[str, Any]) -> str:
    low, high = scope.get("min"), scope.get("max")
    if low is not None and high is not None:
        floors = f"этажностью от {_number(low)} до {_number(high)}"
    elif low is not None:
        floors = f"этажностью от {_number(low)} и выше"
    else:
        floors = f"этажностью до {_number(high)}"
    return f"{_one_line(scope.get('condition'))} — объекты {floors}"


def _scope_lines(result: dict[str, Any]) -> list[str]:
    scope = plan_scope(result)
    if scope is None:
        return []
    counts = {}
    for warning in result.get("warnings") or []:
        if warning.startswith("scope:") and "=" in warning:
            key, value = warning[len("scope:") :].split("=", 1)
            counts[key] = value
    line = (
        f"- **Проверены только объекты, к которым относится значение:** "
        f"{scope_text(scope)}"
    )
    if counts:
        line += (
            f"; вне условия — {counts.get('outside_condition', 0)}, "
            f"без этажности (не проверены) — {counts.get('without_value', 0)}"
        )
    return [line]


def _transport_lines(result: dict[str, Any]) -> list[str]:
    params = ((result.get("source") or {}).get("check_plan") or {}).get("params") or {}
    if params.get("mode") == "transport":
        limit = params.get("limit") or {}
        speed = float(params.get("speed_m_per_min") or 0)
        route = (
            float(limit.get("minutes") or 0) * speed
            if limit.get("kind") == "time"
            else float(limit.get("meters") or 0)
        )
        radius = route / float(params.get("detour_factor") or 1)
        return [
            f"- **Транспортная доступность оценена приближённо:** радиус "
            f"{radius:.0f} м по прямой при средней скорости транспорта "
            f"{_number(round(speed * 0.06, 1))} км/ч; дорожная сеть и расписания не "
            "учитываются."
        ]
    if params.get("accessibility_mode") == "transport":
        access = params.get("accessibility") or {}
        return [
            "- **Транспортная доступность оценена приближённо:** "
            f"{_number(access.get('meters'))} м по прямой при средней скорости "
            "транспорта; дорожная сеть и расписания не учитываются."
        ]
    return []


_POPULATION_SOURCES = {
    "explicit": "задано при запуске проверки",
    "indicator": "индикатор Urban API, согласованный с жилым фондом",
    "housing_stock": "ёмкость жилого фонда (жилая площадь / норма на человека)",
}


def _population_lines(result: dict[str, Any]) -> list[str]:
    values = {}
    ignored = None
    for warning in result.get("warnings") or []:
        if warning.startswith("population:") and "=" in warning:
            key, value = warning[len("population:") :].split("=", 1)
            values[key] = value
        elif warning.startswith("population_indicator_ignored"):
            ignored = True
    if "source" not in values:
        return []
    line = (
        f"- **Население сценария:** {values.get('residents', '—')} чел. — "
        f"{_POPULATION_SOURCES.get(values['source'], values['source'])}"
    )
    if ignored:
        line += "; индикатор Urban API не совпадает с жилым фондом и не использован"
    return [line]


def _parameters(result: dict[str, Any]) -> list[str]:
    plan = (result.get("source") or {}).get("check_plan") or {}
    requirements = plan.get("declared_requirements") or {}
    entities = {
        layer["role"]: layer.get("entity") or layer["role"]
        for layer in requirements.get("layers") or []
    }
    template = result.get("template") or plan.get("template") or ""
    title = _TEMPLATE_TITLES.get(template, template)
    lines = [
        f"- **Параметры проверки:** шаблон «{title}» "
        f"({template}@v{result.get('template_version')})"
    ]
    for key, value in (plan.get("params") or {}).items():
        if key in _PARAM_LABELS:
            lines.append(
                f"  - {_PARAM_LABELS[key]}: {_param_value(key, value, entities)}"
            )
    for item in result.get("resolved_requirements") or []:
        if not item.get("resolved"):
            continue
        target = item.get("layer") or ""
        if item.get("requirement_type") == "attribute":
            target = f"{target}, поле «{item.get('field')}»"
            if item.get("unit"):
                target += f" ({item['unit']})"
            if item.get("quality") == "derived":
                target += f", вычислено через {item.get('derive')}"
        role = entities.get(item.get("role"), item.get("role"))
        lines.append(f"  - Данные «{role}»: {target}")
    return lines


def _param_value(key: str, value: Any, entities: dict[str, str]) -> str:
    if isinstance(value, list) and key != "bands":
        return ", ".join(entities.get(item, str(item)) for item in value)
    if isinstance(value, str):
        return entities.get(value, _VALUE_LABELS.get(value, value))
    if key == "threshold_source" and isinstance(value, dict):
        if value.get("kind") == "constant":
            return f"{_number(value.get('value'))} {value.get('unit', '')}".strip()
        return f"значение атрибута зоны «{value.get('role')}»"
    if key in {"limit", "accessibility"} and isinstance(value, dict):
        if value.get("kind") == "time":
            return f"{_number(value.get('minutes'))} мин"
        return f"{_number(value.get('meters'))} м"
    if key == "numerator" and isinstance(value, dict):
        return f"площадь «{entities.get(value.get('layer'), value.get('layer'))}»"
    if key == "bands" and isinstance(value, list):
        return "; ".join(
            f"{_number(band.get('min'))}–"
            f"{'…' if band.get('max') is None else _number(band.get('max'))}"
            f" → {_number(band.get('distance_m'))} м"
            for band in value
        )
    return _number(value)


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)
