"""Deterministic Markdown report of one compliance run.

The report is built from the ``compliance_summary`` payload only: verdicts, counts
and evidence come from the executed templates, never from the LLM.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from src.agents.services.compilance.compliance_sources import (
    grouped_references,
    merged_sources,
    source_reference,
)

REPORT_SLOT = "compliance_report"
REPORT_TITLE = "Отчёт о проверке соответствия нормам"
REPORT_MIME_TYPE = "text/markdown"
MAX_VIOLATORS_PER_NORM = 20

_TEMPLATE_TITLES = {
    "distance_from_source": "расстояние от источника",
    "distance_table": "расстояние по таблице диапазонов атрибута",
    "presence_within": "наличие объектов в радиусе",
    "zonal_attribute_threshold": "порог атрибута в зоне",
    "zonal_ratio": "доля площади в зоне",
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
}
_VALUE_LABELS = {
    "buffered": "буфер вокруг источника",
    "source_geometry": "собственная геометрия",
    "intersects": "пересекает",
    "within": "внутри",
    "contains": "содержит",
    "matched": "объект попадает в зону",
    "not_matched": "объект не попадает в зону",
}
_UNIT_LABELS = {"count": "шт."}
_OPERATOR_LABELS = {
    "matched": "попадание в зону",
    "not_matched": "вне зоны",
}


def build_compliance_report(summary: dict[str, Any]) -> str | None:
    """Return the report, or ``None`` when no norm was actually checked."""

    results = summary.get("results") or []
    checked = [
        item
        for item in results
        if item.get("compliance_status") in {"violated", "passed"}
    ]
    if not checked:
        return None
    violated = [item for item in checked if item["compliance_status"] == "violated"]
    passed = [
        item
        for item in checked
        if item["compliance_status"] == "passed" and not _vacuous(item)
    ]
    vacuous = [item for item in checked if _vacuous(item)]
    ordered = violated + passed + vacuous
    numbers = {id(item): index for index, item in enumerate(ordered, 1)}
    groups = [item for item in ordered if _equivalents(item)]
    equivalent_count = sum(len(_equivalents(item)) for item in groups)
    not_checked = len(results) - len(checked)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        "## Сводка",
        "",
        "| Показатель | Количество |",
        "| --- | ---: |",
        f"| Норм проверено | {len(checked) + equivalent_count} |",
        f"| Не прошли проверку | {len(violated)} |",
        f"| Прошли проверку | {len(passed) + len(vacuous)} |",
        f"| из них без применимых объектов | {len(vacuous)} |",
        f"| Проверены как эквивалентные | {equivalent_count} |",
        f"| Не удалось проверить | {not_checked} |",
    ]
    skipped = int(summary.get("skipped_without_plan") or 0)
    if skipped:
        lines.append(f"| Пропущено без исполнимого плана | {skipped} |")
    lines += [
        "",
        "Счётчики объектов указаны для каждой нормы отдельно; их нельзя "
        "складывать в число уникальных объектов.",
    ]

    lines += ["", "## Не прошли проверку", ""]
    lines += _norm_sections(violated, numbers) or ["_Нет._"]
    lines += ["", "## Прошли проверку", ""]
    lines += _norm_sections(passed, numbers) or ["_Нет._"]
    if vacuous:
        lines += [
            "",
            "## Прошли без применимых объектов",
            "",
            "В слоях сценария нет объектов, к которым относится норма, поэтому "
            "нарушений найти было не на чем. Это формальное прохождение, а не "
            "подтверждение соответствия: проверьте, что нужные объекты есть в "
            "сценарии и распознаны правильно.",
            "",
        ]
        lines += _norm_sections(vacuous, numbers)
    lines += ["", "## Проверены как эквивалентные", ""]
    if groups:
        lines += [
            "Эквивалентные нормы задают одинаковую проверку на тех же данных. "
            "Она выполнена один раз, и её результат относится ко всем нормам группы.",
            "",
        ]
        for item in groups:
            lines += _equivalent_group(item, numbers[id(item)])
    else:
        lines.append("_Нет._")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).rstrip() + "\n"


def report_filename(scenario_id: int | str, created_at: datetime) -> str:
    return f"{REPORT_SLOT}_{scenario_id}_{created_at:%Y%m%d-%H%M}.md"


def _equivalents(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Sources merged into this check, excluding the one that was executed."""

    return merged_sources(result.get("source") or {})


def _vacuous(result: dict[str, Any]) -> bool:
    """Passed only because the scenario has no object the norm applies to."""

    return result["compliance_status"] == "passed" and "no_applicable_objects" in (
        result.get("warnings") or []
    )


def _norm_sections(results: list[dict[str, Any]], numbers: dict[int, int]) -> list[str]:
    lines: list[str] = []
    for result in results:
        lines += _norm_section(result, numbers[id(result)])
    return lines


def _norm_section(result: dict[str, Any], number: int) -> list[str]:
    source = result.get("source") or {}
    coverage = result.get("coverage") or {}
    counts = result.get("summary") or {}
    partial = result.get("verification_status") == "partial"
    lines = [f"### {number}. {source_reference(source)}", ""]

    vacuous = _vacuous(result)
    if result["compliance_status"] == "violated":
        status = "не прошла проверку"
    elif vacuous:
        status = "прошла формально: применимых объектов в сценарии нет"
    else:
        status = "прошла проверку"
    if partial:
        status += " (проверена частично, вывод относится только к проверенным объектам)"
    lines.append(f"- **Статус:** {status}")
    requirement = _one_line(source.get("extraction_text"))
    if requirement:
        lines.append(f"- **Требование:** {requirement}")
    if vacuous:
        lines.append("- **Объекты:** применимых — 0")
    else:
        lines.append(
            f"- **Объекты:** применимых — {coverage.get('applicable_objects', 0)}, "
            f"проверено — {coverage.get('checked_objects', 0)}, "
            f"не проверено — {coverage.get('unchecked_objects', 0)}; "
            f"заполненность данных — {_percent(coverage.get('fill_rate'))}"
        )
        lines.append(
            f"- **Результат:** с нарушением — {counts.get('violated_objects', 0)}, "
            f"без нарушений — {counts.get('passed_objects', 0)}"
        )
    lines += _parameters(result)
    equivalents = _equivalents(result)
    if equivalents:
        lines.append(
            f"- **Эквивалентные нормы:** {grouped_references(equivalents)} "
            "(см. раздел «Проверены как эквивалентные»)"
        )
    lines += _violators(result)
    lines.append("")
    return lines


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


def _violators(result: dict[str, Any]) -> list[str]:
    violations = [item for item in result.get("evidence") or [] if item.get("violated")]
    if not violations:
        return []
    total = int((result.get("summary") or {}).get("violated_objects") or 0)
    lines = [
        "",
        "| № | Объект | Значение | Условие | Связанные объекты |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(violations[:MAX_VIOLATORS_PER_NORM], 1):
        obj = item.get("object_ref") or {}
        name = obj.get("name") or obj.get("id") or "—"
        if obj.get("id") and obj.get("id") != name:
            name += f" (`{obj['id']}`)"
        lines.append(
            f"| {index} | {_cell(name)} | {_cell(_measured(item))} "
            f"| {_cell(_condition(item))} | {_cell(_related(item))} |"
        )
    rest = max(total, len(violations)) - MAX_VIOLATORS_PER_NORM
    if rest > 0:
        lines += [
            "",
            f"…и ещё объектов с нарушением: {rest}. Полный список — на карте.",
        ]
    return lines


def _measured(item: dict[str, Any]) -> str:
    value = item.get("measured_value")
    if value is None:
        return "—"
    return f"{_number(value)} {_unit(item.get('unit'))}".strip()


def _unit(value: Any) -> str:
    return _UNIT_LABELS.get(value, value or "")


def _condition(item: dict[str, Any]) -> str:
    operator = item.get("operator")
    if operator in _OPERATOR_LABELS:
        text = _OPERATOR_LABELS[operator]
        if item.get("radius_m"):
            text += f" ({_number(item['radius_m'])} м)"
        return text
    threshold = item.get("threshold")
    if operator is None or threshold is None:
        return "—"
    return f"{operator} {_number(threshold)} {_unit(item.get('unit'))}".strip()


def _related(item: dict[str, Any]) -> str:
    refs = [
        *(item.get("generator_refs") or []),
        *(item.get("zone_refs") or []),
    ]
    for single in ("generator_ref", "zone_ref"):
        if item.get(single) and item[single] not in refs:
            refs.append(item[single])
    names = list(dict.fromkeys(ref.get("name") or ref.get("id") for ref in refs))
    names = [name for name in names if name]
    if not names:
        return "—"
    text = "; ".join(names[:3])
    if len(names) > 3:
        text += f" и ещё {len(names) - 3}"
    return text


def _equivalent_group(result: dict[str, Any], number: int) -> list[str]:
    if result["compliance_status"] == "violated":
        verdict = "не прошла проверку"
    elif _vacuous(result):
        verdict = "прошла формально (применимых объектов нет)"
    else:
        verdict = "прошла проверку"
    lines = [
        f"### Группа нормы № {number}: {source_reference(result.get('source') or {})}",
        "",
        f"- **Результат:** {verdict}; объектов с нарушением — "
        f"{(result.get('summary') or {}).get('violated_objects', 0)}",
        "- **Эквивалентные нормы:**",
    ]
    for item in _equivalents(result):
        line = f"  - {source_reference(item)}"
        requirement = _one_line(item.get("extraction_text"))
        if requirement:
            line += f" — {requirement}"
        lines.append(line)
    lines += ["", f"Всего норм в группе: {1 + len(_equivalents(result))}.", ""]
    return lines


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _cell(value: str) -> str:
    return _one_line(value).replace("|", "\\|")


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)
