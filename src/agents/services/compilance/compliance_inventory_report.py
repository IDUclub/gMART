"""Deterministic Markdown report of a «какие ограничения есть» run.

Built from the ``restriction_inventory`` payload only: the zones, their counts and
the norm texts come from the CheckPlans and the zone builder, never from the LLM.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.services.compilance.compliance_inventory import (
    ZONE_KIND_TITLES,
    describe_zone,
)
from src.agents.services.compilance.compliance_sources import (
    grouped_references,
    merged_sources,
    source_reference,
)

INVENTORY_REPORT_SLOT = "restriction_inventory_report"
INVENTORY_REPORT_TITLE = "Ограничения на территории сценария"

_NOT_SHOWN = {
    "no_objects": "в сценарии нет объектов, от которых действует норма",
    "unverifiable": "не хватает данных для построения зоны",
    "unsupported": "план нормы не поддерживается",
}


def build_inventory_report(summary: dict[str, Any]) -> str | None:
    """Return the report, or ``None`` when no zone was drawn."""

    zones = summary.get("zones") or []
    shown = [item for item in zones if item.get("status") == "shown"]
    if not shown:
        return None
    lines = [f"# {INVENTORY_REPORT_TITLE}", ""]
    label = (summary.get("scope") or {}).get("label")
    if label:
        lines += [f"Область — {label}.", ""]
    territory = summary.get("territory") or {}
    lines += [
        "## Сводка",
        "",
        "| Показатель | Количество |",
        "| --- | ---: |",
        f"| Норм с пространственным планом | {summary.get('total_norms', 0)} |",
        f"| Показано на карте | {len(shown)} |",
        f"| из них зон ограничения | {summary.get('restriction_zones', 0)} |",
        f"| из них зон требуемого размещения | {summary.get('required_zones', 0)} |",
        f"| Нет объектов в сценарии | {summary.get('no_objects_norms', 0)} |",
        f"| Не удалось построить | {summary.get('unverifiable_norms', 0)} |",
        f"| Не поддерживается | {summary.get('unsupported_norms', 0)} |",
        f"| Объединено одинаковых зон | {summary.get('duplicate_checks', 0)} |",
        f"| Норм без исполнимого плана | {summary.get('skipped_without_plan', 0)} |",
    ]
    if territory:
        lines += [
            f"| Документов, действующих на территории | "
            f"{territory.get('documents_in_force', 0)} |",
            f"| Документов, не действующих на ней | "
            f"{territory.get('documents_out_of_force', 0)} |",
        ]
    lines += [
        "",
        "Зона ограничения — где на указанные объекты действует запрет или предел. "
        "Зона требуемого размещения — где указанные объекты должны находиться.",
        "",
        "## Зоны на карте",
        "",
    ]
    for number, item in enumerate(shown, 1):
        lines += _zone_section(item, number)
    hidden = [item for item in zones if item.get("status") != "shown"]
    lines += ["## Не показаны на карте", ""]
    if hidden:
        lines += [
            f"- {source_reference(item.get('source') or {})} — "
            f"{_NOT_SHOWN.get(item.get('status'), item.get('status'))}"
            for item in hidden
        ]
    else:
        lines.append("_Нет._")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).rstrip() + "\n"


def _zone_section(item: dict[str, Any], number: int) -> list[str]:
    source = item.get("source") or {}
    lines = [
        f"### {number}. {source_reference(source)}",
        "",
        f"- **Вид:** {ZONE_KIND_TITLES[item['zone_kind']].lower()}",
        f"- **Территория:** {describe_zone(item)}",
        f"- **Зон на карте:** {item.get('zone_count', 0)}",
    ]
    skipped = int(item.get("skipped_objects") or 0)
    if skipped:
        lines.append(f"- **Пропущено объектов без значения для расчёта:** {skipped}")
    requirement = " ".join((source.get("extraction_text") or "").split())
    if requirement:
        lines.append(f"- **Требование:** {requirement}")
    equivalents = merged_sources(source)
    if equivalents:
        lines.append(f"- **Та же зона по нормам:** {grouped_references(equivalents)}")
    lines.append("")
    return lines
