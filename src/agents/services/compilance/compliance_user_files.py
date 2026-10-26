"""Pipeline pieces of a compliance check of user files.

``RestrictionParserService`` runs the same pipeline for a scenario and for uploaded
files; this module holds what only the files need: the boundary that locates the
design, the entities the selected norms need (the candidates of the mapping), the
pending mapping question, and the summary, answer and report lines about the files.
"""

from __future__ import annotations

from typing import Any

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from src.agents.services.compilance.compliance_entity_mapping import (
    _BLOCK_ZONES,
    FGISTP_ZONE_TYPES,
    EntityLayers,
    EntityMapping,
    EntityTarget,
)
from src.agents.services.compilance.compliance_sources import source_reference
from src.agents.services.compilance.compliance_uploads import UserFiles
from src.agents.services.compilance.compliance_user_layers import (
    ALL_ZONES,
    target_roles,
)
from src.agents.services.service_entities.compliance import (
    CheckPlan,
    DeclaredRequirements,
)

USER_FILES_STATUS = "user_files"
_ZONE_TARGETS = tuple(
    EntityTarget(zone, "functional_zone")
    for zone in dict.fromkeys(
        [*(zone for zone, _ in _BLOCK_ZONES.values()), *FGISTP_ZONE_TYPES]
    )
    if zone is not None
)
_ROLE_TITLES = {"project": "проект", "context": "окружение", "border": "граница"}
_UNSUPPORTED_TITLES = {
    "boundaries": "границы и территории",
    "zouit": "ЗОУИТ",
    "heritage": "объекты культурного наследия",
    "emergency": "зоны риска ЧС",
    "mineral_resources": "участки недр и месторождения полезных ископаемых",
    "land_categories": "категории земель",
    "special_legal_regime": "территории и зоны с особым юридическим статусом",
    "water_bodies": "поверхностные водные объекты",
    "gisogd": "слои ГИСОГД (красные линии, участки и планировочные ограничения)",
}


def _unsupported_text(reason: str | None) -> str | None:
    if not reason or not reason.startswith("not_supported_entity:"):
        return None
    group = reason.split(":", 1)[1]
    return "распознано, нормами пока не используется: " + _UNSUPPORTED_TITLES.get(
        group, group
    )


def boundary_geometry(files: UserFiles) -> dict[str, Any] | None:
    """Where the design is: its boundary, else the hull of its objects."""
    geometries = []
    for role in ("border", "project"):
        for layer in files.by_role(role):
            for feature in layer.features:
                try:
                    geometry = shape(feature["geometry"])
                except (ValueError, TypeError, KeyError, AttributeError):
                    continue
                if not geometry.is_empty:
                    geometries.append(
                        geometry if geometry.is_valid else geometry.buffer(0)
                    )
        if geometries:
            union = unary_union(geometries)
            return mapping(union if role == "border" else union.convex_hull)
    return None


def needed_targets(
    plans: list[tuple[CheckPlan, Any, DeclaredRequirements]],
) -> tuple[list[EntityTarget], list[EntityTarget]]:
    """The entities of the selected norms: all of them, and the non-checked ones.

    The first list is the mapping's candidates; the second — entities some norm
    reads as a source, neighbour or zone — tells which surroundings matter.
    """
    needed: list[EntityTarget] = []
    sources: list[EntityTarget] = []
    for plan, params, requirements in plans:
        targets = set(target_roles(plan, params))
        for layer in requirements.layers:
            if (
                layer.entity_type.value == "functional_zone"
                and layer.entity == ALL_ZONES
            ):
                entities = list(_ZONE_TARGETS)
            else:
                entities = [EntityTarget(layer.entity, layer.entity_type.value)]
            needed.extend(entities)
            if layer.role not in targets:
                sources.extend(entities)
    return list(dict.fromkeys(needed)), list(dict.fromkeys(sources))


def describe_missing(code: str) -> str:
    """Human text for ``UserLayerSource.missing`` codes."""
    if code == "user_files:scenario_only_template":
        return (
            "обеспеченность считается только по сценарию Urban API, объекты файлов "
            "в расчёт не попадают"
        )
    if code.startswith("user_files:no_project_objects:"):
        names = code.removeprefix("user_files:no_project_objects:")
        return f"в файлах проекта нет объектов, к которым относится норма ({names})"
    if code.startswith("user_files:no_objects:"):
        name = code.removeprefix("user_files:no_objects:")
        return f"нет объектов «{name}» ни в проекте, ни в окружении"
    return code


def not_checked_entry(
    raw_plan: dict[str, Any], sources: list[dict[str, Any]], missing: list[str]
) -> dict[str, Any]:
    source = raw_plan.get("source") or {}
    return {
        "restriction_id": source.get("restriction_id"),
        "label": source_reference(source),
        "sources": sources,
        "template": raw_plan.get("template"),
        "requirement": " ".join((source.get("extraction_text") or "").split()),
        "missing": missing,
        "reasons": [describe_missing(code) for code in missing],
    }


def pending_record(
    *,
    files: UserFiles,
    mapping_result: EntityMapping,
    candidates: list[EntityTarget],
    user_query: str,
    scope: dict[str, Any] | None,
    scenario_id: int | None,
) -> dict[str, Any]:
    """What the next message needs to continue the check it answers."""
    return {
        "upload_ids": list(files.upload_ids),
        "user_query": user_query,
        "scope": scope,
        "scenario_id": scenario_id,
        "mapping": mapping_result.to_dict(),
        "candidates": [target.to_dict() for target in candidates],
    }


def pending_candidates(pending: dict[str, Any]) -> list[EntityTarget]:
    return [
        target
        for target in (
            EntityTarget.from_dict(item) for item in pending.get("candidates") or []
        )
        if target is not None
    ]


def user_files_summary(
    *,
    files: UserFiles,
    mapping_result: EntityMapping,
    layers: EntityLayers,
    not_checked: list[dict[str, Any]],
    overlay: bool,
) -> dict[str, Any]:
    return {
        "mode": "overlay" if overlay else "standalone",
        "upload_ids": list(files.upload_ids),
        "layers": files.summary(),
        "mapping": [item.to_dict() for item in mapping_result.proposals],
        "entities": layers.counts(),
        "has_border": layers.border is not None,
        "not_checked": not_checked,
    }


def mapping_event(
    files: UserFiles, mapping_result: EntityMapping, overlay: bool
) -> dict[str, Any]:
    """Structured SSE event: the files' layers and how their objects were mapped."""
    return {
        "type": "user_files",
        "content": {
            "mode": "overlay" if overlay else "standalone",
            "upload_ids": list(files.upload_ids),
            "layers": files.summary(),
            "mapping": [item.to_dict() for item in mapping_result.proposals],
            "pending": [item.signature for item in mapping_result.pending],
        },
    }


def user_files_text(user_files: dict[str, Any]) -> str:
    """Paragraph of the final answer about the files: data used and norms skipped."""
    entities = user_files.get("entities") or {}
    project = sorted(
        f"{name} — {counts['project']}"
        for name, counts in entities.items()
        if counts.get("project")
    )
    lines = []
    where = (
        "поверх сценария (объекты сценария той же категории внутри границы заменены)"
        if user_files.get("mode") == "overlay"
        else "без сценария"
    )
    lines.append(f"Проверены объекты загруженных файлов {where}.")
    if project:
        lines.append("Объекты проекта: " + "; ".join(project) + ".")
    unmapped = sum(
        item["count"]
        for item in user_files.get("mapping") or []
        if not item.get("entity")
        and (item.get("roles") or {}).get("project")
        and item.get("source") != "table"
    )
    if unmapped:
        lines.append(
            f"Не сопоставлено с объектами норм и не проверялось объектов проекта: "
            f"{unmapped}."
        )
    unsupported = sum(
        item["count"]
        for item in user_files.get("mapping") or []
        if _unsupported_text(item.get("reason"))
    )
    if unsupported:
        lines.append(
            f"Распознано, нормами пока не используется объектов: {unsupported}."
        )
    not_checked = user_files.get("not_checked") or []
    if not_checked:
        labels = list(dict.fromkeys(entry["label"] for entry in not_checked))
        shown = "; ".join(labels[:5]) + ("; …" if len(labels) > 5 else "")
        lines.append(
            "Не проверено из-за отсутствия нужных объектов в файлах: "
            f"{len(not_checked)} ({shown}). Подробности — в отчёте."
        )
    return " ".join(lines)


def user_files_report(user_files: dict[str, Any] | None) -> list[str]:
    """Markdown sections of the report about the files."""
    if not user_files:
        return []
    lines = ["", "## Загруженные файлы", ""]
    lines += [
        "Проверка выполнена "
        + (
            "по объектам файлов поверх сценария: объекты сценария той же категории "
            "внутри границы проекта заменены объектами файлов."
            if user_files.get("mode") == "overlay"
            else "только по объектам загруженных файлов, без сценария."
        ),
        "",
        "| Слой | Роль | Объектов |",
        "| --- | --- | ---: |",
    ]
    for layer in user_files.get("layers") or []:
        lines.append(
            f"| {layer['name']} | {_ROLE_TITLES.get(layer['role'], layer['role'])}"
            f" | {layer['feature_count']} |"
        )
    lines += [
        "",
        "### Сопоставление с объектами норм",
        "",
        "| Теги объектов | Объектов | Объект норм | Как определено |",
        "| --- | ---: | --- | --- |",
    ]
    how = {
        "table": "по таблице",
        "llm": "моделью",
        "user": "подтверждено пользователем",
        "none": "—",
    }
    for item in user_files.get("mapping") or []:
        method = how.get(item.get("source"), "—")
        if item.get("source") == "llm":
            method += f", уверенность {round((item.get('confidence') or 0) * 100)}%"
        status = _unsupported_text(item.get("reason")) or item.get("entity") or "—"
        lines.append(f"| {item['label']} | {item['count']} | {status}" f" | {method} |")
    not_checked = user_files.get("not_checked") or []
    lines += ["", "## Не проверено: нет данных в файлах", ""]
    if not not_checked:
        lines.append("_Нет._")
    for entry in not_checked:
        lines.append(f"- **{entry['label']}** — {'; '.join(entry['reasons'])}.")
        if entry.get("requirement"):
            lines.append(f"  Требование: {entry['requirement']}")
    return lines
