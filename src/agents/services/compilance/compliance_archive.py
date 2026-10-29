"""Deterministic, checkpointable compliance cache and downloadable archive."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from src.__version__ import __VERSION__
from src.agents.common.files.layer_cache import default_cache_dir
from src.agents.services.compilance.compliance_layers import STATUS_RU
from src.agents.services.compilance.compliance_report import (
    STRICTEST_NORM_NOTICE,
    TRANSPORT_NOTICE,
    _parameters,
    _population_lines,
    _scope_lines,
    _strictest_lines,
    _transport_lines,
    strictest_applicability,
    transport_accessibility,
)
from src.agents.services.compilance.compliance_sources import (
    grouped_references,
    source_reference,
)
from src.agents.services.compilance.compliance_user_files import user_files_report


def _component(value):
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)).strip(" .")
    if not text or text in {".", ".."}:
        raise ValueError("Invalid cache path component")
    return text[:180]


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _cell(value):
    return (
        str(value if value is not None else "—").replace("|", "\\|").replace("\n", " ")
    )


REASON_LABELS = {
    "retrieval_failed": "ошибка получения данных",
    "missing_data": "нет необходимых слоёв или атрибутов",
    "unsupported_plan": "нет поддерживаемого исполнимого плана",
    "execution_failed": "ошибка выполнения проверки",
    "inventory": "перечень без проверки объектов",
    "other": "проверка недоступна",
}


def reason_category(item):
    """Stable categories never include entity names or downstream exception text."""
    reason = str(item.get("status_reason") or "").lower()
    if "перечень ограничений" in reason:
        return "inventory"
    if any(
        key in reason
        for key in (
            "retrieval_failed",
            "source_layers:",
            "objects_by_geometry:",
            "zones_building:",
        )
    ):
        return "retrieval_failed"
    if any(
        key in reason
        for key in ("unsupported", "исполн", "check_plan", "planner_status")
    ):
        return "unsupported_plan"
    if any(
        key in reason for key in ("execution_failed", "tool_error", "zone_build_failed")
    ):
        return "execution_failed"
    if any(
        key in reason
        for key in (
            "layer:",
            "attribute:",
            "user_files:",
            "данных",
            "объект",
            "сло",
            "атрибут",
        )
    ):
        return "missing_data"
    return "other"


def not_checked_categories(summary):
    return dict(
        Counter(
            reason_category(item)
            for item in summary.get("results", [])
            if item.get("status") == "not_checked" and not item.get("equivalent_to")
        )
    )


def complete_summary(summary, restrictions, equivalent_sources):
    """Include every fetched norm, with no duplicate execution folders."""
    for item in summary["results"]:
        if not item.get("equivalent_to"):
            item["formal_pass"] = item.get(
                "status"
            ) == "passed" and "no_applicable_objects" in (item.get("warnings") or [])
    known = {item["restriction_id"] for item in summary["results"]}
    equivalents = {
        str(source.get("restriction_id")): rid
        for rid, sources in equivalent_sources.items()
        for source in sources
        if str(source.get("restriction_id")) != rid
    }
    file_missing = {
        str(item.get("restriction_id")): item
        for item in (summary.get("user_files") or {}).get("not_checked", [])
    }
    executed = {item["restriction_id"]: item for item in summary["results"]}
    for raw in restrictions:
        plan = raw.get("check_plan") or {}
        source = (
            raw.get("source")
            or plan.get("source")
            or {
                "restriction_id": raw.get("id"),
                "document_name": (raw.get("provenance") or {}).get("name"),
                "clause_number": (raw.get("provenance") or {}).get("numbering"),
            }
        )
        rid = str(
            source.get("restriction_id")
            or raw.get("id")
            or raw.get("restriction_id")
            or "unknown"
        )
        if rid in known:
            continue
        known.add(rid)
        representative = executed.get(equivalents.get(rid)) or {}
        missing = file_missing.get(rid) or {}
        summary["results"].append(
            {
                "restriction_id": rid,
                "source": source,
                "template": plan.get("template"),
                "check_type_en": representative.get("check_type_en"),
                "check_type_ru": representative.get("check_type_ru"),
                "status": representative.get("status", "not_checked"),
                "status_reason": (
                    representative.get("status_reason")
                    if representative
                    else "; ".join(missing.get("reasons", []))
                    or raw.get("status_reason")
                    or "Нет исполнимого плана или необходимых данных"
                ),
                "equivalent_to": equivalents.get(rid),
                "archive_path": representative.get("archive_path"),
                "formal_pass": representative.get("formal_pass", False),
            }
        )
    for item in (summary.get("user_files") or {}).get("not_checked", []):
        rid = str(item.get("restriction_id") or "unknown")
        if rid not in known:
            known.add(rid)
            summary["results"].append(
                {
                    **item,
                    "status": "not_checked",
                    "archive_path": None,
                    "status_reason": item.get("reason")
                    or "Нет необходимых объектов в файлах",
                }
            )
    summary["counts"] = dict(
        Counter(
            item.get("status", "not_checked")
            for item in summary["results"]
            if not item.get("equivalent_to")
        )
    )
    summary["total_restrictions"] = len(summary["results"])
    summary["executed_norms"] = sum(
        bool(item.get("archive_path")) and not item.get("equivalent_to")
        for item in summary["results"]
    )
    summary["formal_passed_norms"] = sum(
        bool(item.get("formal_pass")) and not item.get("equivalent_to")
        for item in summary["results"]
    )


def llm_summary(summary):
    """Only deterministic summary data, without input layers, raw plans or evidence."""
    norms = [
        item
        for item in summary.get("results", [])
        if item.get("status") in {"violated", "partial", "passed"}
        and not item.get("equivalent_to")
    ]
    norms.sort(
        key=lambda item: {"violated": 0, "partial": 1, "passed": 2}[item["status"]]
    )
    limit = 50
    omitted = max(0, len(norms) - limit)
    categories = not_checked_categories(summary)
    return {
        "counts": summary.get("counts"),
        "scope": summary.get("scope"),
        "territory": summary.get("territory"),
        "formal_passed_norms": summary.get("formal_passed_norms", 0),
        "not_checked": {"count": sum(categories.values()), "categories": categories},
        "omitted_norms": omitted,
        "note": f"и ещё {omitted}" if omitted else None,
        "norms": [
            {
                "reference": source_reference(item.get("source") or {}),
                "status": item.get("status"),
                "reason": item.get("status_reason"),
                "violations": (item.get("summary") or {}).get("violated_objects", 0),
                "formal_pass": item.get("formal_pass", False),
            }
            for item in norms[:limit]
        ],
    }


def build_summary_report(summary):
    results = summary.get("results", [])
    lines = [
        "# Сводный отчёт о проверке норм",
        "",
        f"Сценарий: {summary.get('scenario_id', 'files')}. Время: {summary.get('created_at', '')}.",
        f"Область: {(summary.get('scope') or {}).get('label', 'все применимые нормы')}.",
        f"Документы на территории: {_json(summary.get('territory', {}))}",
        "",
    ]
    counts = Counter(
        item.get("status", "not_checked")
        for item in results
        if not item.get("equivalent_to")
    )
    checked = sum(
        bool(item.get("archive_path")) and not item.get("equivalent_to")
        for item in results
    )
    lines += [
        f"Проверено: {checked}. Выполнено: {counts['passed']}. Не выполнено: {counts['violated']}. "
        f"Частично: {counts['partial']}. Эквивалентных: {summary.get('duplicate_checks', 0)}. "
        f"Не запущено: {counts['not_checked']}.",
        "",
    ]
    formal_passed = summary.get("formal_passed_norms", 0)
    if formal_passed:
        lines += [
            f"Из них формально, без применимых объектов: {formal_passed}. Проверьте полноту данных: отсутствие объектов не подтверждает соответствие проекта.",
            "",
        ]
    for status, title in (
        ("violated", "Не выполнено"),
        ("partial", "Частично"),
        ("passed", "Выполнено"),
    ):
        lines += [
            f"## {title}",
            "",
            "| Пункт нормы | Тип проверки | Нарушений | Причина | Где смотреть |",
            "|---|---|---:|---|---|",
        ]
        for item in results:
            if item.get("status") != status or item.get("equivalent_to"):
                continue
            reference = source_reference(item.get("source") or {})
            path = item.get("archive_path")
            reason = item.get("status_reason") or (
                "Выполнено формально: применимых объектов нет"
                if item.get("formal_pass")
                else None
            )
            lines.append(
                f"| {_cell(reference)} | {_cell(item.get('check_type_ru', item.get('template')))} | "
                f"{(item.get('summary') or {}).get('violated_objects', 0)} | {_cell(reason)} | "
                f"{f'[{path}/report.md]({path}/report.md)' if path else 'summary.json'} |"
            )
        lines.append("")
    missing = not_checked_categories(summary)
    reasons = "; ".join(
        f"{REASON_LABELS[reason]}: {count}" for reason, count in sorted(missing.items())
    )
    lines += [
        "## Не запущено",
        "",
        f"Не запущено норм: {counts['not_checked']}. "
        f"Причины: {reasons or 'отсутствуют'}. Полный список находится в summary.json.",
        "",
    ]
    missing_entities = {}
    for entry in (summary.get("user_files") or {}).get("not_checked") or []:
        for code in entry.get("missing") or []:
            if not isinstance(code, str):
                continue
            if code.startswith("user_files:no_project_objects:"):
                names = code.removeprefix("user_files:no_project_objects:").split(",")
            elif code.startswith("user_files:no_objects:"):
                names = [code.removeprefix("user_files:no_objects:")]
            else:
                continue
            for name in names:
                if name.strip():
                    missing_entities.setdefault(name.strip(), None)
    if missing_entities:
        names = ", ".join(list(missing_entities)[:10])
        if len(missing_entities) > 10:
            names += f" и ещё {len(missing_entities) - 10}"
        lines += [f"В файлах нет объектов {names}.", ""]
    lines += user_files_report(summary.get("user_files"))
    lines += [
        "",
        "## Где смотреть подробно",
        "",
        "Отчёты по нормам: violated/, partial/, passed/. "
        "Геометрия всех норм: layers.geojson. Параметры и контрольные суммы: manifest.json.",
    ]
    return "\n".join(lines) + "\n"


def build_norm_report(record, layer):
    source = record.get("source") or {}
    plan = record.get("check_plan") or {}
    coverage = record.get("coverage") or {}
    lines = [
        f"# {source_reference(source)} — {STATUS_RU[record['status']]}",
        "",
        "## Исходный текст нормы",
        "",
        source.get("extraction_text") or "Не указан.",
        "",
        "## Что проверялось",
        "",
        f"Тип: {record.get('check_type_ru')} / {record.get('check_type_en')}.",
        f"Шаблон: {record.get('template')}@{record.get('template_version')}.",
        f"Параметры: `{_json(plan.get('params', {}))}`.",
        f"Предикат: {(plan.get('params') or {}).get('predicate', (plan.get('params') or {}).get('join_predicate', 'intersects'))}.",
        "",
        "## Данные",
        "",
        "| Слой | Источник | Объектов |",
        "|---|---|---:|",
    ]
    for item in record.get("input_layers", []):
        lines.append(
            f"| {_cell(item['name'])} | {_cell(item['origin'])} | {item['count']} |"
        )
    lines += [
        f"Fill-rate: {coverage.get('fill_rate', 0)}.",
        f"Разрешение атрибутов: `{_json(record.get('resolved_requirements', []))}`.",
        "",
        "## Результат",
        "",
        f"Проверено объектов: {coverage.get('checked_objects', 0)}. "
        f"Нарушений: {(record.get('summary') or {}).get('violated_objects', 0)}. "
        f"Не проверено: {coverage.get('unchecked_objects', 0)}.",
        "",
        record.get("status_reason")
        or (
            "Норма выполнена формально: применимых объектов нет. Проверьте полноту данных сценария."
            if "no_applicable_objects" in (record.get("warnings") or [])
            else None
        )
        or "Все применимые объекты соответствуют проверенному требованию.",
    ]
    if strictest_applicability(record):
        lines += ["", STRICTEST_NORM_NOTICE]
    if transport_accessibility(record):
        lines += ["", TRANSPORT_NOTICE]
    for details in (
        _parameters(record),
        _population_lines(record),
        _scope_lines(record),
        _strictest_lines(record),
        _transport_lines(record),
    ):
        lines += details
    lines += [
        "",
        "## Нарушения",
        "",
        "| Объект | Измерено | Требуется | Дефицит | Объяснение |",
        "|---|---|---|---|---|",
    ]
    violations = [
        f.get("properties") or {}
        for f in (layer or {}).get("features", [])
        if (f.get("properties") or {}).get("feature_kind") == "violation"
    ]
    for item in violations[:50]:
        lines.append(
            "| "
            + " | ".join(
                _cell(item.get(key))
                for key in (
                    "object_name",
                    "measured_value",
                    "required_value",
                    "deficit",
                    "explanation_ru",
                )
            )
            + " |"
        )
    # TODO: подробный LLM-комментарий к нарушению; сейчас объяснения детерминированы.
    lines += [
        "",
        "Полная геометрия и список нарушений: result.geojson.",
        "",
        "## Эквивалентные нормы",
        "",
        grouped_references(record.get("equivalent_sources") or [source]),
        *[
            f"- {source_reference(item)} — {item.get('restriction_id')}"
            for item in record.get("equivalent_sources") or []
        ],
        "",
        "## Файлы",
        "",
        "report.md, result.json, result.geojson, layers/<role>__<entity>.geojson.",
        "",
    ]
    return "\n".join(lines)


class ComplianceArchive:
    def __init__(self, request_id, scenario_id, *, root: Path | None = None):
        self.root = (
            (root or default_cache_dir()) / "compliance" / _component(request_id)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self.scenario_id = scenario_id if scenario_id is not None else "files"

    def write_execution(self, execution, layer, zone=None, shared_calls=()):
        result = execution.result
        launched = any(
            call["function"]["name"].startswith("Check")
            for call in execution.tool_calls
        )
        launched = (
            launched
            or bool(execution.effects_tool_calls)
            or result.verification_status == "complete"
        )
        if not launched:
            result.status = "not_checked"
            result.status_reason = (
                "; ".join(result.missing_requirements + result.warnings)
                or "Проверка не запущена"
            )
            return None
        path = f"{result.status}/{_component(result.restriction_id)}"
        result.archive_path = path
        folder = self.root / path
        folder.mkdir(parents=True, exist_ok=True)
        record = result.model_dump(
            mode="json", exclude={"violated_features", "passed_features"}
        )
        record.update(
            check_plan=(
                execution.plan.model_dump(mode="json") if execution.plan else None
            ),
            equivalent_sources=result.source.get("equivalent_sources", []),
            effects_tool_calls=execution.effects_tool_calls,
            timings_ms=execution.timings_ms,
            input_revision=next(
                (
                    c["function"]["arguments"].get("input_revision")
                    for c in execution.tool_calls
                    if c["function"]["name"].startswith("Check")
                ),
                None,
            ),
        )
        first = next(iter((layer or {}).get("features", [])), {}).get("properties", {})
        record.update(
            check_type_en=result.check_type_en
            or first.get("check_type_en", result.template),
            check_type_ru=result.check_type_ru
            or first.get("check_type_ru", result.template),
        )
        inputs = {**getattr(zone, "input_layers", {}), **execution.layers}
        if zone is not None and zone.zones is not None:
            inputs[zone.layer_name or source_reference(zone.source)] = zone.zones
        record["tool_calls"] = list(
            {
                _json(call): call
                for call in [
                    *[
                        call
                        for call in shared_calls
                        if call["function"]["name"] == "GetProjectTerritory"
                        or call["function"]["arguments"].get("layer_name") in inputs
                    ],
                    *execution.tool_calls,
                ]
            }.values()
        )
        roles = {
            item.entity: item.role for item in result.effective_requirements.layers
        }
        record["input_layers"] = []
        for name, collection in inputs.items():
            role = roles.get(
                name, "territory" if name == "project_territory" else "source"
            )
            if zone is not None and collection is zone.zones:
                role = "zone"
            filename = f"layers/{_component(role)}__{_component(name)}.geojson"
            self._write(folder / filename, _json(collection))
            origins = []
            for call in record["tool_calls"]:
                function = call["function"]
                args = function["arguments"]
                matches = name == args.get("layer_name") or (
                    name == "project_territory"
                    and function["name"] == "GetProjectTerritory"
                )
                if matches:
                    origins.append(function["name"])
            record["input_layers"].append(
                {
                    "name": name,
                    "path": filename,
                    "count": len(collection.get("features", [])),
                    "origin": ", ".join(dict.fromkeys(origins))
                    or (
                        "user_files"
                        if self.scenario_id == "files"
                        else "источник не указан"
                    ),
                }
            )
        geometry = layer or {"type": "FeatureCollection", "features": []}
        self._write(folder / "result.json", _json(record))
        self._write(folder / "result.geojson", _json(geometry))
        self._write(folder / "report.md", build_norm_report(record, geometry))
        return record

    def write_inputs(self, path, project, scenario, zones, calls):
        """Normalize shared input layers once; all copying/JSON stays off the loop."""
        scenario = scenario.snapshot() if scenario is not None else None
        shared = {"project_territory": project}
        if scenario:
            shared.update(
                {
                    f"Источники — {name}": layer
                    for name, layer in scenario["sources"].items()
                }
            )
            shared.update(scenario["sources"])
        extra = {}
        records = {}
        for rid, zone in zones.items():
            record = {
                field.name: getattr(zone, field.name)
                for field in fields(zone)
                if field.name != "input_layers"
            }
            refs = {}
            for name, layer in zone.input_layers.items():
                if name in shared and shared[name] == layer:
                    refs[name] = ["shared", name]
                else:
                    key = hashlib.sha256(_json(layer).encode("utf-8")).hexdigest()
                    extra[key] = layer
                    refs[name] = ["extra", key]
            record["input_layer_refs"] = refs
            records[rid] = record
        self._write(
            path,
            _json(
                {
                    "project": project,
                    "scenario": scenario,
                    "zones": records,
                    "calls": calls,
                    "extra_layers": extra,
                }
            ),
        )

    @staticmethod
    def read_inputs(path):
        data = json.loads(path.read_text("utf-8"))
        shared = {"project_territory": data.get("project")}
        scenario = data.get("scenario")
        if scenario:
            shared.update(
                {
                    f"Источники — {name}": layer
                    for name, layer in scenario["sources"].items()
                }
            )
            shared.update(scenario["sources"])
        zones = data.get("zones", {})
        for record in zones.values() if isinstance(zones, dict) else []:
            if "input_layer_refs" in record:
                record["input_layers"] = {
                    name: (shared if pool == "shared" else data["extra_layers"])[key]
                    for name, (pool, key) in record.pop("input_layer_refs").items()
                }
        return data

    def write_inventory(self, path, zones, layers):
        records = [
            {
                field.name: getattr(zone, field.name)
                for field in fields(zone)
                if field.name != "input_layers"
            }
            for zone in zones
        ]
        self._write(path, _json({"zones": records, "layers": layers}))

    def cache_result(self, result):
        record = result.model_dump(
            mode="json", exclude={"violated_features", "passed_features"}
        )
        self._write(
            self.root / ".executions" / f"{_component(result.restriction_id)}.json",
            _json(record),
        )
        os.utime(self.root, None)

    def read_cached_results(self):
        return {
            path.stem: json.loads(path.read_text("utf-8"))
            for path in (self.root / ".executions").glob("*.json")
        }

    def read_layer(self, archive_path):
        if not archive_path:
            return None
        path = self.root / archive_path / "result.geojson"
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Invalid cached result path")
        return json.loads(path.read_text("utf-8"))

    def finalize(self, summary, layers):
        checkpoint = self.root / "archive.json"
        if checkpoint.exists():
            info = json.loads(checkpoint.read_text("utf-8"))
            if (self.root / info["filename"]).exists():
                saved_summary = json.loads(
                    (self.root / "summary.json").read_text("utf-8")
                )
                summary.clear()
                summary.update(saved_summary)
                return self.root / "summary.md", self.root / info["filename"]
        summary.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        summary["scenario_id"] = self.scenario_id
        stamp = datetime.fromisoformat(summary["created_at"]).strftime("%Y%m%d-%H%M")
        filename = f"compliance_{self.scenario_id}_{stamp}.zip"
        summary["files"] = {"summary": "summary.md", "archive": filename}
        self._write(self.root / "summary.md", build_summary_report(summary))
        self._write(self.root / "summary.json", _json(summary))
        self._write(
            self.root / "layers.geojson",
            _json(
                {
                    "type": "FeatureCollection",
                    "meta": {
                        "group_by": "norm_id",
                        "group_properties": list(
                            dict.fromkeys(
                                key
                                for layer in layers.values()
                                for key in layer.get("meta", {}).get(
                                    "group_properties", []
                                )
                            )
                        ),
                    },
                    "features": [
                        feature
                        for layer in layers.values()
                        for feature in layer.get("features", [])
                    ],
                }
            ),
        )
        self._write(
            self.root / "README.md",
            "\n".join(
                [
                    "# Результаты проверки норм",
                    "",
                    "summary.md — сводный отчёт.",
                    "summary.json — все нормы, статусы и причины пропуска.",
                    "layers.geojson — все слои норм в EPSG:4326.",
                    "Группировка объектов — по norm_id.",
                    "feature_kind: zone, violation, object.",
                    "Зоны окрашиваются по status.",
                    "violated/ — нарушенные нормы.",
                    "partial/ — проверки с неполными данными.",
                    "passed/ — выполненные нормы.",
                    "В каждой папке: report.md, result.json, result.geojson и входные layers/.",
                    "Эквивалентные нормы объединены в папке выполненной проверки.",
                    "manifest.json содержит параметры, версии и SHA-256 файлов.",
                    "Контрольная сумма самого manifest.json не включена (самоссылка).",
                    "",
                ]
            ),
        )
        files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and ".executions" not in path.relative_to(self.root).parts
            and path.name
            not in {"manifest.json", "archive.json", "inputs.json", "inventory.json"}
            and path.suffix != ".zip"
        )
        manifest = {
            "request_id": summary["request_id"],
            "scenario_id": self.scenario_id,
            "created_at": summary["created_at"],
            "gmart_version": __VERSION__,
            "templates": sorted(
                {
                    f"{r.get('template')}@{r.get('template_version')}"
                    for r in summary["results"]
                    if r.get("archive_path")
                }
            ),
            "parameters": {
                key: summary.get(key)
                for key in ("scope", "territory", "user_files", "query")
            },
            "files": {
                str(path.relative_to(self.root).as_posix()): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in files
            },
        }
        self._write(self.root / "manifest.json", _json(manifest))
        archive = self.root / filename
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zipped:
            for path in [*files, self.root / "manifest.json"]:
                zipped.write(path, path.relative_to(self.root).as_posix())
        self._write(checkpoint, _json({"filename": filename}))
        return self.root / "summary.md", archive

    @staticmethod
    def _write(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(content, encoding="utf-8")
        os.replace(partial, path)
