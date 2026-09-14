"""Durable evidence with bounded, loss-aware views for the analytical planner."""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation

from fastapi.encoders import jsonable_encoder

from src.agents.api_clients.chat_storage_client.request_models import TablePayload


class AnalysisContext:
    def __init__(self, data=None):
        data = data or {}
        self.artifacts = data.get("artifacts", [])
        self.completed = data.get("completed", [])
        self.query = data.get("query", "")
        self.goal = data.get("goal")
        self.inspected = []
        self.inspection_signatures = set()

    def add_artifact(self, event, step, request_id):
        kind = event.get("type")
        if kind not in {
            "table",
            "feature_collection",
            "compliance_summary",
            "compliance_result",
            "check_plan",
            "requirement_resolution",
            "validation",
            "artifact_ref",
            "analysis_text",
            "source_evidence",
        }:
            return None
        content = event.get("content")
        if not isinstance(content, dict):
            return None
        content = jsonable_encoder(content)
        if kind == "table":
            TablePayload.model_validate(content)
        encoded = json.dumps(content, ensure_ascii=False, default=str)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        previous = next(
            (
                a
                for a in self.artifacts
                if a["fingerprint"] == fingerprint
                and a["kind"] == kind
                and a["request_id"] == request_id
            ),
            None,
        )
        if previous:
            return previous["id"]
        # Retain full source artifacts, independently from prompt compaction.
        artifact = {
            "id": f"{request_id}:a{len(self.artifacts)+1}",
            "kind": kind,
            "step": step,
            "request_id": request_id,
            "confirmed": False,
            "fingerprint": fingerprint,
            "content": content,
        }
        self.artifacts.append(artifact)
        return artifact["id"]

    def finish(self, step, task, scenario_id, status, summary, request_id):
        for artifact in self.artifacts:
            if artifact["step"] == step and artifact["request_id"] == request_id:
                artifact["confirmed"] = status == "completed"
        self.completed.append(
            {
                "step": step,
                "task": task,
                "scenario_id": scenario_id,
                "status": status,
                "summary": summary,
                "request_id": request_id,
            }
        )

    def get(self, artifact_id):
        result = next((a for a in self.artifacts if a["id"] == artifact_id), None)
        if result is None or not result["confirmed"]:
            raise ValueError("Unknown or unconfirmed evidence reference")
        return result

    def index(self):
        result = []
        scopes = {
            (c["request_id"], c["step"]): c["scenario_id"] for c in self.completed
        }
        for a in self.artifacts:
            c = a["content"]
            entry = {k: a[k] for k in ("id", "kind", "step", "request_id", "confirmed")}
            entry["title"] = c.get("title") or c.get("name") or a["kind"]
            entry["scenario_id"] = scopes.get((a["request_id"], a["step"]))
            if a["kind"] == "table":
                entry.update(
                    columns=c.get("columns", []),
                    rows=len(c.get("rows", [])),
                    total_rows=c.get("total_rows"),
                    complete=c.get("complete", True),
                )
            elif a["kind"] == "feature_collection":
                entry["features"] = len(
                    (c.get("feature_collection") or {}).get("features", [])
                )
            result.append(entry)
        return result

    def view(self, max_chars=9000):
        # Never clip JSON or turn a partial table into a complete one. Older full
        # data remain inspectable by ID even when their preview no longer fits.
        def size(value):
            return len(
                json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            )

        view = {
            "completed": [],
            "artifacts": [],
            "selected_evidence": [],
            "omitted_summaries": 0,
            "artifact_count": len(self.artifacts),
            "catalog_paging_id": "_catalog",
            "omitted_artifacts": 0,
            "prior_request": self.query[:1000],
            "prior_request_truncated": len(self.query) > 1000,
        }
        # Reserve space for actual results before catalogue metadata. A long
        # sequence of layers must not hide earlier calculation rows and sources.
        candidates = list(self.inspected)
        seen = set()
        for a in reversed(self.artifacts):
            if not a["confirmed"] or a["kind"] not in {
                "table",
                "source_evidence",
                "compliance_summary",
            }:
                continue
            preview = self.slice(a["id"], 0, 3)
            key = (a["fingerprint"], preview.get("scenario_id"))
            if key not in seen:
                candidates.append(preview)
                seen.add(key)
        candidates.sort(
            key=lambda p: (
                0
                if p in self.inspected
                else 1 if p.get("name") == "provision_summary" else 2
            )
        )
        for preview in candidates:
            if size(view) + size(preview) < max_chars // 2:
                view["selected_evidence"].append(preview)
        for entry in reversed(self.index()):
            entry = {k: v for k, v in entry.items() if k != "columns"}
            entry["title"] = entry["title"][:100]
            if size(view) + size(entry) <= max_chars * 3 // 4:
                view["artifacts"].insert(0, entry)
            else:
                view["omitted_artifacts"] += 1
        for item in reversed(self.completed):
            candidate = {
                **item,
                "summary": item["summary"][:800],
                "summary_truncated": len(item["summary"]) > 800,
            }
            if size(view) + size(candidate) > max_chars - 32:
                view["omitted_summaries"] += 1
            else:
                view["completed"].insert(0, candidate)
        for preview in candidates:
            if preview in view["selected_evidence"]:
                continue
            if size(view) + size(preview) <= max_chars - 32:
                view["selected_evidence"].append(preview)
        return view

    def slice(self, artifact_id, offset, limit):
        if artifact_id == "_catalog":
            return {
                "artifact_id": "_catalog",
                "offset": offset,
                "entries": self.index()[offset : offset + limit],
                "total": len(self.artifacts),
            }
        a = self.get(artifact_id)
        content = a["content"]
        scope = next(
            (
                c["scenario_id"]
                for c in self.completed
                if c["request_id"] == a["request_id"] and c["step"] == a["step"]
            ),
            None,
        )
        if a["kind"] == "table":
            rows = content.get("rows", [])
            return {
                "artifact_id": artifact_id,
                "name": content.get("name"),
                "title": content.get("title"),
                "scenario_id": scope,
                "offset": offset,
                "rows": rows[offset : offset + limit],
                "stored_rows": len(rows),
                "source_complete": content.get("complete", True),
                "columns": content.get("columns", []),
            }
        if a["kind"] == "feature_collection":
            rows = (content.get("feature_collection") or {}).get("features", [])
            return {
                "artifact_id": artifact_id,
                "offset": offset,
                "properties": [
                    r.get("properties", {}) for r in rows[offset : offset + limit]
                ],
                "stored_features": len(rows),
                "geometry_available_in_artifact": True,
            }
        if a["kind"] == "analysis_text":
            text = content["text"]
            return {
                "artifact_id": artifact_id,
                "offset": offset,
                "text": text[offset : offset + limit * 100],
                "total_characters": len(text),
                "complete": offset + limit * 100 >= len(text),
            }
        return {"artifact_id": artifact_id, "content": content}

    def inspect(self, requests):
        previews = [self.slice(r.artifact_id, r.offset, r.limit) for r in requests]
        signatures = [
            json.dumps(p, ensure_ascii=False, sort_keys=True) for p in previews
        ]
        if any(
            p.get("artifact_id") == "_catalog" and not p["entries"] for p in previews
        ):
            raise ValueError(
                "Cannot inspect an empty catalog; execute a pending requirement"
            )
        if all(s in self.inspection_signatures for s in signatures):
            raise ValueError(
                "Evidence slice already inspected and unchanged; execute a pending requirement"
            )
        self.inspection_signatures.update(signatures)
        self.inspected = previews

    def population(self, adjustment):
        ref = adjustment.base
        artifact = self.get(ref.artifact_id)
        if artifact["kind"] != "table":
            raise ValueError("Population must reference a confirmed table")
        try:
            raw = artifact["content"]["rows"][ref.row][ref.column]
            if isinstance(raw, bool):
                raise ValueError("Population must be numeric")
            value = Decimal(str(raw)) * adjustment.multiplier
            if (
                not value.is_finite()
                or value <= 0
                or value != value.to_integral_value()
            ):
                raise ValueError(
                    "Population adjustment must yield a positive whole number"
                )
            return int(value)
        except (KeyError, IndexError, InvalidOperation) as exc:
            raise ValueError("Population evidence is missing") from exc

    def compare(self, specs):
        rows = []
        for spec in specs:
            values = []
            source_rows = []
            for ref in (spec.before, spec.after):
                a = self.get(ref.artifact_id)
                if a["kind"] != "table":
                    raise ValueError("A metric must reference a table")
                try:
                    raw = a["content"]["rows"][ref.row][ref.column]
                    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                        raise ValueError("Metric is not numeric")
                    value = Decimal(str(raw))
                    if not value.is_finite():
                        raise ValueError("Metric is not finite")
                    values.append(value)
                    source_rows.append(a["content"]["rows"][ref.row])
                except (KeyError, IndexError, InvalidOperation) as exc:
                    raise ValueError(
                        "Metric evidence is missing or nonnumeric"
                    ) from exc
            for dimension in ("unit", "year", "territory_id", "methodology"):
                before_dimension, after_dimension = (
                    r.get(dimension) for r in source_rows
                )
                if (
                    before_dimension is not None
                    and after_dimension is not None
                    and before_dimension != after_dimension
                ):
                    raise ValueError(f"Incompatible comparison dimension: {dimension}")
            if any(
                r.get("unit") is not None and r["unit"] != spec.unit
                for r in source_rows
            ):
                raise ValueError("Comparison unit differs from source evidence")
            before, after = values
            rows.append(
                {
                    "metric": spec.name,
                    "unit": spec.unit,
                    "before": str(before),
                    "after": str(after),
                    "delta": str(after - before),
                    "percent": (
                        str((after - before) / abs(before) * 100) if before else None
                    ),
                    "source_before": spec.before.model_dump(),
                    "source_after": spec.after.model_dump(),
                }
            )
        return {
            "type": "table",
            "content": {
                "name": "analysis_comparison",
                "title": "Сравнение подтверждённых показателей",
                "columns": [
                    {"key": k, "label": label}
                    for k, label in [
                        ("metric", "Показатель"),
                        ("unit", "Единицы"),
                        ("before", "Было"),
                        ("after", "Стало"),
                        ("delta", "Изменение"),
                        ("percent", "Изменение, %"),
                        ("source_before", "Источник исходного значения"),
                        ("source_after", "Источник нового значения"),
                    ]
                ],
                "rows": rows,
                "total_rows": len(rows),
                "complete": True,
            },
        }

    def dump(self):
        return {
            "artifacts": self.artifacts,
            "completed": self.completed,
            "query": self.query,
            "goal": self.goal,
        }

    def provision_comparisons(self, query, existing=()):
        """Derive requested comparisons from actual cells, with scenario provenance."""
        from src.agents.services.provision.provision_context import (
            ProvisionContextBuilder,
        )
        from src.agents.services.service_entities.orchestrator_plan import (
            MetricComparison,
        )

        if not re.search(r"сравн|сопостав|изменени\w*\s+дефицит", query, re.I):
            return list(existing)
        scopes = {
            (c["request_id"], c["step"]): c["scenario_id"]
            for c in self.completed
            if c["status"] == "completed"
        }
        sources = {}
        metric_labels = {
            column["label"]: column["key"]
            for column in ProvisionContextBuilder.SUMMARY_TABLE_COLUMNS
            if column["key"] != "service"
        }
        for a in self.artifacts:
            sid = scopes.get((a["request_id"], a["step"]))
            if (
                not a["confirmed"]
                or a["kind"] != "table"
                or a["content"].get("name")
                not in {"provision_summary", "provision_metrics"}
                or not sid
            ):
                continue
            if not re.search(rf"(?<!\d){sid}(?!\d)", query):
                continue
            title_service = re.fullmatch(
                r"Обеспеченность сервисом «(.+)»", a["content"].get("title", "")
            )
            for index, row in enumerate(a["content"].get("rows", [])):
                if row.get("service"):
                    for metric in metric_labels.values():
                        if metric in row and row[metric] is not None:
                            sources.setdefault((sid, row["service"]), {})[metric] = {
                                "artifact_id": a["id"],
                                "row": index,
                                "column": metric,
                            }
                elif (
                    title_service
                    and row.get("metric") in metric_labels
                    and row.get("value") is not None
                ):
                    sources.setdefault((sid, title_service.group(1)), {})[
                        metric_labels[row["metric"]]
                    ] = {"artifact_id": a["id"], "row": index, "column": "value"}
        ordered = sorted(
            {sid for sid, _ in sources}, key=lambda sid: query.find(str(sid))
        )
        if len(ordered) < 2:
            return list(existing)
        metrics = (
            ["deficit"]
            if re.search(r"дефицит", query, re.I)
            else ["capacity", "demand", "deficit", "surplus", "balance"]
        )
        specs = list(existing)
        seen = {
            json.dumps([s.before.model_dump(), s.after.model_dump()], sort_keys=True)
            for s in specs
        }
        for service in sorted({service for _, service in sources}):
            if (ordered[0], service) not in sources:
                continue
            for sid in ordered[1:]:
                if (sid, service) not in sources:
                    continue
                before, after = sources[(ordered[0], service)], sources[(sid, service)]
                for metric in metrics:
                    if metric not in before or metric not in after:
                        continue
                    identity = json.dumps(
                        [before[metric], after[metric]], sort_keys=True
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    specs.append(
                        MetricComparison.model_validate(
                            {
                                "name": f"{service}: {metric}, {ordered[0]} → {sid}",
                                "unit": "чел",
                                "before": before[metric],
                                "after": after[metric],
                            }
                        )
                    )
        return specs
