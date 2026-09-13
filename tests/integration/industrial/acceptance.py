"""Outcome oracle. It never prescribes agent ordering or model decisions."""

from decimal import Decimal, InvalidOperation
from math import isfinite

from shapely.geometry import shape

from .control import entities
from .geometry import is_fifty_metre_buffer


def scoped(context, sid, kind):
    completed = {
        (c["request_id"], c["step"])
        for c in context.get("completed", [])
        if c.get("scenario_id") == sid and c.get("status") == "completed"
    }
    return [
        a
        for a in context.get("artifacts", [])
        if a.get("confirmed")
        and a["kind"] == kind
        and (a["request_id"], a["step"]) in completed
    ]


def provision_rows(artifacts):
    rows = []
    for a in artifacts:
        content = a["content"]
        if content.get("name") == "provision_summary":
            rows.extend(content.get("rows", []))
        elif content.get("name") == "provision_metrics":
            metrics = {r.get("metric"): r.get("value") for r in content.get("rows", [])}
            rows.append(
                {
                    "service": content.get("title", ""),
                    "capacity": metrics.get("Вместимость (чел)"),
                    "demand": metrics.get("Спрос (чел)"),
                    "deficit": metrics.get("Дефицит (чел)"),
                }
            )
    return rows


def verify_result(final, context, contract):
    checks = []

    def check(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    artifacts = context.get("artifacts", [])
    ids = [a["id"] for a in artifacts]
    check("completed", final.get("status") == "completed" and not final.get("missing"))
    check("answer", bool(final.get("answer", "").strip()))
    check("unique_artifacts", len(ids) == len(set(ids)))
    confirmed = {a["id"] for a in artifacts if a.get("confirmed")}
    check("evidence_references", set(final.get("evidence_ids", [])) <= confirmed)
    for a in artifacts:
        if not a.get("confirmed"):
            continue
        content = a.get("content", {})
        if a["kind"] == "table":
            rows = content.get("rows", [])
            check(
                "table_complete:" + a["id"],
                content.get("complete", True)
                and content.get("total_rows", len(rows)) == len(rows),
            )
        if a["kind"] == "feature_collection":
            layer = content.get("feature_collection", {})
            valid = layer.get("type") == "FeatureCollection" and isinstance(
                layer.get("features"), list
            )
            try:
                for f in layer.get("features", []):
                    geom = shape(f["geometry"])
                    x1, y1, x2, y2 = geom.bounds
                    valid = (
                        valid
                        and geom.is_valid
                        and not geom.is_empty
                        and all(isfinite(v) for v in geom.bounds)
                        and -180 <= x1 <= x2 <= 180
                        and -90 <= y1 <= y2 <= 90
                    )
            except (ValueError, TypeError, KeyError):
                valid = False
            check("wgs84_geometry:" + a["id"], valid)
    for expected in contract.get("provision", []):
        sid, service = expected["scenario_id"], expected["service"]
        rows = provision_rows(scoped(context, sid, "table"))
        matches = [
            r
            for r in rows
            if service.casefold() in str(r.get("service", "")).casefold()
        ]
        check(
            f"provision:{sid}:{service}",
            bool(matches)
            and all(
                [r.get(k) for k in ("capacity", "demand", "deficit")]
                == expected["values"]
                for r in matches
            ),
            "capacity, demand, deficit must agree in the requested scenario scope",
        )
        if expected.get("layers"):
            layers = [
                a
                for a in scoped(context, sid, "feature_collection")
                if a["content"]
                .get("name", "")
                .casefold()
                .startswith("provision." + service.casefold() + ".")
            ]
            # Require actual calculated building/service/link properties, not arbitrary maps.
            props = [
                f.get("properties", {})
                for a in layers
                for f in a["content"].get("feature_collection", {}).get("features", [])
            ]
            check(
                f"calculation_layers:{sid}:{service}",
                any("demand_left" in p for p in props)
                and any("capacity_left" in p for p in props)
                and any(
                    a["content"].get("feature_collection", {}).get("features")
                    and any(
                        f["geometry"]["type"] in ("LineString", "MultiLineString")
                        for f in a["content"]["feature_collection"]["features"]
                    )
                    for a in layers
                ),
            )
    for expected in contract.get("source_layers", []):
        sid, domain, type_id = (
            expected["scenario_id"],
            expected["domain"],
            expected["type_id"],
        )
        wanted = entities(sid, domain, type_id)["features"]
        layers = scoped(context, sid, "feature_collection")
        found = False
        for a in layers:
            actual = a["content"].get("feature_collection", {}).get("features", [])
            if len(actual) != len(wanted):
                continue
            try:
                found |= all(
                    any(
                        shape(f["geometry"]).equals_exact(shape(w["geometry"]), 1e-7)
                        and f.get("properties", {}).get(domain + "_id")
                        == w["properties"][domain + "_id"]
                        and f.get("properties", {}).get("source_version")
                        == w["properties"]["source_version"]
                        for f in actual
                    )
                    for w in wanted
                )
            except (KeyError, ValueError):
                continue
        check(f"source_layer:{sid}:{domain}:{type_id}", found)
    for expected in contract.get("population", []):
        sid = expected["scenario_id"]
        rows = [
            r
            for a in artifacts
            if a.get("confirmed") and a["kind"] == "table"
            for r in a["content"].get("rows", [])
        ]
        matches = [
            r
            for r in rows
            if r.get("scenario_id") == sid and r.get("indicator_id") == 1
        ]
        check(
            f"population:{sid}",
            bool(matches)
            and all(
                r.get("value") == expected["value"] and r.get("unit") == "человек"
                for r in matches
            ),
        )
    for sid in contract.get("buffers", []):
        check(
            f"buffer_50m:{sid}",
            any(
                is_fifty_metre_buffer(
                    a["content"].get("feature_collection", {}),
                    entities(sid, "physical_object", 7),
                )
                for a in scoped(context, sid, "feature_collection")
            ),
        )
    for sid in contract.get("zones", []):
        from .sources import read

        expected = read(f"/api/v1/scenarios/{sid}/functional_zones", {})
        found = False
        for a in scoped(context, sid, "feature_collection"):
            features = a["content"].get("feature_collection", {}).get("features", [])
            if len(features) != 3:
                continue
            found |= all(
                any(
                    f.get("properties", {}).get("functional_zone_id")
                    == w["properties"]["functional_zone_id"]
                    and f["properties"].get("source") == w["properties"]["source"]
                    and f["properties"].get("year") == 2026
                    and shape(f["geometry"]).equals_exact(shape(w["geometry"]), 1e-7)
                    for f in features
                )
                for w in expected["features"]
            )
        check(f"functional_zones:{sid}", found)
    if contract.get("comparisons"):
        expected_deficits = {
            (p["scenario_id"], p["service"]): p["values"][2]
            for p in contract["provision"]
        }
        artifact_map = {a["id"]: a for a in artifacts if a.get("confirmed")}
        scopes = {
            (c["request_id"], c["step"]): c["scenario_id"]
            for c in context.get("completed", [])
            if c.get("status") == "completed"
        }

        def deficit_reference(ref):
            a = artifact_map[ref["artifact_id"]]
            row = a["content"]["rows"][ref["row"]]
            sid = scopes[(a["request_id"], a["step"])]
            if ref["column"] == "deficit":
                service = row["service"]
            elif ref["column"] == "value" and row.get("metric") == "Дефицит (чел)":
                service = next(
                    s
                    for s in ("Школа", "Детский сад")
                    if s.casefold() in a["content"]["title"].casefold()
                )
            else:
                raise ValueError("Not a deficit reference")
            value = expected_deficits[(sid, service)]
            if row[ref["column"]] != value:
                raise ValueError("Incorrect referenced metric")
            return sid, service, Decimal(value)

        edges = {s: set() for s in ("Школа", "Детский сад")}
        for a in artifacts:
            if (
                a.get("confirmed")
                and a["kind"] == "table"
                and a["content"].get("name") == "analysis_comparison"
            ):
                for r in a["content"]["rows"]:
                    try:
                        before, service, bvalue = deficit_reference(r["source_before"])
                        after, other, avalue = deficit_reference(r["source_after"])
                        if (
                            service == other
                            and before != after
                            and Decimal(str(r["before"])) == bvalue
                            and Decimal(str(r["after"])) == avalue
                            and Decimal(str(r["delta"])) == avalue - bvalue
                        ):
                            edges[service].add((before, after))
                    except (
                        KeyError,
                        IndexError,
                        StopIteration,
                        ValueError,
                        TypeError,
                        InvalidOperation,
                    ):
                        continue
        for service in edges:
            required = {
                p[key]
                for p in contract["comparisons"]
                if p["service"] == service
                for key in ("before", "after")
            }
            reached = {min(required)} if required else set()
            for _ in required:
                for before, after in edges[service]:
                    if {before, after} & reached:
                        reached.update((before, after))
            check("deficit_comparison:" + service, required <= reached)
    restriction_ids = {
        r["id"]
        for a in artifacts
        if a["kind"] == "source_evidence"
        and a.get("confirmed")
        and a["content"].get("system") == "norms"
        for r in a["content"].get("sources", [])
        if r.get("id")
    }
    for expected in contract.get("compliance", []):
        sid = expected["scenario_id"]
        results = [a["content"] for a in scoped(context, sid, "compliance_result")]
        check(
            f"compliance:{sid}",
            any(
                c.get("coverage", {}).get("checked_objects") == 1
                and c.get("coverage", {}).get("unchecked_objects") == 0
                and c.get("summary", {}).get("violated_objects")
                == expected["violations"]
                and c.get("compliance_status") != "unknown"
                and c.get("restriction_id") in restriction_ids
                for c in results
            ),
        )
    if contract.get("sources"):
        from tests.integration.local_stack.source_contract import verify_source_records

        sources = {"documents": [], "norms": []}
        for a in artifacts:
            if a["kind"] == "source_evidence" and a.get("confirmed"):
                content = a["content"]
                sources.setdefault(content["system"], []).extend(content["sources"])
        try:
            verify_source_records(sources)
            check("normative_source_version", True)
        except (AssertionError, KeyError, IndexError, TypeError):
            check("normative_source_version", False)
    return {
        "passed": bool(checks) and all(c["passed"] for c in checks),
        "checks": checks,
    }
