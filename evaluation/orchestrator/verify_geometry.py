"""Recompute buffer geometries and intersection IDs from captured tool inputs.

Uses GeoPandas/Shapely directly; never calls the production MCP implementation.
Checks arithmetic/geometry, not whether the LLM chose the user's intended rule.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
from run_planner import save
from shapely.geometry import shape
from shapely.ops import unary_union


def identity(feature, index):
    props = feature.get("properties", {})
    for namespace in ("service", "physical_object"):
        if props.get(namespace + "_id") is not None:
            key = f"{namespace}/{props[namespace + '_id']}"
            break
    else:
        key = "feature/" + str(feature.get("id") or index)
    if props.get("object_geometry_id") is not None:
        key += f"/geometry/{props['object_geometry_id']}"
    return key


def verify(trace):
    checks = []
    for call in trace.get("tools", {}).get("idu", []):
        if "result" not in call or not isinstance(call["result"], dict):
            continue
        name, args = call["args"][:2]
        if name == "CreateBuffers":
            for layer, rule in args["buffer_info"].items():
                source = next(
                    fc
                    for key, fc in args["objects"].items()
                    if key.casefold() == layer.casefold()
                )
                actual = next(
                    fc
                    for key, fc in call["result"].items()
                    if key.casefold() == layer.casefold()
                )
                count_ok = len(source["features"]) == len(actual["features"])
                errors = []
                if source["features"] and count_ok:
                    src = gpd.GeoDataFrame.from_features(source, crs=4326)
                    crs = src.estimate_utm_crs()
                    # GeoPandas uses 16 segments per quadrant by default.
                    expected = src.to_crs(crs).geometry.buffer(
                        rule["buffer_size"], cap_style=rule.get("buffer_type", "round")
                    )
                    got = (
                        gpd.GeoDataFrame.from_features(actual, crs=4326)
                        .to_crs(crs)
                        .geometry
                    )
                    errors = [
                        float(a.symmetric_difference(b).area / max(a.area, 1))
                        for a, b in zip(expected, got)
                    ]
                checks.append(
                    {
                        "tool": name,
                        "layer": layer,
                        "count": len(actual["features"]),
                        "passed": count_ok and max(errors, default=0) < 1e-5,
                        "max_relative_symmetric_difference": max(errors, default=0),
                    }
                )
        elif name == "CreateRestrictions":
            layers = {k.casefold(): v for k, v in args["layers"].items()}
            expected = set()
            for generator, rule in args["restrictions"].items():
                fc = layers[generator.casefold()]
                area = unary_union([shape(f["geometry"]) for f in fc["features"]])
                for target in rule["to"]:
                    for index, feature in enumerate(
                        layers[target.casefold()]["features"]
                    ):
                        if shape(feature["geometry"]).intersects(area):
                            expected.add((target.casefold(), identity(feature, index)))
            got = {
                (
                    f["properties"]["source_layer"].casefold(),
                    f["properties"]["object_ref"]["id"],
                )
                for f in call["result"]["objects"]["features"]
            }
            checks.append(
                {
                    "tool": name,
                    "expected_count": len(expected),
                    "actual_count": len(got),
                    "passed": expected == got,
                    "missing_ids": sorted(expected - got),
                    "extra_ids": sorted(got - expected),
                }
            )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted(args.runs.glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        try:
            checks = verify(trace)
            if checks:
                results.append({"id": trace["id"], "checks": checks})
        except Exception as exc:
            results.append(
                {"id": trace["id"], "error": type(exc).__name__ + ": " + str(exc)}
            )
    checks = [c for r in results for c in r.get("checks", [])]
    save(
        args.out,
        {
            "cases": results,
            "checks": len(checks),
            "passed": sum(c["passed"] for c in checks),
            "errors": sum("error" in r for r in results),
            "scope": "geometry computation only; not request interpretation",
        },
    )
    print(
        f"Geometry: {sum(c['passed'] for c in checks)}/{len(checks)}; errors: {sum('error' in r for r in results)}",
        flush=True,
    )
