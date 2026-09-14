"""Read-only control Urban API. Unknown routes fail; no upstream passthrough."""

from copy import deepcopy

from fastapi import FastAPI, HTTPException, Request

from .control import (
    PHYSICAL_TYPES,
    SCENARIOS,
    SERVICE_TYPES,
    VERSION,
    collection,
    entities,
    indicators,
    nested_geometries,
    normatives,
    project,
    rectangle,
    scenario,
)


def read(path, params):
    parts = path.strip("/").split("/")
    if parts[:2] != ["api", "v1"]:
        raise HTTPException(404, "Unknown control source")
    parts = parts[2:]
    if parts == ["service_types"] or parts == ["physical_object_types"]:
        rows = SERVICE_TYPES if parts[0] == "service_types" else PHYSICAL_TYPES
        name = params.get("name")
        return [r for r in rows if not name or name.casefold() in r["name"].casefold()]
    if parts[:2] == ["projects", "910"]:
        if len(parts) == 2:
            return project()
        if parts[2:] == ["territory"]:
            return {"geometry": rectangle(30, 60, 0.02, 0.02)}
        if parts[2:] == ["scenarios"]:
            return [scenario(sid) for sid in SCENARIOS]
    if len(parts) == 3 and parts[0] == "territory" and parts[1] in {"910", "911"}:
        if parts[2] == "normatives":
            return normatives()
        if parts[2] == "indicator_values":
            return [{"value": 1000, "indicator": {"indicator_id": 1}}]
    if len(parts) >= 2 and parts[0] == "scenarios" and parts[1].isdigit():
        sid = int(parts[1])
        if sid not in SCENARIOS:
            raise HTTPException(404, "Unknown prepared scenario")
        tail = parts[2:]
        if not tail:
            return scenario(sid)
        context = tail[0] == "context"
        if context:
            tail = tail[1:]
        if tail == ["service_types"]:
            return SERVICE_TYPES
        if tail == ["physical_object_types"]:
            return PHYSICAL_TYPES
        if tail == ["indicators_values"]:
            return indicators(sid)
        if tail == ["geometries_with_all_objects"]:
            return nested_geometries(sid, params, context)
        for noun, domain in [
            ("services", "service"),
            ("physical_objects", "physical_object"),
        ]:
            if tail in ([noun], [noun + "_with_geometry"]):
                data = entities(sid, domain, params.get(domain + "_type_id"), context)
                return (
                    data
                    if tail[0].endswith("geometry")
                    else [f["properties"] for f in data["features"]]
                )
        if tail == ["functional_zone_sources"]:
            return [{"source": VERSION, "year": 2026}]
        if tail == ["functional_zones"]:
            if (
                params.get("source", VERSION) != VERSION
                or int(params.get("year", 2026)) != 2026
            ):
                raise HTTPException(404, "Unknown functional-zone version")
            from .control import feature

            share = {91007: 0.7, 91008: 0.5, 91009: 0.3}.get(sid, 0.5)
            return collection(
                [
                    feature(
                        rectangle(30 + 0.02 * offset, 60, 0.02 * part, 0.02),
                        {
                            "functional_zone_id": sid * 10 + i,
                            "name": name,
                            "source": VERSION,
                            "year": 2026,
                            "scenario_id": sid,
                            "share": part,
                        },
                    )
                    for i, (offset, part, name) in enumerate(
                        [
                            (0, share, "Жилая зона"),
                            (share, 0.9 - share, "Деловая зона"),
                            (0.9, 0.1, "Рекреационная зона"),
                        ]
                    )
                ],
                sid,
            )
    raise HTTPException(404, "Unsupported control source route")


def lookup(app, path, params):
    entry = {"path": path, "params": params, "version": VERSION}
    app.state.audit.append(entry)
    for fault in app.state.faults:
        if fault["path"] != path or fault.get("times") == 0:
            continue
        if "times" in fault:
            fault["times"] -= 1
        entry["fault"] = fault["kind"]
        if fault["kind"] == "unavailable":
            raise HTTPException(503, "Injected source outage")
        if fault["kind"] == "unauthorized":
            raise HTTPException(401, "Injected expired authorization")
        result = deepcopy(read(path, params))
        if fault["kind"] == "missing_normative":
            return [
                r for r in result if r["service_type"]["id"] != fault["service_type_id"]
            ]
        if fault["kind"] == "truncated":
            result["features"] = result["features"][:1]
            result["meta"].update(complete=False, truncated=True)
            result["complete"] = False
            return result
        raise ValueError("Unsupported injected fault")
    return read(path, params)


def create_app(faults=None):
    app = FastAPI(title="Industrial control inputs")
    app.state.audit = []
    app.state.faults = deepcopy(faults or [])

    @app.get("/health")
    def health():
        return {"version": VERSION, "kind": "source-data-only"}

    @app.get("/audit")
    def audit():
        return app.state.audit

    @app.get("/api/{path:path}")
    def get(path: str, request: Request):
        params = dict(request.query_params)
        return lookup(app, "/api/" + path, params)

    return app


app = create_app()
