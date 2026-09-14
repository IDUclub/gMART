"""Verify real calculations against independently worked control totals; no LLM."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from dotenv import dotenv_values
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from .control import VERSION, entities
from .geometry import is_fifty_metre_buffer
from .transport import headers_for, provision, save

# Whole model scope includes a separate context with 1000 residents and 100/50
# places. Context demand is 100/50, so it changes totals, not project deficit.
# School demand is 100 per 1000, kindergarten 50 per 1000. These are worked
# literal expectations, never read from calculation responses or production code.
EXPECTED = {
    91001: ((900, 1300, 400), (450, 650, 200)),
    91002: ((400, 550, 150), (200, 275, 75)),
    91003: ((1900, 2600, 700), (950, 1300, 350)),
    91004: ((2600, 2600, 0), (1300, 1300, 0)),
    91005: ((900, 1300, 400), (450, 650, 200)),
    91006: ((1400, 1300, 0), (700, 650, 0)),
    91007: ((700, 900, 200), (350, 450, 100)),
    91008: ((900, 900, 0), (450, 450, 0)),
    91009: ((600, 900, 300), (300, 450, 150)),
}


def verify_calculation(sid, result):
    from shapely.geometry import shape

    for service, expected in zip(("22", "21"), EXPECTED[sid]):
        record = result["services"][service]
        summary = record.get("summary")
        if record.get("error") or not summary:
            raise ValueError(f"Calculation failed: scenario={sid}, type={service}")
        actual = tuple(
            summary[k] for k in ("total_capacity", "total_demand", "deficit")
        )
        if actual != expected:
            raise ValueError(
                f"Wrong totals: scenario={sid}, type={service}: {actual} != {expected}"
            )
        for kind in ("buildings", "services", "links"):
            layer = record["layers"][kind]
            if layer.get("type") != "FeatureCollection" or not layer.get("features"):
                raise ValueError(f"Missing {kind} calculation layer")
            for feature in layer["features"]:
                geom = shape(feature["geometry"])
                if not geom.is_valid or geom.is_empty:
                    raise ValueError("Invalid calculation geometry")


async def verify_source_transport():
    """Use the application's actual client, including structured MCP decoding."""
    from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
    from src.agents.services.scenario_data.scenario_data_selection import (
        selection_candidates,
        verified_entity_records,
    )

    client = UrbanMcpClient("http://localhost:18090", None)
    await client.load_tools()

    async def read(group, name, arguments):
        result = await client.execute_tool(group, name, arguments)
        if isinstance(result, dict) and set(result) == {"result"}:
            result = result["result"]
        if not isinstance(result, (list, dict)):
            raise ValueError(f"Missing structured source response: {name}")
        return result

    for domain, noun in (
        ("service_type", "Service"),
        ("physical_object_type", "PhysicalObject"),
    ):
        catalogue = await read(
            "projects", f"GetScenario{noun}Types", {"scenario_id": 91001}
        )
        candidates = selection_candidates({domain: catalogue})
        if not candidates:
            raise ValueError("Empty source catalogue")
        for candidate in candidates.values():
            arguments = {"scenario_id": 91001, f"{domain}_id": candidate["type_id"]}
            records = verified_entity_records(
                await read("projects", f"GetScenario{noun}s", arguments), candidate
            )
            geometry = verified_entity_records(
                await read("projects", f"GetScenario{noun}sWithGeometry", arguments),
                candidate,
            )
            identity = domain.removesuffix("_type") + "_id"
            if not records or {r[identity] for r in records} != {
                r[identity] for r in geometry
            }:
                raise ValueError("Source table and geometry identities differ")
    for group, name, arguments in (
        ("projects", "GetScenarioById", {"scenario_id": 91001}),
        ("projects", "GetProjectById", {"project_id": 910}),
        ("projects", "GetProjectScenarios", {"project_id": 910}),
        ("dictionaries", "GetServiceTypes", {}),
        ("dictionaries", "GetPhysicalObjectTypes", {}),
        ("indicators", "GetScenarioIndicatorsValues", {"scenario_id": 91001}),
        ("projects", "GetScenarioFunctionalZoneSources", {"scenario_id": 91001}),
        (
            "projects",
            "GetScenarioFunctionalZones",
            {"scenario_id": 91001, "source": VERSION, "year": 2026},
        ),
        ("territories", "GetTerritoryNormatives", {"territory_id": 911}),
    ):
        await read(group, name, arguments)


async def run(config, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"cases": [], "passed": False}
    source = {"name": "application_source_transport", "passed": False}
    try:
        async with asyncio.timeout(120):
            await verify_source_transport()
        source["passed"] = True
    except Exception as exc:
        source["error"] = type(exc).__name__ + ": " + str(exc)[:250]
    report["cases"].append(source)
    save(output / "report.json", report)
    print(f"Sources: {'PASS' if source['passed'] else source['error']}", flush=True)
    if not source["passed"]:
        return False
    async with httpx.AsyncClient(timeout=120, trust_env=False) as http:
        documents = {"name": "real_document_search", "passed": False}
        try:
            from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient

            headers = await headers_for(http, config)
            async with Client(
                StreamableHttpTransport("http://localhost:18100/mcp", headers=headers)
            ) as client:
                result = await DvdMcpClient(client).search(
                    "Расстояние от здания школы до открытой автомобильной стоянки",
                    document_names=["LOCAL SDK TEST"],
                    version="2026",
                    limit=3,
                    scenario_id=91001,
                )
                save(output / "document-search.json", result)
                if not any(
                    h.get("name") == "LOCAL SDK TEST"
                    and str(h.get("version")) == "2026"
                    and h.get("numbering") == "1.1"
                    for h in result.get("hits", [])
                ):
                    raise ValueError(
                        "Seeded clause is not available through real DVD search"
                    )
                documents["passed"] = True
        except Exception as exc:
            documents["error"] = type(exc).__name__ + ": " + str(exc)[:250]
        report["cases"].append(documents)
        save(output / "report.json", report)
        print(
            f"Documents: {'PASS' if documents['passed'] else documents['error']}",
            flush=True,
        )
        if not documents["passed"]:
            return False
        canonical = {"name": "real_canonical_compliance_route", "passed": False}
        try:
            from .compliance_probe import verify_compliance_transport

            result = await verify_compliance_transport(await headers_for(http, config))
            save(output / "canonical-compliance.json", result)
            canonical["passed"] = True
        except Exception as exc:
            canonical["error"] = type(exc).__name__ + ": " + str(exc)[:1000]
        report["cases"].append(canonical)
        save(output / "report.json", report)
        print(
            f"Compliance route: {'PASS' if canonical['passed'] else canonical['error']}",
            flush=True,
        )
        if not canonical["passed"]:
            return False
        for sid in EXPECTED:
            row = {"scenario_id": sid, "passed": False}
            try:
                result = await provision(await headers_for(http, config), sid)
                save(output / f"{sid}.json", result)
                verify_calculation(sid, result)
                row["passed"] = True
            except Exception as exc:
                row["error"] = type(exc).__name__ + ": " + str(exc)[:250]
            report["cases"].append(row)
            save(output / "report.json", report)
            print(f"{sid}: {'PASS' if row['passed'] else row['error']}", flush=True)
        geometry = {"name": "real_geometry_buffers", "passed": False}
        try:
            headers = await headers_for(http, config)
            async with Client(
                StreamableHttpTransport("http://localhost:18002/mcp", headers=headers)
            ) as client:
                source = entities(91001, "physical_object", 7)
                result = await client.call_tool(
                    "CreateBuffers",
                    {
                        "buffer_info": {
                            "parking": {
                                "buffer_size": 50,
                                "buffer_type": "round",
                                "title": "50 m",
                            }
                        },
                        "objects": {"parking": source},
                    },
                )
                result = json.loads(result.content[0].text)
                save(output / "buffers.json", result)
                if not is_fifty_metre_buffer(result["parking"], source):
                    raise ValueError("Incorrect 50 m buffer")
                from shapely.geometry import shape

                buffer = shape(result["parking"]["features"][0]["geometry"])
                if not buffer.intersects(
                    shape(entities(91001, "service", 22)["features"][0]["geometry"])
                ):
                    raise ValueError("Before school must violate distance")
                if buffer.intersects(
                    shape(entities(91006, "service", 22)["features"][0]["geometry"])
                ):
                    raise ValueError("Revised school must pass distance")
                geometry["passed"] = True
        except Exception as exc:
            geometry["error"] = type(exc).__name__
        report["cases"].append(geometry)
        print(f"Geometry: {'PASS' if geometry['passed'] else 'FAIL'}", flush=True)
    report["passed"] = all(r["passed"] for r in report["cases"])
    save(output / "report.json", report)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        0 if asyncio.run(run(dotenv_values(args.env_file), args.output)) else 1
    )
