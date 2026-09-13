"""Control inputs are read through the same HTTP boundary as Urban API."""

from fastapi.testclient import TestClient

from tests.integration.industrial.sources import create_app


def test_prepared_versions_have_distinct_capacity_and_shared_population():
    client = TestClient(create_app())
    before = client.get(
        "/api/v1/scenarios/91005/services_with_geometry", params={"service_type_id": 22}
    )
    after = client.get(
        "/api/v1/scenarios/91006/services_with_geometry", params={"service_type_id": 22}
    )
    assert before.status_code == after.status_code == 200
    assert sum(f["properties"]["capacity"] for f in before.json()["features"]) == 800
    assert sum(f["properties"]["capacity"] for f in after.json()["features"]) == 1300
    for sid in (91005, 91006):
        values = client.get(f"/api/v1/scenarios/{sid}/indicators_values").json()
        assert (
            next(v["value"] for v in values if v["indicator"]["indicator_id"] == 1)
            == 12000
        )
    assert before.json()["meta"]["revision"] != after.json()["meta"]["revision"]


def test_unknown_data_and_writes_cannot_silently_succeed():
    client = TestClient(create_app())
    assert client.get("/api/v1/scenarios/772").status_code == 404
    assert client.post("/api/v1/scenarios/91005", json={}).status_code == 405
    assert client.get("/api/v1/scenarios/91005/unknown").status_code == 404


def test_missing_normative_fault_does_not_change_other_source_records():
    client = TestClient(
        create_app(
            faults=[
                {
                    "path": "/api/v1/territory/911/normatives",
                    "kind": "missing_normative",
                    "service_type_id": 22,
                }
            ]
        )
    )
    rows = client.get("/api/v1/territory/911/normatives").json()
    assert [row["service_type"]["id"] for row in rows] == [21]
    assert client.get("/api/v1/scenarios/91001").json()["scenario_id"] == 91001


def test_transient_source_fault_is_bounded_and_audited():
    client = TestClient(
        create_app(
            faults=[
                {"path": "/api/v1/scenarios/91001", "kind": "unavailable", "times": 1}
            ]
        )
    )
    assert client.get("/api/v1/scenarios/91001").status_code == 503
    assert client.get("/api/v1/scenarios/91001").status_code == 200
    audit = client.get("/audit").json()
    assert audit[0]["fault"] == "unavailable"
    assert "fault" not in audit[1]


async def test_mcp_source_catalog_supports_real_typed_selection():
    from fastmcp import Client

    from tests.integration.industrial.server import groups

    async with Client(groups["projects"]) as client:
        tools = {t.name: t for t in await client.list_tools()}
        assert "GetScenarioServiceTypes" in tools
        assert (
            "service_type_id" in tools["GetScenarioServices"].inputSchema["properties"]
        )
        result = await client.call_tool(
            "GetScenarioServicesWithGeometry",
            {"scenario_id": 91001, "service_type_id": 22},
        )
        import json

        data = json.loads(result.content[0].text)
        assert len(data["features"]) == 1
        assert data["features"][0]["properties"]["capacity"] == 800
