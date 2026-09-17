from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from src.agents.api_clients.pzz_api_client import PzzApiClient
from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.mcp_clients.pzz_mcp_client import PzzMcpClient


async def test_upload_registers_owner_and_prevents_cross_user_read(state_store):
    client = PzzApiClient("http://pzz", None, "owner", state_store)
    client._request = AsyncMock(return_value={"upload_id": "file-1"})
    await client.upload("a.json", b"{}", "application/json")
    await client.validate_upload("file-1")
    other = PzzApiClient("http://pzz", None, "other", state_store)
    other._request = AsyncMock()
    with pytest.raises(ValueError, match="belong"):
        await other.read_geojson("file-1")
    other._request.assert_not_awaited()


async def test_mcp_checks_building_upload_ownership_before_submission():
    client = PzzMcpClient(None)
    client.api_client = SimpleNamespace(
        validate_upload=AsyncMock(side_effect=ValueError("foreign upload"))
    )
    client.execute_tool = AsyncMock()
    with pytest.raises(ValueError, match="foreign"):
        await client.call(
            "submit_building_pzz_check_task", {"buildings_upload_id": "foreign"}
        )
    client.execute_tool.assert_not_awaited()


async def test_rest_forwards_service_identity_and_maps_expired_token(
    monkeypatch, state_store
):
    real_client = httpx.AsyncClient
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(401, json={"detail": "expired"})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    auth = SimpleNamespace(
        get_authorization_headers=AsyncMock(
            return_value={"Authorization": "Bearer service-token"}
        )
    )
    client = PzzApiClient("http://pzz", auth, "caller", state_store)
    with pytest.raises(TokenExpiredError):
        await client.classify_summary("task-id")
    assert requests[0].headers["Authorization"] == "Bearer service-token"
    assert requests[0].headers["X-User-Id"] == "caller"
    assert requests[0].url.path == "/tasks/task-id/classify-summary"


async def test_custom_reference_files_use_rest_form_contract(state_store):
    client = PzzApiClient("http://pzz", None, "owner", state_store)
    for upload_id in ("cadastral", "zones", "labels", "classifier"):
        await state_store.register_pzz_upload(upload_id, "owner")
    client._request = AsyncMock(return_value={"external_id": "task"})
    await client.submit_file_task(
        "pzz_check",
        {
            "cadastral_upload_id": "cadastral",
            "pzz_zones_upload_id": "zones",
            "cadastral_vri_col": "vri",
            "pzz_zone_code_col": "zone",
            "pzz_zone_name_col": "name",
            "priority": 1,
            "force_recompute": False,
        },
        "labels",
        "classifier",
        "request-id",
    )
    call = client._request.await_args
    assert call.args == ("POST", "/tasks/pzz-check")
    assert call.kwargs["data"]["cadastral_feature_collection_upload_id"] == "cadastral"
    assert call.kwargs["data"]["pzz_zone_vri_labels_upload_id"] == "labels"
    assert call.kwargs["data"]["vri_classifier_upload_id"] == "classifier"
    assert call.kwargs["data"]["force_recompute"] == "false"
    assert call.kwargs["headers"]["Idempotency-Key"] == "request-id"


async def test_zone_table_conversion_precedes_upload(state_store):
    client = PzzApiClient("http://pzz", None, "owner", state_store)
    client._request = AsyncMock(
        side_effect=[{"zones": [{"zone_code": "Ж-1"}]}, {"upload_id": "converted"}]
    )
    await client.upload("zones.csv", b"code,name", "text/csv", kind="zone_descriptions")
    first, second = client._request.await_args_list
    assert first.args == ("POST", "/pzz/zone-descriptions/convert")
    assert second.args == ("POST", "/uploads")
    assert second.kwargs["files"]["file"][0] == "zone-descriptions.json"
    assert await state_store.get_pzz_upload_owner("converted") == "owner"


def test_vector_upload_reprojects_to_wgs84(tmp_path):
    import json

    import geopandas as gpd
    from shapely.geometry import Point

    path = tmp_path / "layer.parquet"
    gpd.GeoDataFrame(
        {"vri": ["ИЖС"]}, geometry=[Point(1000000, 1000000)], crs=3857
    ).to_parquet(path)
    result = json.loads(PzzApiClient._vector_to_geojson(path.read_bytes(), ".parquet"))
    assert result["type"] == "FeatureCollection"
    assert result["features"][0]["properties"]["vri"] == "ИЖС"
    assert result["features"][0]["geometry"]["coordinates"][0] == pytest.approx(
        8.98315284
    )
