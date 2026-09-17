"""REST operations not exposed by the PZZ MCP catalogue (uploads and VRI summary)."""

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import httpx

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.common.service_auth import ServiceTokenAuth


class PzzApiClient:
    def __init__(self, base_url: str, service_auth, user_id: str, state_store):
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id
        self.state_store = state_store
        self.auth = ServiceTokenAuth(service_auth, user_id)

    async def _request(self, method: str, path: str, **kwargs):
        async with httpx.AsyncClient(auth=self.auth, timeout=120) as client:
            response = await client.request(method, self.base_url + path, **kwargs)
            if response.status_code == 401:
                raise TokenExpiredError("PZZ authentication expired")
            response.raise_for_status()
            return response.json()

    async def upload(
        self, filename: str, content: bytes, content_type: str, kind: str = "layer"
    ):
        suffix = Path(filename).suffix.lower()
        if suffix in {".gpkg", ".gml", ".kml", ".parquet", ".geoparquet"}:
            content = await asyncio.to_thread(self._vector_to_geojson, content, suffix)
            filename, content_type = "layer.geojson", "application/geo+json"
        elif kind == "zone_descriptions" and suffix in {".csv", ".xlsx"}:
            converted = await self._request(
                "POST",
                "/pzz/zone-descriptions/convert",
                files={"file": (filename, content, content_type)},
            )
            content = json.dumps(converted["zones"], ensure_ascii=False).encode()
            filename, content_type = "zone-descriptions.json", "application/json"
        result = await self._request(
            "POST", "/uploads", files={"file": (filename, content, content_type)}
        )
        await self.state_store.register_pzz_upload(result["upload_id"], self.user_id)
        return result

    async def validate_upload(self, upload_id: str) -> None:
        if await self.state_store.get_pzz_upload_owner(upload_id) != self.user_id:
            raise ValueError("Upload must belong to the caller; use POST /pzz/uploads")

    async def read_geojson(self, upload_id: str) -> dict:
        await self.validate_upload(upload_id)
        return await self._request("GET", "/uploads/" + quote(upload_id, safe=""))

    async def classify_summary(self, external_id: str) -> dict:
        return await self._request(
            "GET", "/tasks/" + quote(external_id, safe="") + "/classify-summary"
        )

    @staticmethod
    def _vector_to_geojson(content: bytes, suffix: str) -> bytes:
        import geopandas as gpd

        with TemporaryDirectory(prefix="gmart-pzz-") as directory:
            path = Path(directory) / ("layer" + suffix)
            path.write_bytes(content)
            frame = (
                gpd.read_parquet(path)
                if suffix in {".parquet", ".geoparquet"}
                else gpd.read_file(path)
            )
            if frame.crs is None:
                raise ValueError("Uploaded vector layer must declare its CRS")
            return frame.to_crs(4326).to_json().encode()

    async def submit_file_task(
        self, mode, arguments, labels_id, classifier_id, request_id
    ):
        """The REST equivalent of submit_* also accepts custom reference files."""
        field_names = {
            "cadastral_upload_id": "cadastral_feature_collection_upload_id",
            "pzz_zones_upload_id": "pzz_zones_feature_collection_upload_id",
        }
        data, files = {}, {}
        for key, value in arguments.items():
            if key.endswith("_upload_id"):
                await self.validate_upload(value)
            if key.endswith("_geojson"):
                prefix = key.removesuffix("_geojson")
                files[prefix + "_feature_collection_file"] = (
                    prefix + ".geojson",
                    json.dumps(value).encode(),
                    "application/geo+json",
                )
            else:
                data[field_names.get(key, key)] = (
                    str(value).lower() if isinstance(value, bool) else str(value)
                )
        for key, upload_id in (
            ("pzz_zone_vri_labels_upload_id", labels_id),
            ("vri_classifier_upload_id", classifier_id),
        ):
            if upload_id:
                await self.validate_upload(upload_id)
                data[key] = upload_id
        return await self._request(
            "POST",
            "/tasks/pzz-check" if mode == "pzz_check" else "/tasks/classify-only",
            data=data,
            files=files or None,
            headers={"Idempotency-Key": request_id},
        )
