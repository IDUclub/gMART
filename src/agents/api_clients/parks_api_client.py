"""ParkPlanner 1.0.1 contract, verified against /v3/api-docs and its web client."""

import json

import httpx


class ParksApiClient:
    def __init__(self, base_url: str, *, transport=None):
        self.base_url = base_url.rstrip("/")
        self.transport = transport

    async def _request(self, method, path, **kwargs):
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            response = await client.request(method, self.base_url + path, **kwargs)
            response.raise_for_status()
            data = response.json()
        if isinstance(data, dict) and data.get("code", 0) != 0:
            raise ValueError("ParkPlanner returned an application error")
        return data

    async def generate(self, inputs):
        data = await self._request(
            "POST",
            "/map/generate",
            json={
                "parkGeoJson": json.dumps(inputs["park_geojson"], ensure_ascii=False),
                "tags": inputs["tags"],
                "locationRequests": inputs.get("location_requests") or [],
            },
        )
        if not data.get("taskId"):
            raise ValueError("ParkPlanner did not return taskId")
        return str(data["taskId"])

    async def status(self, task_id):
        return await self._request(
            "GET", "/map/getResultStatus", params={"taskId": task_id}
        )

    async def result(self, task_id):
        data = await self._request("GET", "/map/getResult", params={"taskId": task_id})
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
            raise ValueError("ParkPlanner returned invalid GeoJSON")
        return data

    async def economy(self, task_id):
        return await self._request(
            "GET", "/map/getEconomyParams", params={"taskId": task_id}
        )
