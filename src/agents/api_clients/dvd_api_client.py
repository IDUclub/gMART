from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from fastapi import UploadFile
from idu_service_auth import KeycloakTokenClient

from src.agents.common.exceptions.api_exceptions import DownstreamServiceError
from src.agents.common.exceptions.base_exceptions import (
    AgentsInputException,
    AgentsNotFound,
)
from src.common.service_auth import service_headers


class DvdApiClient:
    """Request-scoped REST client for IDU_DVD user-document operations."""

    def __init__(
        self,
        base_url: str,
        service_auth: KeycloakTokenClient,
        user_id: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.service_auth = service_auth
        self.user_id = user_id

    async def upload_user_document(
        self,
        file: UploadFile,
        *,
        project_id: str,
        scenario_id: str | None = None,
        name: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        file.file.seek(0)
        data = {"project_id": project_id}
        if scenario_id:
            data["scenario_id"] = scenario_id
        if name:
            data["name"] = name
        if version:
            data["version"] = version

        files = {
            "file": (
                file.filename or "document",
                file.file,
                file.content_type or "application/octet-stream",
            )
        }
        return await self._request("POST", "/user-documents", data=data, files=files)

    async def list_user_documents(
        self,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, str] = {}
        if project_id:
            params["project_id"] = project_id
        if scenario_id:
            params["scenario_id"] = scenario_id
        return await self._request("GET", "/user-documents", params=params)

    async def get_user_document_job(self, job_id: str) -> dict[str, Any]:
        """Return the current user-owned ingestion snapshot from IDU_DVD."""
        return await self._request("GET", f"/user-documents/jobs/{job_id}")

    async def update_user_document(
        self,
        name: str,
        file: UploadFile,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Submit a delta update for a document owned by the current user."""
        file.file.seek(0)
        params: dict[str, str] = {}
        if project_id:
            params["project_id"] = project_id
        if scenario_id:
            params["scenario_id"] = scenario_id
        data = {"version": version} if version else {}
        files = {
            "file": (
                file.filename or "document",
                file.file,
                file.content_type or "application/octet-stream",
            )
        }
        encoded_name = quote(name, safe="")
        return await self._request(
            "PATCH",
            f"/user-documents/{encoded_name}",
            params=params,
            data=data,
            files=files,
        )

    async def delete_user_document(
        self,
        name: str,
        *,
        project_id: str | None = None,
        scenario_id: str | None = None,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Delete all versions, or one version, of a user-owned document."""
        params: dict[str, str] = {}
        if project_id:
            params["project_id"] = project_id
        if scenario_id:
            params["scenario_id"] = scenario_id
        if version:
            params["version"] = version
        encoded_name = quote(name, safe="")
        return await self._request(
            "DELETE", f"/user-documents/{encoded_name}", params=params
        )

    async def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = await service_headers(self.service_auth, self.user_id)
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.request(method, url, headers=headers, **kwargs)
        except httpx.RequestError as exc:
            raise DownstreamServiceError(
                service=self.base_url,
                downstream_status=None,
                message=f"IDU_DVD is unavailable: {exc}",
                error_input=repr(exc),
            ) from exc
        return self._parse_response(response)

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            body: Any = response.json()
        except ValueError:
            body = response.text

        if 200 <= response.status_code < 300:
            return body if isinstance(body, dict) else {"result": body}

        detail = body.get("detail") if isinstance(body, dict) else body
        message = str(detail or f"IDU_DVD returned {response.status_code}")
        if response.status_code == 404:
            raise AgentsNotFound(message=message, error_input=body)
        if response.status_code in {400, 409, 413, 415, 422}:
            raise AgentsInputException(message=message, error_input=body)
        raise DownstreamServiceError(
            service=self.base_url,
            downstream_status=response.status_code,
            message=message,
            error_input=body,
        )
