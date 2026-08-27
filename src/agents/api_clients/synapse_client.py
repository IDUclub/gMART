from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx


class SynapseClientError(RuntimeError):
    pass


class SynapseAuthError(SynapseClientError):
    pass


class SynapseUnavailableError(SynapseClientError):
    pass


class SynapseResponseError(SynapseClientError):
    def __init__(self, status_code: int, operation: str) -> None:
        super().__init__(f"Synapse {operation} returned HTTP {status_code}")
        self.status_code = status_code
        self.operation = operation


@dataclass(frozen=True)
class SynapseSseEvent:
    event: str
    event_id: str | None
    data: dict[str, Any]


class SynapseApiClient:
    """Authenticated client for the immutable Synapse HTTP API."""

    def __init__(
        self,
        base_url: str,
        email: str,
        password: str,
        *,
        workflow_id: str,
        run_config_id: str,
        approval_mode: str = "auto",
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self._password = password
        self.workflow_id = workflow_id
        self.run_config_id = run_config_id
        self.approval_mode = approval_mode
        self._request_timeout = httpx.Timeout(
            timeout_seconds, connect=min(timeout_seconds, 10.0)
        )
        self._sse_timeout = httpx.Timeout(None, connect=min(timeout_seconds, 10.0))
        self._owned_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._request_timeout,
        )
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._access_expires_at = 0.0
        self._auth_lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owned_client:
            await self._client.aclose()

    @staticmethod
    def _token_exp(token: str) -> float:
        try:
            payload = token.split(".", 2)[1]
            decoded = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
            value = json.loads(decoded).get("exp")
            return float(value) if value is not None else 0.0
        except (IndexError, ValueError, TypeError, json.JSONDecodeError):
            return 0.0

    def _store_tokens(self, payload: dict[str, Any]) -> None:
        access_token = payload.get("access_token")
        refresh_token = payload.get("refresh_token")
        if not isinstance(access_token, str) or not access_token:
            raise SynapseAuthError("Synapse authentication response is invalid")
        self._access_token = access_token
        self._refresh_token = refresh_token if isinstance(refresh_token, str) else None
        self._access_expires_at = self._token_exp(access_token)

    async def _login(self) -> None:
        try:
            response = await self._client.post(
                "/api/auth/login",
                json={"email": self.email, "password": self._password},
            )
        except httpx.HTTPError as exc:
            raise SynapseUnavailableError(
                "Synapse authentication is unavailable"
            ) from exc
        if response.status_code != 200:
            raise SynapseAuthError("Synapse technical-user authentication failed")
        self._store_tokens(response.json())

    async def _refresh(self) -> None:
        if not self._refresh_token:
            await self._login()
            return
        try:
            response = await self._client.post(
                "/api/auth/refresh", json={"refresh_token": self._refresh_token}
            )
        except httpx.HTTPError as exc:
            raise SynapseUnavailableError(
                "Synapse token refresh is unavailable"
            ) from exc
        if response.status_code != 200:
            await self._login()
            return
        self._store_tokens(response.json())

    async def _ensure_access_token(self, *, force_refresh: bool = False) -> str:
        async with self._auth_lock:
            expires_soon = bool(
                self._access_expires_at and self._access_expires_at <= time.time() + 30
            )
            if force_refresh or expires_soon:
                await self._refresh()
            elif not self._access_token:
                await self._login()
            assert self._access_token is not None
            return self._access_token

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        for auth_attempt in range(2):
            token = await self._ensure_access_token(force_refresh=auth_attempt == 1)
            response = None
            attempts = 3 if method.upper() == "GET" else 1
            for transport_attempt in range(attempts):
                try:
                    response = await self._client.request(
                        method,
                        path,
                        params=params,
                        json=json_body,
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=self._request_timeout,
                    )
                    break
                except httpx.HTTPError as exc:
                    if transport_attempt + 1 == attempts:
                        raise SynapseUnavailableError(
                            f"Synapse {operation} is unavailable"
                        ) from exc
                    await asyncio.sleep(0.5 * 2**transport_attempt)
            assert response is not None
            if response.status_code == 401 and auth_attempt == 0:
                continue
            if not 200 <= response.status_code < 300:
                raise SynapseResponseError(response.status_code, operation)
            if response.status_code == 204:
                return {}
            payload = response.json()
            if not isinstance(payload, dict):
                raise SynapseClientError(
                    f"Synapse {operation} response is not an object"
                )
            return payload
        raise SynapseAuthError("Synapse rejected the refreshed technical-user token")

    async def create_project(self, prompt: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/api/projects",
            operation="project creation",
            json_body={
                "user_prompt": prompt,
                "approval_mode": self.approval_mode,
                "workflow_id": self.workflow_id,
                "run_config_id": self.run_config_id,
            },
        )

    async def get_project(self, project_id: str) -> dict[str, Any]:
        return await self._request(
            "GET", f"/api/projects/{project_id}", operation="project lookup"
        )

    async def find_projects(self, marker: str) -> list[dict[str, Any]]:
        payload = await self._request(
            "GET",
            "/api/projects",
            operation="project reconciliation",
            params={
                "q": marker,
                "workflow_id": self.workflow_id,
                "run_config_id": self.run_config_id,
                "limit": 10,
            },
        )
        projects = payload.get("projects")
        return [item for item in projects or [] if isinstance(item, dict)]

    async def send_message(
        self,
        project_id: str,
        content: str,
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/projects/{project_id}/messages",
            operation="message submission",
            params={"run_id": run_id} if run_id else None,
            json_body={"content": content, "metadata": metadata or {}},
        )

    async def stop_project(self, project_id: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            f"/api/projects/{project_id}/stop",
            operation="project stop",
            json_body={"reason": "Stopped through gMART"},
        )

    async def stream_events(
        self,
        project_id: str,
        *,
        run_id: str | None,
        last_event_id: str | None,
        since: str | None,
    ) -> AsyncIterator[SynapseSseEvent]:
        for auth_attempt in range(2):
            token = await self._ensure_access_token(force_refresh=auth_attempt == 1)
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "text/event-stream",
            }
            if last_event_id:
                headers["Last-Event-ID"] = last_event_id
            params = {
                key: value
                for key, value in {"run_id": run_id, "since": since}.items()
                if value
            }
            try:
                async with self._client.stream(
                    "GET",
                    f"/api/projects/{project_id}/events",
                    params=params,
                    headers=headers,
                    timeout=self._sse_timeout,
                ) as response:
                    if response.status_code == 401 and auth_attempt == 0:
                        continue
                    if response.status_code != 200:
                        raise SynapseResponseError(response.status_code, "event stream")
                    async for event in self._parse_sse(response.aiter_lines()):
                        yield event
                    return
            except httpx.HTTPError as exc:
                raise SynapseUnavailableError(
                    "Synapse event stream is unavailable"
                ) from exc
        raise SynapseAuthError("Synapse rejected the refreshed event-stream token")

    @staticmethod
    async def _parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[SynapseSseEvent]:
        event_type = "message"
        event_id: str | None = None
        data_lines: list[str] = []
        async for line in lines:
            if line == "":
                if data_lines:
                    raw_data = "\n".join(data_lines)
                    try:
                        data = json.loads(raw_data)
                    except json.JSONDecodeError:
                        data = {"text": raw_data}
                    if not isinstance(data, dict):
                        data = {"value": data}
                    yield SynapseSseEvent(event_type, event_id, data)
                event_type = "message"
                event_id = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event_type = value
            elif field == "id":
                event_id = value
            elif field == "data":
                data_lines.append(value)
        if data_lines:
            raw_data = "\n".join(data_lines)
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                data = {"text": raw_data}
            if not isinstance(data, dict):
                data = {"value": data}
            yield SynapseSseEvent(event_type, event_id, data)
