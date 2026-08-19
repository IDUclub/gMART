from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import UploadFile

from src.agents.api_clients.dvd_api_client import DvdApiClient
from src.agents.common.exceptions.base_exceptions import AgentsInputException


class FakeAsyncClient:
    response: httpx.Response
    calls: list[tuple[str, str, dict]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.response


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    FakeAsyncClient.calls = []
    monkeypatch.setattr(
        "src.agents.api_clients.dvd_api_client.service_headers",
        AsyncMock(return_value={"Authorization": "Bearer service", "X-User-Id": "u-1"}),
    )
    monkeypatch.setattr(
        "src.agents.api_clients.dvd_api_client.httpx.AsyncClient", FakeAsyncClient
    )


def client():
    return DvdApiClient("http://dvd", Mock(), "u-1")


async def test_upload_forwards_multipart_and_user_headers():
    FakeAsyncClient.response = httpx.Response(
        202,
        json={"job_id": "job-1", "status": "queued"},
        request=httpx.Request("POST", "http://dvd/user-documents"),
    )
    upload = UploadFile(filename="mine.docx", file=BytesIO(b"body"))

    result = await client().upload_user_document(
        upload, project_id="42", scenario_id="772", name="Mine", version="1"
    )

    assert result == {"job_id": "job-1", "status": "queued"}
    method, url, kwargs = FakeAsyncClient.calls[0]
    assert (method, url) == ("POST", "http://dvd/user-documents")
    assert kwargs["headers"]["X-User-Id"] == "u-1"
    assert kwargs["data"] == {
        "project_id": "42",
        "scenario_id": "772",
        "name": "Mine",
        "version": "1",
    }
    assert kwargs["files"]["file"][0] == "mine.docx"


async def test_upload_validation_error_is_preserved():
    FakeAsyncClient.response = httpx.Response(
        415,
        json={"detail": "Поддерживаются только DOCX"},
        request=httpx.Request("POST", "http://dvd/user-documents"),
    )
    upload = UploadFile(filename="mine.pdf", file=BytesIO(b"body"))

    with pytest.raises(AgentsInputException, match="Поддерживаются"):
        await client().upload_user_document(upload, project_id="42")


async def test_job_status_uses_user_scoped_endpoint_and_headers():
    FakeAsyncClient.response = httpx.Response(
        200,
        json={"job_id": "job-1", "status": "processing", "overall_progress": 48},
        request=httpx.Request("GET", "http://dvd/user-documents/jobs/job-1"),
    )

    result = await client().get_user_document_job("job-1")

    assert result["overall_progress"] == 48
    method, url, kwargs = FakeAsyncClient.calls[0]
    assert (method, url) == ("GET", "http://dvd/user-documents/jobs/job-1")
    assert kwargs["headers"]["X-User-Id"] == "u-1"


async def test_update_forwards_encoded_name_file_and_scope():
    FakeAsyncClient.response = httpx.Response(
        202,
        json={"job_id": "job-2", "status": "queued"},
        request=httpx.Request(
            "PATCH", "http://dvd/user-documents/%D0%9C%D0%BE%D0%B9%20%D0%B4%D0%BE%D0%BA"
        ),
    )
    upload = UploadFile(filename="replacement.docx", file=BytesIO(b"new body"))

    result = await client().update_user_document(
        "Мой док",
        upload,
        project_id="42",
        scenario_id="772",
        version="2",
    )

    assert result == {"job_id": "job-2", "status": "queued"}
    method, url, kwargs = FakeAsyncClient.calls[0]
    assert method == "PATCH"
    assert url.endswith(
        "/user-documents/%D0%9C%D0%BE%D0%B9%20%D0%B4%D0%BE%D0%BA"
    )
    assert kwargs["params"] == {"project_id": "42", "scenario_id": "772"}
    assert kwargs["data"] == {"version": "2"}
    assert kwargs["files"]["file"][0] == "replacement.docx"


async def test_delete_forwards_encoded_name_and_optional_version():
    FakeAsyncClient.response = httpx.Response(
        200,
        json={
            "name": "Регламент / 2026",
            "versions_removed": ["2"],
            "points_deleted": 3,
            "points_updated": 1,
        },
        request=httpx.Request("DELETE", "http://dvd/user-documents/document"),
    )

    result = await client().delete_user_document(
        "Регламент / 2026", project_id="42", version="2"
    )

    assert result["versions_removed"] == ["2"]
    method, url, kwargs = FakeAsyncClient.calls[0]
    assert method == "DELETE"
    assert url.endswith(
        "/user-documents/%D0%A0%D0%B5%D0%B3%D0%BB%D0%B0%D0%BC%D0%B5%D0%BD%D1%82%20%2F%202026"
    )
    assert kwargs["params"] == {"project_id": "42", "version": "2"}
