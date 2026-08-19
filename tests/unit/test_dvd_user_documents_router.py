from __future__ import annotations

from io import BytesIO

import pytest
from fastapi import UploadFile

from src.agents.common.exceptions.base_exceptions import AgentsInputException
from src.agents.routers.dvd_controller import (
    delete_user_document,
    list_user_documents,
    stream_user_document_job,
    update_user_document,
    upload_user_document,
)


class FakeDvdApiClient:
    def __init__(self):
        self.uploads = []
        self.updates = []
        self.deletes = []

    async def upload_user_document(self, file, **kwargs):
        self.uploads.append((file.filename, kwargs))
        return {"job_id": "job-1", "status": "queued"}

    async def list_user_documents(self, **kwargs):
        return {
            "count": 1,
            "documents": [{"doc_id": "doc-1", "name": "Мой документ", "version": "1"}],
            "scope": kwargs,
        }

    async def get_user_document_job(self, job_id):
        return {
            "job_id": job_id,
            "status": "done",
            "stage": "indexing",
            "stage_index": 7,
            "stage_total": 7,
            "task_progress": 100,
            "overall_progress": 100,
        }

    async def update_user_document(self, name, file, **kwargs):
        self.updates.append((name, file.filename, kwargs))
        return {"job_id": "job-2", "status": "queued"}

    async def delete_user_document(self, name, **kwargs):
        self.deletes.append((name, kwargs))
        return {
            "name": name,
            "versions_removed": ["1"],
            "points_deleted": 4,
            "points_updated": 0,
        }


class FakeUrbanApiClient:
    async def get_project_by_scenario(self, token, scenario_id):
        assert token == "test-token"
        assert scenario_id == 772
        return 4242


async def test_upload_resolves_project_from_scenario():
    dvd_api = FakeDvdApiClient()
    result = await upload_user_document(
        file=UploadFile(filename="mine.docx", file=BytesIO(b"document")),
        project_id=None,
        scenario_id="772",
        name="Мой документ",
        version=None,
        token="test-token",
        dvd_api_client=dvd_api,
        urban_api_client=FakeUrbanApiClient(),
    )

    assert result == {"job_id": "job-1", "status": "queued"}
    filename, kwargs = dvd_api.uploads[0]
    assert filename == "mine.docx"
    assert kwargs["project_id"] == "4242"
    assert kwargs["scenario_id"] == "772"
    assert kwargs["name"] == "Мой документ"


async def test_list_forwards_user_scope():
    result = await list_user_documents(
        project_id="4242",
        scenario_id="772",
        dvd_api_client=FakeDvdApiClient(),
    )

    assert result["documents"][0]["name"] == "Мой документ"
    assert result["scope"] == {"project_id": "4242", "scenario_id": "772"}


async def test_scope_is_required():
    with pytest.raises(AgentsInputException):
        await list_user_documents(
            project_id=None,
            scenario_id=None,
            dvd_api_client=FakeDvdApiClient(),
        )


async def test_update_forwards_file_and_user_scope():
    dvd_api = FakeDvdApiClient()
    result = await update_user_document(
        name="Мой документ",
        file=UploadFile(filename="replacement.docx", file=BytesIO(b"new")),
        project_id="4242",
        scenario_id="772",
        version="2",
        dvd_api_client=dvd_api,
    )

    assert result == {"job_id": "job-2", "status": "queued"}
    assert dvd_api.updates == [
        (
            "Мой документ",
            "replacement.docx",
            {"project_id": "4242", "scenario_id": "772", "version": "2"},
        )
    ]


async def test_delete_forwards_name_scope_and_version():
    dvd_api = FakeDvdApiClient()
    result = await delete_user_document(
        name="Мой документ",
        project_id="4242",
        scenario_id="772",
        version=None,
        dvd_api_client=dvd_api,
    )

    assert result["points_deleted"] == 4
    assert dvd_api.deletes == [
        (
            "Мой документ",
            {"project_id": "4242", "scenario_id": "772", "version": None},
        )
    ]


class FakeRequest:
    async def is_disconnected(self):
        return False


async def test_job_progress_is_streamed_as_sse_snapshot():
    response = await stream_user_document_job(
        job_id="job-1",
        request=FakeRequest(),
        dvd_api_client=FakeDvdApiClient(),
    )

    chunks = [chunk async for chunk in response.body_iterator]
    body = b"".join(
        chunk if isinstance(chunk, bytes) else chunk.encode() for chunk in chunks
    ).decode()
    assert response.media_type == "text/event-stream"
    assert '"overall_progress":100' in body
    assert '"status":"done"' in body
