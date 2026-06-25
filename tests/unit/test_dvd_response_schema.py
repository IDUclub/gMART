"""Unit tests for the SSE event contract — every event the service emits must validate
against DvdResponse, and the error-wrapper's iteration-less chunk must validate too."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.schema.dvd_response import DvdResponse
from src.agents.services.dvd_rag_service import DvdRagService


def test_status_event_validates():
    model = DvdResponse(**DvdRagService._status("searching", "ищу"))
    assert model.type == "status"


def test_chunk_event_validates_with_iteration():
    model = DvdResponse(**DvdRagService._chunk("text", done=False, iteration=2))
    assert model.content.iteration == 2


def test_tool_call_event_validates():
    event = DvdRagService._tool_call(
        "rag_search",
        [{"function": {"name": "search_all", "arguments": {}}}],
        mcp_source="DVD_MCP_URL",
    )
    model = DvdResponse(**event)
    assert model.content.mcp_source == "DVD_MCP_URL"


def test_pipeline_started_event_validates():
    model = DvdResponse(**DvdRagService._pipeline_started_event("rid-1"))
    assert model.content.request_id == "rid-1"


def test_chat_created_service_event_validates():
    model = DvdResponse(**DvdRagService._chat_created_event("cid", "title"))
    assert model.content.event.chat_id == "cid"


def test_warning_event_validates():
    model = DvdResponse(**DvdRagService._project_lookup_failed_event(772))
    assert model.type == "warning"
    assert model.content.code == "project_id_unavailable"
    assert model.content.scenario_id == 772


def test_error_wrapper_chunk_without_iteration_validates():
    # stream_with_error_handling emits chunks with no `iteration` key
    model = DvdResponse(type="chunk", content={"text": "", "done": True})
    assert model.content.iteration == 0


def test_error_event_validates():
    model = DvdResponse(type="error", content={"message": "boom", "traceback": "tb"})
    assert model.type == "error"


def test_unknown_status_literal_is_rejected():
    with pytest.raises(ValidationError):
        DvdResponse(type="status", content={"status": "nope", "text": "x"})
