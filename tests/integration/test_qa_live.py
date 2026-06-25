"""Integration: a full end-to-end QA run against the live stack (Ollama + IDU_DVD MCP + Redis).

Skips automatically unless all three services are up. ChatStorage / Urban API are intentionally
stubbed (history + persistence) so the test focuses on the real RAG path: an LLM that plans the
retrieval, a real IDU_DVD vector search, an LLM that drafts + self-reviews the answer, and Redis
buffering. The document base must contain ingested documents for the answer to be grounded.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.helpers import answer_text, final_chunk, types_of

pytestmark = pytest.mark.integration


async def test_live_qa_end_to_end(require_redis, require_ollama, require_dvd_mcp):
    from src.agents.services.dvd_rag_service import DvdRagService
    from src.agents.services.pipeline_state import PipelineStateStore

    store = PipelineStateStore(require_redis)
    # Real Ollama (require_ollama is the base URL) + real Redis; ChatStorage/Urban API stubbed.
    svc = DvdRagService(require_ollama, Mock(), Mock(), store)
    svc.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    svc._schedule_persist_answer = Mock()

    events = [
        event
        async for event in svc.run_document_qa_pipeline(
            dvd_mcp_client=require_dvd_mcp,
            token=os.environ.get("URBAN_API_TOKEN", "x"),
            model=os.environ.get("DVD_TEST_MODEL", "gpt-oss:20b"),
            temperature=0.0,
            user_query="Какие требования к озеленению территории?",
            chat_id="live-smoke-chat",
        )
    ]

    assert types_of(events)[0] == "pipeline_started"
    assert final_chunk(events) is not None
    assert answer_text(events).strip() != ""
