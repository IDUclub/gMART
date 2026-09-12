"""Completed requests replay their public events without invoking any dependency."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.services.pipeline_state import PipelineStatus
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)


@pytest.mark.parametrize("family", ["restriction", "compliance", "provision"])
@pytest.mark.parametrize(
    "status", [PipelineStatus.DONE, PipelineStatus.FAILED, PipelineStatus.CANCELLED]
)
@pytest.mark.parametrize("chat_id", [None, "chat-1"])
async def test_terminal_public_reconnect_is_read_only(
    state_store, family, status, chat_id
):
    request_id = state_store.new_request_id()
    await state_store.create(
        request_id,
        chat_id="chat-1",
        user_query="q",
        scenario_id=772,
        model="m",
        temperature=0,
    )
    buffered = [
        {"type": "pipeline_started", "content": {"request_id": request_id}},
        {
            "type": "tool_call",
            "content": {"execution_mode": "sequential", "tool_calls": []},
        },
        {"type": "chunk", "content": {"text": "Saved answer", "done": True}},
    ]
    if status != PipelineStatus.DONE:
        buffered[-1] = {
            "type": "error",
            "content": {"message": "Saved failure", "traceback": ""},
        }
    for event in buffered:
        await state_store.buffer_event(request_id, event)
    await state_store.set_status(request_id, status)
    before = await state_store.get_state(request_id)
    cls = ProvisionService if family == "provision" else RestrictionParserService
    service = object.__new__(cls)
    service.state_store = state_store
    service.resolve_model = AsyncMock(
        side_effect=AssertionError("Replay must not resolve a model")
    )
    service.get_chat_messages = AsyncMock(
        side_effect=AssertionError("Replay must not fetch history")
    )
    service._schedule_add_message_parts_to_chat = Mock()
    boundary = SimpleNamespace(
        execute_tool=AsyncMock(
            side_effect=AssertionError("Replay must not execute tools")
        )
    )
    kwargs = dict(
        token="t",
        model=None,
        temperature=0,
        user_query="q",
        scenario_id=772,
        request_id=request_id,
        chat_id=chat_id,
    )
    if family == "provision":
        run = service.run_provision_pipeline
        kwargs.update(idu_mcp_client=boundary, effects_mcp_client=boundary)
    elif family == "compliance":
        run = service.run_compliance_pipeline
        kwargs.update(mcp_client=boundary, normgraph_mcp_client=boundary)
    else:
        run = service.run_restriction_execution_pipline
        kwargs.update(mcp_client=boundary)
    expected = [e for e in buffered if e["type"] != "tool_call"]
    for _ in range(2):
        assert [event async for event in run(**kwargs)] == expected
    assert await state_store.get_buffered_events(request_id) == buffered
    assert await state_store.get_state(request_id) == before
    service.resolve_model.assert_not_awaited()
    service.get_chat_messages.assert_not_awaited()
    service._schedule_add_message_parts_to_chat.assert_not_called()
    boundary.execute_tool.assert_not_awaited()
