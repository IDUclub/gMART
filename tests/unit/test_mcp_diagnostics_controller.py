"""Unit tests for the allowlisted MCP diagnostics API handlers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.agents.routers.mcp_diagnostics_controller import (
    call_tool,
    list_prompts,
    list_sources,
    list_tools,
)
from src.agents.schema.mcp_diagnostics import McpToolCallRequest
from src.agents.services.mcp_diagnostics_service import _is_safe_tool


@pytest.mark.asyncio
async def test_lists_configured_sources():
    service = MagicMock()
    service.sources.return_value = [
        {"id": "idu", "title": "IDU MCP", "available": True}
    ]

    result = await list_sources(service)

    assert result[0]["id"] == "idu"
    service.sources.assert_called_once_with()


@pytest.mark.asyncio
async def test_lists_tools_for_selected_source():
    service = AsyncMock()
    service.list_tools.return_value = [
        {
            "type": "function",
            "source": "urban",
            "group": "projects",
            "read_only": True,
            "function": {"name": "GetScenarios"},
        }
    ]

    result = await list_tools("urban", service)

    assert result[0]["group"] == "projects"
    service.list_tools.assert_awaited_once_with("urban")


@pytest.mark.asyncio
async def test_lists_prompts_for_selected_source():
    service = AsyncMock()
    service.list_prompts.return_value = [{"name": "SearchDocuments"}]

    result = await list_prompts("dvd", service)

    assert result == [{"name": "SearchDocuments"}]
    service.list_prompts.assert_awaited_once_with("dvd")


@pytest.mark.asyncio
async def test_calls_tool_with_source_group_arguments_and_meta():
    service = AsyncMock()
    service.call_tool.return_value = {"type": "FeatureCollection", "features": []}
    request = McpToolCallRequest(
        source="urban",
        group="projects",
        name="GetScenarioObjects",
        arguments={"scenario_id": 772},
        meta={"source": "diagnostics"},
    )

    result = await call_tool(request, service)

    assert result == {"result": {"type": "FeatureCollection", "features": []}}
    service.call_tool.assert_awaited_once_with(
        "urban",
        "GetScenarioObjects",
        {"scenario_id": 772},
        group="projects",
        meta={"source": "diagnostics"},
    )


def test_rejects_unknown_source_and_non_object_arguments():
    with pytest.raises(ValidationError):
        McpToolCallRequest(source="external", name="GetServices", arguments={})
    with pytest.raises(ValidationError):
        McpToolCallRequest(name="GetServices", arguments=[])


def test_read_only_annotation_takes_priority_over_tool_name():
    forbidden = SimpleNamespace(
        name="GetButMutates",
        annotations={"readOnlyHint": False},
    )
    allowed = SimpleNamespace(
        name="CustomLookup",
        annotations={"readOnlyHint": True},
    )

    assert _is_safe_tool(forbidden) is False
    assert _is_safe_tool(allowed) is True


def test_legacy_tools_use_conservative_name_allowlist():
    assert _is_safe_tool(SimpleNamespace(name="CalculateObjectEffects")) is True
    assert _is_safe_tool(SimpleNamespace(name="DeleteScenario")) is False
