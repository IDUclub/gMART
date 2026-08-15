"""Memory hygiene around the geometry tools.

A restrictions call builds hundreds of megabytes of short-lived objects in a
worker thread; glibc keeps the freed blocks in that thread's arena instead of
returning them, so the process only ever grows. release_memory collects cycles
and hands the arenas back — and must run even when the tool raises, or a failing
call would be the one that costs the most.
"""

from __future__ import annotations

import pytest

from src.idu_mcp.common import memory as memory_module
from src.idu_mcp.common.memory import memory_snapshot, release_memory, rss_bytes
from src.idu_mcp.tools_services.geometry_tools import GeometryTools


def test_release_memory_collects_and_trims(monkeypatch):
    calls = []
    monkeypatch.setenv("MCP_RELEASE_MEMORY", "1")
    monkeypatch.setattr(memory_module.gc, "collect", lambda: calls.append("gc"))

    class FakeLibc:
        @staticmethod
        def malloc_trim(arg):
            calls.append(("trim", arg))

    monkeypatch.setattr(memory_module, "_libc", lambda: FakeLibc)

    release_memory()

    assert calls == ["gc", ("trim", 0)]


def test_release_can_be_switched_off(monkeypatch):
    calls = []
    monkeypatch.setenv("MCP_RELEASE_MEMORY", "0")
    monkeypatch.setattr(memory_module.gc, "collect", lambda: calls.append("gc"))

    release_memory()

    assert calls == []


def test_release_survives_a_platform_without_malloc_trim(monkeypatch):
    monkeypatch.setenv("MCP_RELEASE_MEMORY", "1")
    monkeypatch.setattr(memory_module, "_libc", lambda: None)

    release_memory()  # musl: nothing to trim, and nothing to raise


def test_snapshot_reports_rss_and_types():
    snapshot = memory_snapshot(top=5)

    assert snapshot["rss_mb"] > 0
    assert snapshot["objects"] > 0
    assert len(snapshot["top_types"]) <= 5
    assert rss_bytes() > 0


@pytest.mark.asyncio
async def test_buffers_release_memory_even_when_the_tool_fails(monkeypatch):
    released = []
    monkeypatch.setattr(
        "src.idu_mcp.tools_services.geometry_tools.release_memory",
        lambda: released.append(True),
    )
    tools = GeometryTools()

    await tools.async_generate_geometry_buffers({}, {})
    assert released == [True]

    def boom(*_args, **_kwargs):
        raise ValueError("bad geometry")

    monkeypatch.setattr(tools, "create_restrictions", boom)
    with pytest.raises(ValueError):
        await tools.async_create_restrictions({}, [], [], {})

    assert released == [True, True]
