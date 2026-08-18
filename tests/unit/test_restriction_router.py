import inspect
from unittest.mock import Mock

import pytest

from src.agents.dto.restriction_request_dto import RestrictionRequestDTO
from src.agents.routers import restriction_parser_controller as controller
from src.agents.services.restriction_parser_service import RestrictionParserService


def test_standard_restriction_pipeline_cannot_accept_normgraph_client():
    parameters = inspect.signature(
        RestrictionParserService.run_restriction_execution_pipline
    ).parameters

    assert "normgraph_mcp_client" not in parameters
    assert (
        "normgraph_mcp_client"
        in inspect.signature(
            RestrictionParserService.run_compliance_pipeline
        ).parameters
    )


@pytest.mark.asyncio
async def test_restriction_route_does_not_pass_normgraph_client(monkeypatch):
    captured: dict = {}

    async def fake_stream(pipeline, *args, **kwargs):
        captured.update(kwargs)
        yield {"type": "chunk", "content": {"text": "готово", "done": True}}

    monkeypatch.setattr(controller, "stream_with_error_handling", fake_stream)
    service = Mock()
    request = RestrictionRequestDTO(
        request="Построй буфер вокруг школ",
        scenario_id=772,
    )

    chunks = [
        chunk
        async for chunk in controller.generate_restrictions_response(
            request=Mock(),
            user_request=request,
            idu_mcp_client=Mock(),
            restriction_service=service,
        )
    ]

    assert len(chunks) == 1
    assert "normgraph_mcp_client" not in captured
    assert captured["mcp_client"] is not None
