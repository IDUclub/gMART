"""Exercise canonical compliance routing, persisted plans and real geometry."""

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from redis.asyncio import Redis

from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.services.compilance.compliance_executor import (
    ComplianceTemplateExecutor,
)
from src.agents.services.compilance.compliance_result_harness import (
    ComplianceResultHarness,
)
from src.agents.services.normgraph.normgraph_restriction_retriever import (
    NormGraphRestrictionRetriever,
)
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)

from .acceptance import complete_spatial_result


async def verify_compliance_transport(headers):
    # This route must not use a model: the document clause already has a saved
    # executable plan. Inject only its real dependencies; unexpected free-form
    # planning fails visibly instead of silently spending an LLM acceptance run.
    service = object.__new__(RestrictionParserService)
    redis = Redis.from_url("redis://localhost:16389/0", decode_responses=True)
    service.state_store = PipelineStateStore(redis)
    service.compliance_executor = ComplianceTemplateExecutor()
    service.compliance_result_harness = ComplianceResultHarness()
    service.normgraph_retriever = NormGraphRestrictionRetriever(None)
    results = []
    try:
        async with (
            Client(
                StreamableHttpTransport("http://localhost:18002/mcp", headers=headers)
            ) as idu,
            Client(
                StreamableHttpTransport("http://localhost:18020/mcp", headers=headers)
            ) as norms,
        ):
            for sid, violations in [(91001, 1), (91006, 0)]:
                events = [
                    event
                    async for event in service._run_restriction_execution_pipline(
                        mcp_client=IduMcpClient(idu),
                        temperature=0,
                        model="unused",
                        user_query="Проверь только пункт 1.1 синтетического документа LOCAL SDK TEST, версия 2026: от здания школы до открытой автомобильной стоянки не менее 50 м.",
                        scenario_id=sid,
                        token_ref=[headers.get("Authorization", "")],
                        persist_history=False,
                        normgraph_mcp_client=NormGraphMcpClient(norms),
                        history_agent="compliance",
                    )
                ]
                rows = [
                    e["content"] for e in events if e["type"] == "compliance_result"
                ]
                if len(rows) != 1 or not complete_spatial_result(
                    rows[0], sid, violations
                ):
                    raise ValueError(
                        f"Canonical compliance routing/coverage failed for {sid}: {rows}"
                    )
                results.append({"scenario_id": sid, "events": events})
    finally:
        await redis.aclose()
    return results
