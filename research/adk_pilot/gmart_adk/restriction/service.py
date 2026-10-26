"""Drop-in restriction service backed by the ADK workflow.

``AdkRestrictionService.run_restriction_execution_pipline`` takes the keyword
arguments the A2A executor and the REST controller pass to
``RestrictionParserService`` and yields the same pipeline items, so either
caller can be pointed at it without changing the frontend or A2A contracts.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from google.adk import Runner
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.apps import App
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.sessions import BaseSessionService, InMemorySessionService
from google.adk.workflow import RetryConfig
from google.genai import types

from research.adk_pilot.gmart_adk.events import LegacyEventAdapter
from research.adk_pilot.gmart_adk.harness import (
    HISTORY,
    MODEL,
    TEMPERATURE,
    USER_QUERY,
)
from research.adk_pilot.gmart_adk.llm import GmartLlm
from research.adk_pilot.gmart_adk.restriction.workflow import (
    NORMGRAPH,
    SCENARIO_ID,
    STREAMED_AGENTS,
    RestrictionWorkflowDeps,
    build_restriction_workflow,
    current_mcp_client,
)
from src.agents.services.restriction.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.restriction.restriction_context import (
    RestrictionContextBuilder,
)
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.restriction.restriction_tool_executor import (
    RestrictionToolExecutor,
)

APP_NAME = "gmart_restrictions"
USER_ID = "gmart"


class AdkRestrictionService:
    def __init__(
        self,
        llm_adapter: Any,
        *,
        default_model: str = "default",
        plan_builder: RestrictionPlanBuilder | None = None,
        tool_executor: RestrictionToolExecutor | None = None,
        context_builder: RestrictionContextBuilder | None = None,
        session_service: BaseSessionService | None = None,
        plugins: Sequence[BasePlugin] = (),
        tool_retry: RetryConfig | None = None,
    ) -> None:
        self.default_model = default_model
        self.workflow = build_restriction_workflow(
            RestrictionWorkflowDeps(
                llm=GmartLlm(model=default_model, adapter=llm_adapter),
                plan_builder=plan_builder or RestrictionPlanBuilder(llm_adapter),
                tool_executor=tool_executor or RestrictionToolExecutor(),
                context_builder=context_builder or RestrictionContextBuilder(),
                tool_retry=tool_retry,
            )
        )
        self.session_service = session_service or InMemorySessionService()
        self.app = App(name=APP_NAME, root_agent=self.workflow, plugins=list(plugins))
        self.runner = Runner(app=self.app, session_service=self.session_service)

    async def run_restriction_execution_pipline(
        self,
        mcp_client: Any,
        temperature: float,
        model: str | None,
        user_query: str,
        scenario_id: int,
        request_id: str | None = None,
        history: list[dict] | None = None,
        normgraph_restrictions: list[dict[str, Any]] | None = None,
        **_unused: Any,  # token, persist_history, chat_id: not part of the pilot
    ) -> AsyncGenerator[dict[str, Any], None]:
        request_id = request_id or uuid.uuid4().hex
        yield RestrictionParserService._pipeline_started_event(request_id)

        session = await self.session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=request_id,
            state={
                MODEL: model or self.default_model,
                TEMPERATURE: temperature,
                USER_QUERY: user_query,
                HISTORY: history or [],
                SCENARIO_ID: scenario_id,
                NORMGRAPH: normgraph_restrictions or [],
            },
        )
        adapter = LegacyEventAdapter(STREAMED_AGENTS)
        token = current_mcp_client.set(mcp_client)
        try:
            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=user_query)]
                ),
                run_config=RunConfig(streaming_mode=StreamingMode.SSE),
            ):
                for item in adapter.convert(event):
                    yield item
        finally:
            try:
                current_mcp_client.reset(token)
            except ValueError:
                # Closed from another context (e.g. a consumer that stopped
                # early); the value dies with the context it was set in.
                pass
