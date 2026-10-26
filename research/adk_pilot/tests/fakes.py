"""Doubles shared by the pilot tests: one scripted LLM, MCP results, and both
engines wired to them."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from research.adk_pilot.gmart_adk.restriction.service import AdkRestrictionService
from src.agents.services.restriction.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.service_entities import GeometryToolCallResult
from tests.helpers import FakeLlmClient

SERVICES = ["Школа"]
OBJECTS = ["Жилой дом"]
MODEL = "test-model"
TEMPERATURE = 0.7
SCENARIO_ID = 42
REQUEST_ID = "req-1"
CONTEXT = '{"affected_count": 1, "affected_objects": [{"name": "Жилой дом 1"}]}'


def collection(feature_id: str) -> dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": feature_id,
                "geometry": {"type": "Point", "coordinates": [30.3, 59.9]},
                "properties": {"name": feature_id},
            }
        ],
    }


def plan_json(mode: str, **overrides: Any) -> str:
    plan: dict[str, Any] = {
        "mode": mode,
        "source_entities": [{"name": "Школа", "entity_type": "service"}],
        "target_entities": [],
        "buffer_rules": [
            {"source_name": "Школа", "buffer_size": 100, "title": "Зона 100 м"}
        ],
        "restriction_rules": [],
        "selection_reasons": [],
        "confidence": 0.9,
        "clarification_question": None,
        "original": "запрос",
    }
    if mode == "restrictions":
        plan["target_entities"] = [
            {"name": "Жилой дом", "entity_type": "physical_object"}
        ]
        plan["restriction_rules"] = [
            {
                "source_name": "Школа",
                "target_names": ["Жилой дом"],
                "title": "Зона 100 м",
                "description": "Жилые дома в зоне школ",
            }
        ]
    if mode == "needs_clarification":
        plan.update(
            source_entities=[],
            buffer_rules=[],
            clarification_question="Какой радиус зоны нужен?",
        )
    plan.update(overrides)
    return json.dumps(plan, ensure_ascii=False)


class RecordingLlm(FakeLlmClient):
    """FakeLlmClient that snapshots each request, ``think`` included.

    The current planner appends to the message list it already sent when it
    repairs JSON, so the list is copied at call time.
    """

    async def chat(
        self, model=None, messages=None, options=None, stream=False, **kwargs
    ):
        result = await super().chat(
            model=model,
            messages=copy.deepcopy(messages),
            options=copy.deepcopy(options),
            stream=stream,
            **kwargs,
        )
        self.chat_calls[-1].think = kwargs.get("think")
        return result

    def requests(self) -> list[dict[str, Any]]:
        return [
            {
                "model": call.model,
                "messages": call.messages,
                "options": call.options,
                "format": call.format,
                "stream": call.stream,
                "think": call.think,
            }
            for call in self.chat_calls
        ]


@dataclass
class Scenario:
    query: str
    plans: list[str]
    texts: list[str] = field(default_factory=lambda: ["Объяснение плана.", "Ответ."])
    history: list[dict[str, str]] = field(default_factory=list)

    def llm(self) -> RecordingLlm:
        llm = RecordingLlm()
        llm.json_responses = list(self.plans)
        llm.answer_texts = list(self.texts)
        return llm


def tool_executor() -> SimpleNamespace:
    def result(data: dict, tool: str) -> GeometryToolCallResult:
        return GeometryToolCallResult(
            tool_result=data,
            tool_calls=[{"tool_name": tool, "arguments": {"scenario_id": SCENARIO_ID}}],
            messages=[],
        )

    layers = {"Школа": collection("school"), "Жилой дом": collection("building")}
    buffers = {"Школа": collection("buffer")}
    restrictions = {
        "objects": collection("building"),
        "generators": collection("buffer"),
    }
    return SimpleNamespace(
        retrieve_layers_for_plan=AsyncMock(return_value=result(layers, "get_layers")),
        run_buffer_plan=AsyncMock(return_value=result(buffers, "buffer")),
        run_restriction_plan=AsyncMock(return_value=result(restrictions, "intersect")),
    )


def plan_builder(llm: RecordingLlm) -> RestrictionPlanBuilder:
    builder = RestrictionPlanBuilder(llm)
    builder.get_entity_catalogs = AsyncMock(return_value=(SERVICES, OBJECTS))
    return builder


def context_builder() -> SimpleNamespace:
    return SimpleNamespace(
        generate_restrictions_context=AsyncMock(return_value=CONTEXT)
    )


def legacy_service(llm: RecordingLlm) -> RestrictionParserService:
    """The current service with its stores stubbed, as tests/unit does."""
    service = object.__new__(RestrictionParserService)
    service.llm_client = llm
    service.plan_builder = plan_builder(llm)
    service.tool_executor = tool_executor()
    service.context_builder = context_builder()
    service.state_store = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        new_request_id=lambda: REQUEST_ID,
        create=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        save_checkpoint=AsyncMock(),
        set_status=AsyncMock(),
        buffer_event=AsyncMock(),
    )
    return service


def adk_service(llm: RecordingLlm, **kwargs: Any) -> AdkRestrictionService:
    return AdkRestrictionService(
        llm,
        default_model=MODEL,
        plan_builder=plan_builder(llm),
        tool_executor=kwargs.pop("tool_executor", None) or tool_executor(),
        context_builder=context_builder(),
        **kwargs,
    )


async def run_legacy(scenario: Scenario) -> tuple[list[dict], RecordingLlm]:
    llm = scenario.llm()
    service = legacy_service(llm)
    if scenario.history:
        # The current service reads history from ChatStorage; feed it directly.
        service.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
        service.build_llm_history = lambda *a, **k: scenario.history
    events = [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=TEMPERATURE,
            model=MODEL,
            user_query=scenario.query,
            scenario_id=SCENARIO_ID,
            token_ref=["token"],
            chat_id="chat-1" if scenario.history else None,
            request_id=REQUEST_ID,
            persist_history=False,
        )
    ]
    return events, llm


async def run_adk(scenario: Scenario, **kwargs: Any) -> tuple[list[dict], RecordingLlm]:
    llm = scenario.llm()
    service = adk_service(llm, **kwargs)
    events = [
        event
        async for event in service.run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=TEMPERATURE,
            model=MODEL,
            user_query=scenario.query,
            scenario_id=SCENARIO_ID,
            request_id=REQUEST_ID,
            history=scenario.history,
        )
    ]
    return events, llm
