from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compliance_result_harness import ComplianceResultHarness
from src.agents.services.pipeline_state import PipelineStatus
from src.agents.services.restriction_parser_service import RestrictionParserService
from tests.helpers import FakeLlmClient


def _summary(request_id: str, status: str = "unverifiable") -> dict:
    result = {
        "restriction_id": f"restriction-{request_id}",
        "template": "distance_from_source",
        "template_version": 1,
        "verification_status": status,
        "compliance_status": "unknown",
        "coverage": {
            "applicable_objects": 0,
            "checked_objects": 0,
            "unchecked_objects": 0,
            "fill_rate": 0,
        },
        "summary": {"violated_objects": 0, "passed_objects": 0},
        "missing_requirements": ["template_execution_failed"],
        "warnings": ["Urban API returned status 404"],
        "source": {
            "document_name": "СП 42.13330.2016",
            "clause_number": "10.4",
            "extraction_text": "Радиус обслуживания ограничен 1000 м",
        },
        "evidence": [],
    }
    if status == "unsupported":
        result["warnings"] = ["check_plan_validation_failed"]
        result["missing_requirements"] = ["planner_status:unsupported"]
    return {
        "request_id": request_id,
        "total_norms": 1,
        "violated_norms": 0,
        "passed_norms": 0,
        "unverifiable_norms": int(status == "unverifiable"),
        "unsupported_norms": int(status == "unsupported"),
        "not_applicable_norms": 0,
        "partial_norms": 0,
        "results": [result],
    }


def _stored_summary(summary: dict) -> dict:
    return {
        "role": "assistant",
        "parts": [
            {
                "kind": "data",
                "payload": {
                    "event_type": "compliance_summary",
                    "content": summary,
                },
            }
        ],
    }


def test_extracts_the_latest_summary_from_chat_storage_data_parts():
    first = _summary("first")
    latest = _summary("latest", status="unsupported")

    assert (
        ComplianceResultHarness.latest_summary(
            [_stored_summary(first), _stored_summary(latest)]
        )
        == latest
    )


def test_accepts_legacy_custom_kind_for_existing_history():
    summary = _summary("legacy")
    messages = [
        {
            "role": "assistant",
            "parts": [
                {"kind": "compliance_summary", "payload": summary},
            ],
        }
    ]

    assert ComplianceResultHarness.latest_summary(messages) == summary


def test_prepares_grounded_follow_up_without_large_geometry_payloads():
    summary = _summary("current")
    summary["results"][0]["violated_features"] = {
        "type": "FeatureCollection",
        "features": [{"geometry": {"coordinates": [1, 2]}}],
    }
    summary["results"][0]["effective_requirements"] = {
        "layers": [
            {
                "role": "objects",
                "entity": "Жилой дом",
                "entity_type": "physical_object",
                "required": True,
            }
        ],
        "attributes": [],
    }
    harness = ComplianceResultHarness()

    prepared = harness.prepare_follow_up(
        "Какие ограничения не удалось проверить и почему?",
        [_stored_summary(summary)],
        history=[{"role": "user", "content": "Проверь нормы"}],
    )

    assert prepared is not None
    assert prepared.summary["request_id"] == "current"
    system_prompt = prepared.messages[0]["content"]
    assert "Urban API returned status 404" in system_prompt
    assert "Радиус обслуживания ограничен 1000 м" in system_prompt
    assert "Жилой дом" in system_prompt
    assert "coordinates" not in system_prompt
    assert "Ни одна норма не была полностью проверена" in system_prompt
    assert prepared.messages[-1] == {
        "role": "user",
        "content": "Какие ограничения не удалось проверить и почему?",
    }


def test_explicit_rerun_request_is_not_intercepted():
    harness = ComplianceResultHarness()

    prepared = harness.prepare_follow_up(
        "Перепроверь ограничения заново",
        [_stored_summary(_summary("current"))],
        history=[],
    )

    assert prepared is None


def test_normalizes_misleading_no_violations_opening_when_nothing_was_checked():
    answer = ComplianceResultHarness.normalize_answer(
        _summary("current"),
        "**Нарушения не обнаружены.**\n\nШесть норм проверить не удалось.",
    )

    assert answer.startswith("Ни одна норма не была полностью проверена")
    assert "Нарушения не обнаружены" not in answer


def test_missing_summary_is_not_intercepted():
    harness = ComplianceResultHarness()

    assert harness.prepare_follow_up("Почему?", [], history=[]) is None


def test_follow_up_status_is_part_of_the_sse_contract():
    response = RestrictionsResponse.model_validate(
        {
            "type": "status",
            "content": {
                "status": "compliance_result_analysis",
                "text": "Анализирую результат последней проверки",
            },
        }
    )

    assert response.content.status == "compliance_result_analysis"


async def test_service_answers_follow_up_without_rerunning_compliance_pipeline():
    summary = _summary("current")
    service = object.__new__(RestrictionParserService)
    service.llm_client = FakeLlmClient()
    service.llm_client.answer_texts = [
        "Норму не удалось проверить из-за ошибки Urban API."
    ]
    service.compliance_result_harness = ComplianceResultHarness()
    service.get_chat_messages = AsyncMock(
        return_value=SimpleNamespace(messages=[_stored_summary(summary)])
    )
    service.add_single_message = AsyncMock()
    service.normgraph_retriever = SimpleNamespace(retrieve=AsyncMock())
    service.compliance_executor = SimpleNamespace(execute=AsyncMock())
    service.state_store = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        new_request_id=lambda: "follow-up-request",
        create=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        buffer_event=AsyncMock(),
        set_status=AsyncMock(),
    )

    events = [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0.7,
            model="gpt-oss-20b",
            user_query="Почему это ограничение не удалось проверить?",
            scenario_id=772,
            token_ref=["token"],
            chat_id="chat-1",
            persist_history=True,
            normgraph_mcp_client=object(),
            history_agent="compliance",
        )
    ]

    assert "ошибки Urban API" in "".join(
        event["content"]["text"] for event in events if event["type"] == "chunk"
    )
    service.normgraph_retriever.retrieve.assert_not_awaited()
    service.compliance_executor.execute.assert_not_awaited()
    service.state_store.set_status.assert_awaited_once_with(
        "follow-up-request", PipelineStatus.DONE
    )
