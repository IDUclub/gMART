"""Compliance pipeline around the scope: document choice, its reply, empty scopes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from src.agents.schema.restrictions_response import RestrictionsResponse
from src.agents.services.compilance.compliance_scope import (
    ChoiceReply,
    ComplianceScope,
    ScopeOutcome,
)
from src.agents.services.compilance.compliance_territory import TerritoryDocuments
from src.agents.services.pipeline_state import PipelineStep
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)

_CHOICE = {
    "query": "Проверь нормы по школам из СП 42",
    "topics": ["школа"],
    "entities": ["школа"],
    "documents": [],
    "references": ["СП 42"],
    "matched": True,
    "candidates": [
        {"name": "СП 42.13330.2011", "executable_count": 1},
        {"name": "СП 42.13330.2016", "executable_count": 4},
    ],
}


# Documents in force on the scenario's territory (IDU_DVD).
_IN_FORCE = ("СП 42.13330.2011", "СП 42.13330.2016")


def _service(*, pending=None, outcome=None, reply=None, restrictions=()):
    service = object.__new__(RestrictionParserService)
    service.compliance_territory = SimpleNamespace(
        resolve=AsyncMock(
            return_value=TerritoryDocuments(status="ok", allowed=_IN_FORCE)
        )
    )
    service.state_store = SimpleNamespace(
        exists=AsyncMock(return_value=False),
        new_request_id=lambda: "request-1",
        create=AsyncMock(),
        get_compliance_choice=AsyncMock(return_value=pending),
        set_compliance_choice=AsyncMock(),
        get_checkpoint=AsyncMock(return_value={}),
        save_checkpoint=AsyncMock(),
        buffer_event=AsyncMock(),
        set_status=AsyncMock(),
    )
    service.get_chat_messages = AsyncMock(return_value=SimpleNamespace(messages=[]))
    service.add_single_message = AsyncMock()
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *args: None
    )
    service.compliance_scope = SimpleNamespace(
        resolve=AsyncMock(return_value=outcome or ScopeOutcome(kind="scoped")),
        resolve_choice=AsyncMock(return_value=reply),
        empty_scope_message=AsyncMock(return_value="Исполнимых норм нет."),
    )
    service.normgraph_retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=SimpleNamespace(
                restrictions=list(restrictions),
                unsupported_count=3,
                tool_call={"function": {"name": "list_restrictions", "arguments": {}}},
            )
        )
    )
    service._run_executable_compliance = _record_execution(service)
    return service


def _record_execution(service):
    service.executed_with = None

    async def run(**kwargs):
        service.executed_with = kwargs
        yield {
            "type": "chunk",
            "content": {"text": "Проверка завершена.", "done": True},
        }

    return run


async def _run(service, query, chat_id="chat-1"):
    return [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query=query,
            scenario_id=772,
            token_ref=["token"],
            chat_id=chat_id,
            persist_history=False,
            normgraph_mcp_client=object(),
            history_agent="compliance",
        )
    ]


async def test_ambiguous_document_asks_and_keeps_the_choice_for_the_chat():
    outcome = ScopeOutcome(
        kind="choice", message="Выберите документ:\n1. …", choice=_CHOICE
    )
    service = _service(outcome=outcome)

    events = await _run(service, "Проверь нормы по школам из СП 42")

    clarification = next(e for e in events if e["type"] == "clarification")
    # The SSE controller re-serializes events through the schema: options survive.
    serialized = RestrictionsResponse.model_validate(clarification).model_dump()
    assert serialized["content"]["options"][1]["number"] == 2
    assert clarification["content"]["question"].startswith("Выберите документ")
    assert [o["value"] for o in clarification["content"]["options"]] == [
        "СП 42.13330.2011",
        "СП 42.13330.2016",
    ]
    service.state_store.set_compliance_choice.assert_awaited_once_with(
        "chat-1", _CHOICE
    )
    service.normgraph_retriever.retrieve.assert_not_awaited()
    assert service.executed_with is None


async def test_reply_to_the_choice_checks_the_original_request_with_its_documents():
    service = _service(
        pending=_CHOICE,
        reply=ChoiceReply(kind="selected", documents=("СП 42.13330.2016",)),
        restrictions=[{"id": "r1"}],
    )

    await _run(service, "2")

    service.compliance_scope.resolve.assert_not_awaited()
    service.state_store.set_compliance_choice.assert_awaited_once_with("chat-1", None)
    filters = service.normgraph_retriever.retrieve.await_args.kwargs["filters"]
    assert filters == {
        "entities": ["школа"],
        "document_names": ["СП 42.13330.2016"],
    }
    assert service.executed_with["scope"] == ComplianceScope(
        topics=("школа",),
        entities=("школа",),
        documents=("СП 42.13330.2016",),
        allowed_documents=_IN_FORCE,
    )
    saved = {
        call.args[1]: call.args[2]
        for call in service.state_store.save_checkpoint.await_args_list
    }
    assert saved[PipelineStep.COMPLIANCE_SCOPE]["kind"] == "scoped"


async def test_unreadable_choice_is_asked_again_and_kept():
    service = _service(pending=_CHOICE, reply=ChoiceReply(kind="unresolved"))

    events = await _run(service, "7")

    clarification = next(e for e in events if e["type"] == "clarification")
    assert "такого варианта нет" in clarification["content"]["question"]
    assert "2. СП 42.13330.2016" in clarification["content"]["question"]
    service.state_store.set_compliance_choice.assert_not_awaited()
    service.normgraph_retriever.retrieve.assert_not_awaited()
    [(_, step, saved)] = [
        call.args for call in service.state_store.save_checkpoint.await_args_list
    ]
    assert step == PipelineStep.COMPLIANCE_SCOPE
    assert saved["kind"] == "choice"


async def test_reconnect_after_a_question_replays_it_without_rerunning():
    service = _service()
    service.state_store.exists = AsyncMock(return_value=True)
    service.state_store.get_buffered_events = AsyncMock(
        return_value=[{"type": "clarification", "content": {"question": "Выберите"}}]
    )
    service.state_store.get_checkpoint = AsyncMock(
        return_value={
            PipelineStep.COMPLIANCE_TERRITORY: TerritoryDocuments(
                status="ok", allowed=_IN_FORCE
            ).to_dict(),
            PipelineStep.COMPLIANCE_SCOPE: ScopeOutcome(
                kind="choice", message="Выберите", choice=_CHOICE
            ).to_dict(),
        }
    )

    events = [
        event
        async for event in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query="7",
            scenario_id=772,
            token_ref=["token"],
            chat_id="chat-1",
            request_id="request-1",
            persist_history=False,
            normgraph_mcp_client=object(),
            history_agent="compliance",
        )
    ]

    assert [event["type"] for event in events] == ["clarification"]
    service.compliance_scope.resolve.assert_not_awaited()
    service.compliance_scope.resolve_choice.assert_not_awaited()
    service.normgraph_retriever.retrieve.assert_not_awaited()


async def test_new_request_instead_of_a_choice_drops_it():
    service = _service(pending=_CHOICE, reply=ChoiceReply(kind="not_choice"))

    await _run(service, "Проверь все нормы")

    service.state_store.set_compliance_choice.assert_awaited_once_with("chat-1", None)
    service.compliance_scope.resolve.assert_awaited_once()
    assert service.normgraph_retriever.retrieve.await_args.kwargs["filters"] == {
        "document_names": list(_IN_FORCE)
    }


async def test_filtered_check_without_executable_norms_explains_and_stops():
    scope = ComplianceScope(topics=("школа",), entities=("школа",))
    service = _service(outcome=ScopeOutcome(kind="scoped", scope=scope))

    events = await _run(service, "Проверь нормы по школам", chat_id=None)

    assert events[-1] == {
        "type": "chunk",
        "content": {"text": "Исполнимых норм нет.", "done": True},
    }
    service.compliance_scope.empty_scope_message.assert_awaited_once()
    assert service.compliance_scope.empty_scope_message.await_args.kwargs == {
        "found": 3
    }
    assert service.executed_with is None


async def test_unscoped_empty_corpus_keeps_the_previous_report_path():
    service = _service()

    await _run(service, "Проверь соответствие", chat_id=None)

    service.compliance_scope.empty_scope_message.assert_not_awaited()
    assert service.executed_with is not None


def test_summary_text_starts_with_the_scope():
    summary = {
        "total_norms": 1,
        "violated_norms": 0,
        "passed_norms": 1,
        "unverifiable_norms": 0,
        "unsupported_norms": 0,
        "partial_norms": 0,
        "results": [],
        "scope": ComplianceScope(topics=("школа",), entities=("школа",)).to_dict(),
    }

    text = RestrictionParserService._compliance_summary_text(summary)

    assert text.startswith("Область проверки — темы: «школа».\n\n")


async def test_pending_choice_outlives_the_pipeline_window(state_store):
    from src.agents.services.pipeline_state import (
        COMPLIANCE_CHOICE_TTL,
        PIPELINE_TTL,
    )

    await state_store.set_compliance_choice("chat-1", _CHOICE)

    assert await state_store.get_compliance_choice("chat-1") == _CHOICE
    ttl = await state_store._redis.ttl("pipeline:chat-1:compliance_choice")
    assert PIPELINE_TTL < ttl <= COMPLIANCE_CHOICE_TTL
    await state_store.set_compliance_choice("chat-1", None)
    assert await state_store.get_compliance_choice("chat-1") is None
