import asyncio
from types import SimpleNamespace

import pytest

from src.agents.common.exceptions.base_exceptions import AgentsUnauthorizedException
from src.agents.dto.dvd_request_dto import DocumentQaRequestDTO
from src.agents.routers.dvd_controller import (
    resolve_document_qa_token,
    stream_document_qa,
)
from src.agents.services.dvd.dialogue import pending_question, resolve_reply
from src.agents.services.dvd.runs import run_metadata, stream_document_run
from src.agents.services.service_entities.dvd_plan import StructureRetrievalPlan
from tests.helpers import answer_text, plan_json, verdict_json
from tests.unit.test_dvd_structured_retrieval import CODE, EDITION, Pages, run


def test_article_reply_selects_one_identity_and_new_question_releases_old_address():
    pending = pending_question(
        StructureRetrievalPlan(retrieval_mode="structure", pattern="3.3"),
        candidates(),
        CODE,
    )
    reply = resolve_reply("статья 52", pending)
    assert reply["selected_ids"] == ["article-52"]
    assert "52" in reply["plan"]["pattern"]
    assert resolve_reply("Что указано в пункте 4.2?", pending) is None
    assert resolve_reply("редакция от 01.01.1999", pending) == {"unresolved": True}


async def wait_terminal(store, request_id):
    for _ in range(100):
        meta = await run_metadata(store, request_id)
        if meta and meta["status"] != "running":
            return meta
        await asyncio.sleep(0.02)
    raise AssertionError("producer did not finish")


@pytest.mark.parametrize("token", [None, "alice-token"])
async def test_reconnect_subscribes_once_and_replays_only_missing_events(
    service, fake_llm, token
):
    entered, release = asyncio.Event(), asyncio.Event()
    fake_llm.json_responses = [plan_json()]

    class Slow(Pages):
        async def search_fragments(self, request, *, mode):
            self.calls.append(request)
            entered.set()
            await release.wait()
            return dict(hits=[], complete=True, total=0)

    client = Slow([])
    kwargs = dict(
        model="m",
        dvd_mcp_client=client,
        token=token,
        user_query="Что в п. 3.3?",
        temperature=0,
        persist_history=False,
    )
    first = stream_document_run(service, owner="alice", **kwargs)
    started = await anext(first)
    request_id = started["content"]["request_id"]
    await asyncio.wait_for(entered.wait(), 2)
    await first.aclose()

    # Two subscribers (as on separate workers) must not start another producer.
    async def read():
        return [
            e
            async for e in stream_document_run(
                service, owner="alice", request_id=request_id, after_event=1, **kwargs
            )
        ]

    a, b = asyncio.create_task(read()), asyncio.create_task(read())
    release.set()
    left, right = await asyncio.gather(a, b)
    assert left == right and all(e["event_id"] > 1 for e in left)
    assert len(client.calls) == 1
    assert left[-1]["content"]["done"] is True
    if token:
        assert (
            await resolve_document_qa_token(
                DocumentQaRequestDTO(
                    request="Продолжи", request_id=request_id, after_event=1
                ),
                token=token,
                dvd_mcp_client=SimpleNamespace(_user_id="alice"),
                dvd_rag_service=service,
            )
            == token
        )
    with pytest.raises(AgentsUnauthorizedException):
        await anext(
            stream_document_run(service, owner="bob", request_id=request_id, **kwargs)
        )


async def test_explicit_cancel_stops_producer_without_done(service, fake_llm):
    entered, stopped = asyncio.Event(), asyncio.Event()
    fake_llm.json_responses = [plan_json()]

    class Slow(Pages):
        async def search_fragments(self, request, *, mode):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    kwargs = dict(
        model="m",
        dvd_mcp_client=Slow([]),
        token=None,
        user_query="Что в п. 3.3?",
        temperature=0,
        persist_history=False,
    )
    stream = stream_document_run(service, **kwargs)
    first = await anext(stream)
    request_id = first["content"]["request_id"]
    await asyncio.wait_for(entered.wait(), 2)
    assert await service.state_store.cancel(request_id)
    rest = [e async for e in stream]
    await asyncio.wait_for(stopped.wait(), 2)
    assert rest[-1]["type"] == "error"
    assert not any(e.get("content", {}).get("done") for e in rest)
    assert (await wait_terminal(service.state_store, request_id))[
        "status"
    ] == "cancelled"


def candidates():
    return [
        dict(
            id=f"article-{n}",
            doc_id="gradcode",
            name=CODE,
            version=EDITION,
            selection_path=[f"статья {n}", "3.3"],
            structure_path=[str(n), "3.3"],
            excerpt=f"Требования статьи {n} к строительству",
            parent_id=f"parent-{n}",
        )
        for n in (49, 52)
    ]


async def test_free_text_edition_preserves_clause_across_real_pipeline_turns(
    service, fake_llm
):
    fake_llm.json_responses = [plan_json(), plan_json()]
    first = await run(
        service,
        Pages([dict(ambiguous=True, candidates=candidates())]),
        "Что указано в п. 3.3 град кодекса?",
    )
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[
            dict(role="user", content="Что указано в п. 3.3 град кодекса?"),
            dict(role="assistant", content=answer_text(first)),
        ]
    )
    client = Pages([dict(ambiguous=True, candidates=candidates())])
    second = await run(
        service,
        client,
        "Градостроительный кодекс Российской Федерации редакция номер 190-ФЗ (ред. от 30.01.2026)",
    )
    assert client.calls[0][1]["pattern"] == "3.3"
    assert client.calls[0][1]["doc_id"] == "gradcode"
    assert client.calls[0][1]["version"] == EDITION
    text = answer_text(second)
    assert text.count(CODE) == 1
    assert "статья 49" in text and "статья 52" in text


class Connected:
    method = "GET"
    url = "http://test/documents/qa/stream"
    client = None
    query_params = {}
    headers = {}

    async def is_disconnected(self):
        return False


async def test_http_disconnect_does_not_cancel_document_work(service, fake_llm):
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    fake_llm.json_responses = [plan_json()]

    class Slow(Pages):
        async def search_fragments(self, request, *, mode):
            entered.set()
            await release.wait()
            finished.set()
            return dict(hits=[], complete=True, total=0)

    query = DocumentQaRequestDTO(request="Что в п. 3.3?", model="m")
    stream = stream_document_qa(Connected(), query, None, Slow([]), service)

    async def consume():
        async for _ in stream:
            pass

    subscriber = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        subscriber.cancel()
        with pytest.raises(asyncio.CancelledError):
            await subscriber
        release.set()
        await asyncio.wait_for(finished.wait(), 2)
    finally:
        release.set()
        if not subscriber.done():
            subscriber.cancel()
