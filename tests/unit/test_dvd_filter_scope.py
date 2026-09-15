from types import SimpleNamespace

from src.agents.services.dvd.dialogue import (
    pending_question,
    render_question,
    resolve_reply,
)
from src.agents.services.dvd.dvd_reasoning import RetrievalPlanner
from src.agents.services.dvd.retrieval_scope import apply_scope
from src.agents.services.service_entities.dvd_plan import (
    SemanticRetrievalPlan,
    StructureRetrievalPlan,
)
from tests.helpers import answer_text, plan_json, verdict_json
from tests.unit.test_dvd_structured_retrieval import Pages, run

NAME = "СП 55.13330.2016"


def sp_candidates():
    return [
        dict(
            id=str(i),
            doc_id="sp55",
            name=NAME,
            version="2016",
            numbering="3",
            selection_path=[p],
            excerpt="Название",
            hierarchy=[dict(id=str(i), type=t, numbering="3", name=n)],
        )
        for i, p, t, n in [
            (1, "3", "clause", "Подготовлен к утверждению"),
            (2, "раздел 3", "section", "Термины и определения"),
        ]
    ]


async def test_sp55_followup_uses_same_document_and_new_clause(service, fake_llm):
    fake_llm.json_responses = [plan_json()]
    first = await run(
        service,
        Pages([dict(ambiguous=True, candidates=sp_candidates())]),
        "Что написано в СП 55 пункт 3?",
    )
    service.get_chat_messages.return_value = SimpleNamespace(
        messages=[
            dict(role="user", content="Что написано в СП 55 пункт 3?"),
            dict(role="assistant", content=answer_text(first)),
        ]
    )
    client = Pages(
        [
            dict(
                hits=[
                    dict(
                        id="33",
                        doc_id="sp55",
                        name=NAME,
                        version="2016",
                        numbering="3.3",
                        text="Определение блокированной застройки.",
                        matched=True,
                    )
                ],
                total=1,
                complete=True,
            )
        ]
    )
    fake_llm.json_responses = [verdict_json(satisfied=True)]
    fake_llm.answer_texts = ["Определение блокированной застройки [1]."]
    await run(service, client, "Есть в сп пункт 3.3?")
    mode, request = client.calls[0]
    assert mode == "structure"
    assert request["doc_id"] == "sp55" and request["pattern"] == "3.3"
    assert request["document_names"] == [NAME] and request["version"] == "2016"


async def test_scope_survives_answer_and_scopes_semantic_followup(service, fake_llm):
    await service.state_store.set_document_scope(
        "chat-1", dict(doc_id="sp55", document_names=[NAME], version="2016")
    )
    fake_llm.json_responses = [
        plan_json(document_names=None),
        verdict_json(satisfied=True),
    ]
    fake_llm.answer_texts = ["Требования приведены в тексте [1]."]
    client = Pages(
        [
            dict(
                hits=[
                    dict(
                        id="x",
                        doc_id="sp55",
                        name=NAME,
                        version="2016",
                        text="Требования приведены в тексте.",
                    )
                ],
                total=1,
                complete=True,
            )
        ]
    )
    await run(service, client, "Какие в нём требования к эвакуации?")
    mode, request = client.calls[0]
    assert mode == "filtered" and request["rank_by_relevance"]
    assert request["doc_id"] == "sp55"
    assert "pattern" not in request


def test_section_topic_and_full_quote_have_different_retrieval_modes():
    initial = StructureRetrievalPlan(
        retrieval_mode="structure", pattern="раздел 5", rank_by_relevance=True
    )
    topic = RetrievalPlanner._clamp(
        initial, "В разделе 5 СП 55 найди требования к эвакуации"
    )
    quote = RetrievalPlanner._clamp(initial, "Процитируй раздел 5 СП 55")
    assert topic.rank_by_relevance
    assert not quote.rank_by_relevance
    assert topic.pattern == quote.pattern == "раздел 5"


def test_new_document_and_global_query_release_old_scope():
    plan = SemanticRetrievalPlan(retrieval_mode="semantic", document_names=["СП 309"])
    scoped = apply_scope(
        plan, "Требования в СП 309", dict(doc_id="sp55", version="2016")
    )
    assert scoped.document_names == ["СП 309"] and scoped.doc_id is None
    global_plan = apply_scope(plan, "Поищи во всех документах", dict(doc_id="sp55"))
    assert global_plan.doc_id is None and global_plan.document_names is None


def test_named_hierarchy_has_unique_parents_and_selectable_leaves():
    parent = dict(id="s3", type="section", numbering="3", name="Термины и определения")
    cs = [
        dict(
            id=f"p{i}",
            doc_id="sp55",
            name=NAME,
            version="2016",
            numbering=f"3.{i}",
            selection_path=["раздел 3", f"3.{i}"],
            hierarchy=[
                parent,
                dict(
                    id=f"p{i}",
                    type="definition",
                    numbering=f"3.{i}",
                    name=f"Термин {i}",
                ),
            ],
        )
        for i in (3, 4)
    ]
    pending = pending_question(
        StructureRetrievalPlan(retrieval_mode="structure", pattern="3.*"),
        cs + [cs[0]],
        "термины",
    )
    text = render_question(pending)
    assert text.count(NAME) == 1
    assert text.count("Термины и определения") == 1
    assert text.count("Вариант ") == 2
    assert resolve_reply("2", pending)["selected_ids"] == ["p4"]


def test_document_selection_retains_topic_instead_of_inventing_structure():
    plan = SemanticRetrievalPlan(
        retrieval_mode="semantic", search_query="эвакуация", document_names=["СП 55"]
    )
    cs = [
        dict(
            id=str(i),
            doc_id=f"d{i}",
            name=f"СП 55.13330.{i}",
            version=str(i),
            entity_kind="document",
            structure_path=[],
            selection_path=[],
        )
        for i in (2016, 2026)
    ]
    reply = resolve_reply("2", pending_question(plan, cs, "эвакуация"))
    assert reply["plan"]["doc_id"] == "d2026"
    assert reply["plan"]["search_query"] == "эвакуация"
    assert reply["plan"]["retrieval_mode"] == "semantic"
    assert reply["plan"]["pattern"] is None


def test_interleaved_relevance_does_not_repeat_section_heading():
    cs = []
    for i, section, excerpt in [
        (1, "A", "эвакуация безопасность"),
        (2, "B", "эвакуация"),
        (3, "A", "прочее"),
    ]:
        cs.append(
            dict(
                id=f"p{i}",
                doc_id="d",
                name=NAME,
                version="2016",
                excerpt=excerpt,
                selection_path=[f"раздел {section}", str(i)],
                hierarchy=[
                    dict(
                        id=section,
                        type="section",
                        numbering=section,
                        name=f"Название {section}",
                    ),
                    dict(id=f"p{i}", type="clause", numbering=str(i), name=excerpt),
                ],
            )
        )
    pending = pending_question(
        StructureRetrievalPlan(retrieval_mode="structure", pattern="3"),
        cs,
        "эвакуация безопасность",
    )
    text = render_question(pending)
    assert text.count("Название A") == 1 and text.count("Название B") == 1
    assert [o["members"][0]["id"] for o in pending["options"]] == ["p1", "p3", "p2"]


async def test_not_found_new_document_does_not_restore_previous_document(
    service, fake_llm
):
    await service.state_store.set_document_scope(
        "chat-1", dict(doc_id="old", document_names=[NAME])
    )
    fake_llm.json_responses = [plan_json()]
    await run(
        service, Pages([dict(hits=[], total=0, complete=True)]), "Пункт 3.3 СП 999"
    )
    scope = await service.state_store.get_document_scope("chat-1")
    assert scope["document_names"] == ["СП 999"] and not scope.get("doc_id")
    fake_llm.json_responses = [plan_json()]
    client = Pages([dict(hits=[], total=0, complete=True)])
    await run(service, client, "Есть в нём пункт 4?")
    assert client.calls[0][1]["document_names"] == ["СП 999"]
