"""A hit cut mid-sentence by an older IDU_DVD parse is completed from its next fragment."""

from src.agents.services.dvd.fragment_continuation import (
    complete_cut_fragments,
    continues,
    is_cut,
)
from tests.helpers import FakeDvdMcpClient, plan_json, statuses, verdict_json
from tests.unit.test_dvd_rag_service import _run

# СП 257.1325800.2020 as stored on dev: one source paragraph, two fragments.
ROOF = {
    "id": "roof",
    "next_id": "code",
    "name": "СП 257.1325800.2020",
    "numbering": "6.1.11",
    "kind": "text",
    "text": "6.1.11 Эксплуатируемые кровли гостиниц следует проектировать с учетом СП",
    "char_start": 23839,
    "char_end": 23938,
}
CODE = {
    "id": "code",
    "next_id": "lifts",
    "name": "СП 257.1325800.2020",
    "numbering": "6.1.15",
    "kind": "text",
    "text": "17.13330 и СП 160.1325800.",
    "char_start": 23839,
    "char_end": 23938,
}
LIFTS = {
    "id": "lifts",
    "next_id": None,
    "text": "6.1.12 Пассажирские лифты предусматриваются в соответствии с СП 118.13330.",
    "char_start": 23939,
    "char_end": 24320,
}


class NodeClient(FakeDvdMcpClient):
    def __init__(self, nodes, **kwargs):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.node_calls = []

    async def get_node(self, node_id):
        self.node_calls.append(node_id)
        node = self.nodes[node_id]
        if isinstance(node, Exception):
            raise node
        return {**node, "next": self.nodes.get(node.get("next_id"))}


def test_only_text_that_ends_mid_sentence_is_cut():
    assert is_cut(ROOF)
    assert not is_cut(CODE)
    assert not is_cut({**ROOF, "next_id": None})
    assert not is_cut({**ROOF, "table_html": "<table/>"})
    assert not is_cut({**ROOF, "user_id": "u1"})


def test_continuation_is_proven_by_source_span_code_or_lowercase():
    unshared = {**CODE, "char_start": ROOF["char_end"]}
    assert continues(ROOF, CODE)
    assert continues(ROOF, unshared)
    assert continues({"text": "с учетом требований"}, {"text": "к кровлям."})
    heading = {"text": "6.1 Требования к зданиям гостиниц", "char_end": 10}
    clause = {"text": "6.1.1 При проектировании гостиниц", "char_start": 11}
    assert not continues(heading, clause)
    assert not continues({"text": "с учетом"}, {"text": "Таблица 1"})


async def test_cut_hit_takes_the_rest_of_its_sentence_under_its_label():
    client = NodeClient({"roof": ROOF, "code": CODE, "lifts": LIFTS})
    hits = await complete_cut_fragments(client, [ROOF, CODE, LIFTS])
    assert [h["id"] for h in hits] == ["roof", "lifts"]
    assert hits[0]["text"] == (
        "6.1.11 Эксплуатируемые кровли гостиниц следует проектировать с учетом СП "
        "17.13330 и СП 160.1325800."
    )
    assert hits[0]["numbering"] == "6.1.11" and hits[0]["continued_by"] == ["code"]
    # The sentence ended: the next clause is not fetched, the input is not modified.
    assert client.node_calls == ["roof"]
    assert ROOF["text"].endswith("с учетом СП")


async def test_failed_lookup_keeps_the_hit():
    client = NodeClient({"roof": RuntimeError("down")})
    assert await complete_cut_fragments(client, [ROOF]) == [ROOF]


async def test_complete_hits_need_no_lookup():
    client = NodeClient({})
    hits = [CODE, LIFTS]
    assert await complete_cut_fragments(client, hits) is hits
    assert client.node_calls == []


async def test_answer_is_drafted_from_the_completed_requirement(service, fake_llm):
    client = NodeClient(
        {"roof": ROOF, "code": CODE, "lifts": LIFTS}, hits_per_call=[[ROOF]]
    )
    fake_llm.json_responses = [plan_json(), verdict_json()]
    fake_llm.answer_texts = ["Кровли проектируют с учетом СП 17.13330 [1]."]
    events = await _run(service, client)
    draft = next(call for call in fake_llm.chat_calls if call.stream)
    prompt = "\n".join(str(m["content"]) for m in draft.messages)
    assert "с учетом СП 17.13330 и СП 160.1325800." in prompt
    assert any("оборванные на полуслове: 1" in s for s in statuses(events))
