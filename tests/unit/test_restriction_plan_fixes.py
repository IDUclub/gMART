"""Regression tests for the restriction-pipeline fixes made for the benchmark re-run.

Covers:
  * the JSON-repair retry in ``RestrictionPlanBuilder._request_plan`` preserving
    the conversation (invalid answer + fix instruction + user query) instead of
    blindly re-rolling from the system prompt only;
  * the ``objects`` / ``generators`` layer-name translation in
    ``RestrictionParserService._feature_collections``;
  * the ``ABLATION_NO_CATALOG`` env toggle.
"""

from __future__ import annotations

import json

import geopandas as gpd
import pytest
from shapely.geometry import Point

from src.agents.services.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.restriction_context import RestrictionContextBuilder
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
    _ablation_no_catalog,
)
from src.agents.services.service_entities.restriction_plan import (
    BufferRule,
    EntityRef,
    RestrictionPlan,
    RestrictionRule,
    RestrictionTaskMode,
)
from tests.helpers import FakeLlmClient


def _valid_plan_json() -> str:
    return json.dumps(
        {
            "mode": "buffers_only",
            "source_entities": [],
            "target_entities": [],
            "buffer_rules": [],
            "restriction_rules": [],
            "selection_reasons": [],
            "confidence": 1.0,
            "clarification_question": None,
            "original": "test",
        },
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_repair_retry_preserves_conversation():
    """A retry must resend the prior (invalid) answer + fix instruction, not a
    fresh system-prompt-only conversation."""
    fake = FakeLlmClient()
    fake.json_responses = ["{ this is not valid json", _valid_plan_json()]
    builder = RestrictionPlanBuilder(fake)

    plan = await builder._request_plan(
        model="m",
        prompt="SYSTEM PROMPT",
        user_query="покажи жилые дома в 200 м от школ",
    )
    assert plan.mode == RestrictionTaskMode.BUFFERS_ONLY
    assert len(fake.chat_calls) == 2

    # The retry conversation must carry the whole context forward. (The old bug
    # rebuilt it from the system prompt only, dropping all three of these.)
    retry_msgs = fake.chat_calls[1].messages
    roles = [m["role"] for m in retry_msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert "жилые дома" in retry_msgs[1]["content"]  # user query kept
    assert "not valid json" in retry_msgs[2]["content"]  # invalid answer kept
    assert "невалидный" in retry_msgs[3]["content"].lower()  # repair instruction


def test_feature_collections_translate_reserved_names():
    fc = {"type": "FeatureCollection", "features": []}
    layers = {"objects": fc, "generators": fc, "Жилой дом": fc}
    out = {
        ev["content"]["name"]
        for ev in RestrictionParserService._feature_collections(layers)
    }
    assert "objects" not in out and "generators" not in out
    assert "Объекты в зоне ограничений" in out
    assert "Источники ограничений" in out
    assert "Жилой дом" in out  # catalog names pass through unchanged


def test_ablation_env_toggle(monkeypatch):
    monkeypatch.delenv("ABLATION_NO_CATALOG", raising=False)
    assert _ablation_no_catalog() is False
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv("ABLATION_NO_CATALOG", truthy)
        assert _ablation_no_catalog() is True
    monkeypatch.setenv("ABLATION_NO_CATALOG", "0")
    assert _ablation_no_catalog() is False


def test_buffers_only_plan_is_rejected_for_explicit_intersection_request():
    plan = RestrictionPlan(
        mode="buffers_only",
        source_entities=[
            EntityRef(name="Региональная дорога", entity_type="physical_object")
        ],
        buffer_rules=[
            BufferRule(
                source_name="Региональная дорога",
                buffer_size=200,
                title="Зона 200 м",
            )
        ],
        original="test",
    )

    issues = RestrictionPlanBuilder._find_semantic_issues(
        plan,
        "Какие жилые дома попадают в зону? Выведи объекты.",
    )

    assert issues


def test_buffers_only_plan_is_valid_when_only_geometry_is_requested():
    plan = RestrictionPlan(
        mode="buffers_only",
        source_entities=[
            EntityRef(name="Региональная дорога", entity_type="physical_object")
        ],
        buffer_rules=[
            BufferRule(
                source_name="Региональная дорога",
                buffer_size=200,
                title="Зона 200 м",
            )
        ],
        original="test",
    )

    assert not RestrictionPlanBuilder._find_semantic_issues(
        plan,
        "Построй и покажи буферные зоны 200 метров вокруг дорог.",
    )


def test_user_rule_is_rejected_when_canonical_hit_was_retrieved():
    plan = RestrictionPlan(
        mode="restrictions",
        source_entities=[EntityRef(name="Лес", entity_type="physical_object")],
        target_entities=[EntityRef(name="Жилой дом", entity_type="physical_object")],
        buffer_rules=[BufferRule(source_name="Лес", buffer_size=50, title="Правило")],
        restriction_rules=[
            RestrictionRule(
                source_name="Лес",
                target_names=["Жилой дом"],
                title="Правило",
                description="Пересечение",
            )
        ],
        original="test",
    )

    issues = RestrictionPlanBuilder._find_semantic_issues(
        plan,
        "Примени СП 4.13130.2013 пункт 4.14.",
        [{"id": "canonical"}],
    )

    assert issues


def test_missing_target_triggers_orientation_repair():
    plan = RestrictionPlan(
        mode="needs_clarification",
        source_entities=[EntityRef(name="Жилой дом", entity_type="physical_object")],
        target_entities=[],
        buffer_rules=[
            BufferRule(
                source_name="Жилой дом",
                buffer_size=50,
                title="Расстояние до леса",
                origin="normgraph",
                restriction_id="r-1",
            )
        ],
        restriction_rules=[],
        original="test",
    )

    issues = RestrictionPlanBuilder._find_semantic_issues(
        plan,
        "Проверь и выведи жилые дома ближе 50 метров к лесу.",
        [{"id": "r-1"}],
    )

    assert issues


@pytest.mark.asyncio
async def test_object_context_is_truncated_but_keeps_total_count():
    objects = gpd.GeoDataFrame(
        {
            "restriction_name": ["Зона 200 м"] * 12,
            "restriction_description": ["Пересечение"] * 12,
            "object_ref": [
                {"id": f"physical_object/{i}/geometry/{i}", "name": f"Дом #{i}"}
                for i in range(12)
            ],
            "source_layer": ["Жилой дом"] * 12,
            "restriction_evidence": [[{"reason": "Пересечение"}]] * 12,
        },
        geometry=[Point(i, i) for i in range(12)],
        crs=3857,
    )

    context = json.loads(
        await RestrictionContextBuilder.generate_objects_summary(objects)
    )

    assert context["affected_count"] == 12
    assert len(context["affected_objects"]) == 10
    assert context["details_truncated"] is True


@pytest.mark.asyncio
async def test_final_response_has_fallback_when_ollama_returns_empty_content():
    service = RestrictionParserService.__new__(RestrictionParserService)
    service.llm_client = FakeLlmClient()
    context = '{"affected_count": 12, "details_truncated": true}'

    chunks = [
        chunk
        async for chunk in service.generate_final_response("model", "query", context, 0)
    ]

    assert "12 объектов" in "".join(chunk["content"]["text"] for chunk in chunks)
    assert chunks[-1]["content"]["done"] is True


def test_normgraph_grounding_uses_exact_distance_and_provenance():
    plan = RestrictionPlan(
        mode="restrictions",
        source_entities=[EntityRef(name="Дороги", entity_type="physical_object")],
        target_entities=[EntityRef(name="Жилые дома", entity_type="physical_object")],
        buffer_rules=[
            BufferRule(
                source_name="Дороги",
                buffer_size=999,
                title="Правило",
                origin="normgraph",
                restriction_id="r-1",
            )
        ],
        restriction_rules=[
            RestrictionRule(
                source_name="Дороги",
                target_names=["Жилые дома"],
                title="Правило",
                description="Описание",
                origin="normgraph",
                restriction_id="r-1",
            )
        ],
        original="test",
    )
    hit = {
        "id": "r-1",
        "kind": "минимальное_расстояние",
        "value": {"number": 25.5, "unit": "м"},
        "extraction_text": "Не менее 25,5 м",
        "provenance": {
            "doc_id": "doc-1",
            "name": "СП test",
            "version": "2026",
            "clause_node_id": "clause-1",
            "numbering": "5.2",
        },
    }

    grounded = RestrictionPlanBuilder._ground_normgraph_rules(plan, [hit])

    assert grounded.buffer_rules[0].buffer_size == 25.5
    assert grounded.buffer_rules[0].provenance.document_name == "СП test"
    assert grounded.restriction_rules[0].provenance.clause_number == "5.2"
    assert grounded.restriction_rules[0].provenance.extraction_text == "Не менее 25,5 м"


def test_unretrieved_normgraph_rules_cannot_keep_normgraph_origin():
    plan = RestrictionPlan(
        mode="restrictions",
        source_entities=[EntityRef(name="Лес", entity_type="physical_object")],
        target_entities=[EntityRef(name="Жилой дом", entity_type="physical_object")],
        buffer_rules=[
            BufferRule(
                source_name="Лес",
                buffer_size=50,
                title="Правило СП",
                origin="normgraph",
            )
        ],
        restriction_rules=[
            RestrictionRule(
                source_name="Лес",
                target_names=["Жилой дом"],
                title="Правило СП",
                description="Описание",
                origin="normgraph",
            )
        ],
        original="test",
    )

    grounded = RestrictionPlanBuilder._ground_normgraph_rules(plan, [])

    assert grounded.mode == "needs_clarification"
    assert grounded.buffer_rules == []
    assert grounded.restriction_rules == []


def test_grounding_keeps_a_clarification_the_planner_already_produced():
    """A plan that never had rules did not "fail to ground".

    When an entity is missing from the scenario catalog the planner already
    returns needs_clarification with the available objects listed. Grounding used
    to overwrite that with the generic normative message, which both loses the
    catalog listing and misreports why the query was not answered.
    """

    catalog_question = (
        'Запрошенный объект "детские сады" отсутствует в доступных списках. '
        "Доступные объекты: сервисы: ['аптека', 'остановка наземного транспорта']"
    )
    plan = RestrictionPlan(
        mode="needs_clarification",
        source_entities=[],
        target_entities=[],
        buffer_rules=[],
        restriction_rules=[],
        clarification_question=catalog_question,
        original="test",
    )

    grounded = RestrictionPlanBuilder._ground_normgraph_rules(plan, [])

    assert grounded.mode == "needs_clarification"
    assert grounded.clarification_question == catalog_question
