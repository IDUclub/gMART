import json

import pytest

from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import GoalManager
from src.agents.services.orchestrator.analysis_grounding import grounding_payload
from tests.unit.test_analytical_orchestrator import LAYER, table


def evidence():
    context = AnalysisContext()
    aid = context.add_artifact(table(80), 1, "calc")
    context.add_artifact(LAYER, 1, "calc")
    context.add_artifact(
        {"type": "analysis_text", "content": {"text": "Обеспеченность 100%."}},
        1,
        "calc",
    )
    context.finish(1, "Расчёт", 17, "completed", "Обеспеченность 100%.", "calc")
    return context, aid


async def test_grounding_rejects_false_claim_using_primary_table(fake_llm):
    context, aid = evidence()
    fake_llm.json_responses = [
        json.dumps(
            {
                "issues": [
                    {
                        "quote": "Обеспеченность 100%.",
                        "reason": "В таблице 80%.",
                        "evidence_ids": [aid],
                    }
                ]
            }
        )
    ]
    with pytest.raises(ValueError, match="В таблице 80%"):
        await GoalManager(fake_llm).validate_answer(
            "m", "Оцени обеспеченность", "Обеспеченность 100%.", context
        )


@pytest.mark.parametrize(
    "quote, refs",
    [
        ("Другая цитата", ["calc:a1"]),
        ("Обеспеченность 80%.", ["unknown"]),
    ],
)
async def test_invalid_grounding_feedback_is_repaired(fake_llm, quote, refs):
    context, _ = evidence()
    fake_llm.json_responses = [
        json.dumps(
            {"issues": [{"quote": quote, "reason": "Ошибка", "evidence_ids": refs}]}
        ),
        '{"issues": []}',
    ]
    await GoalManager(fake_llm).validate_answer(
        "m", "Оцени обеспеченность", "Обеспеченность 80%.", context
    )
    assert not fake_llm.json_responses


def test_grounding_has_full_layer_manifest_but_no_specialist_prose():
    context, _ = evidence()
    for i in range(100):
        context.add_artifact(LAYER, 1, f"layers-{i}")
        context.finish(1, "Слой", 17, "completed", "Неверный вывод", f"layers-{i}")
    payload = grounding_payload("Запрос", "Ответ", context)
    assert len(payload["manifest"]) == 102
    assert all(a["scenario_id"] == 17 for a in payload["manifest"])
    assert "100%" not in json.dumps(payload, ensure_ascii=False)
    assert "Неверный вывод" not in json.dumps(payload, ensure_ascii=False)
    assert all(a["kind"] != "analysis_text" for a in payload["manifest"])


def test_compact_metric_table_keeps_deficit_visible_to_synthesis_and_verifier():
    from src.agents.services.provision.provision_context import ProvisionContextBuilder

    context = AnalysisContext()
    content = ProvisionContextBuilder().build_provision_metrics_table(
        {
            "services_count": 2,
            "total_capacity": 400,
            "total_demand": 550,
            "unsatisfied_demand": 150,
            "deficit": 150,
            "surplus": 0,
        },
        "Школа",
    )
    aid = context.add_artifact({"type": "table", "content": content}, 1, "calc")
    context.finish(1, "Расчёт", 17, "completed", "", "calc")
    preview = next(
        x for x in context.view()["selected_evidence"] if x["artifact_id"] == aid
    )
    assert preview["rows"] == content["rows"]


def test_catalogue_inspection_is_metadata_not_a_source_artifact():
    from src.agents.services.service_entities.orchestrator_plan import ArtifactSlice

    context, _ = evidence()
    context.inspect([ArtifactSlice(artifact_id="_catalog", offset=0, limit=3)])
    payload = grounding_payload("Запрос", "Ответ", context)
    assert all(item["artifact_id"] != "_catalog" for item in payload["facts"])
