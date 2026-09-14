"""Failures reduced from the industrial dialogue traces, without live inference."""

import json

import pytest

from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import (
    AnalysisGoal,
    GoalDecision,
    GoalManager,
    GoalState,
)
from src.agents.services.scenario_data.scenario_data_read import scoped_tools
from tests.unit.test_scenario_data_read import make_tool


def test_functional_zone_request_keeps_geometry_source_in_catalogue():
    zones = make_tool("GetScenarioFunctionalZones", "projects")
    card = make_tool("GetProjectById", "projects")
    assert scoped_tools(
        [card, zones], "Получить слой функциональных зон проекта 91001"
    ) == [zones]


@pytest.mark.parametrize("agent", ["documents", "norms", "compliance"])
def test_delegation_preserves_document_scope_when_controller_shortens_task(agent):
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Проверить только указанную норму",
            "requirements": [
                {
                    "id": "norm",
                    "agent": agent,
                    "scenario_id": 91001,
                    "description": "Получить пункт 1.1 документа LOCAL SDK TEST, версия 2026",
                    "source_quote": "Используй только LOCAL SDK TEST, версия 2026, пункт 1.1",
                    "required_artifacts": ["analysis_text"],
                }
            ],
        }
    )
    step = (
        GoalState(AnalysisContext(), goal)
        .validate_decision(
            GoalDecision(
                action="continue", requirement_id="norm", task="Проверь норму"
            ),
            {agent},
        )
        .steps[0]
    )
    assert "LOCAL SDK TEST" in step.task
    assert "2026" in step.task
    assert "1.1" in step.task


def test_multi_variant_goal_can_hold_all_atomic_results():
    goal = AnalysisGoal.model_validate(
        {
            "objective": "Три варианта с независимыми результатами",
            "requirements": [
                {
                    "id": f"r{i}",
                    "agent": "scenario_data",
                    "scenario_id": i + 1,
                    "description": "Получить результат",
                    "source_quote": "Все варианты",
                    "required_artifacts": ["table"],
                }
                for i in range(15)
            ],
        }
    )
    assert len(goal.requirements) == 15


async def test_goal_rejects_combined_type_and_repairs_to_atomic_selections(fake_llm):
    draft = {
        "objective": "Получить услуги",
        "requirements": [
            {
                "id": "services",
                "agent": "scenario_data",
                "scenario_id": 17,
                "description": "Получить школы и детские сады",
                "source_ids": [1],
                "subject": "Школа, Детский сад",
                "entity_kind": "services",
                "required_artifacts": ["table", "feature_collection"],
            }
        ],
    }
    repaired = {
        "objective": draft["objective"],
        "requirements": [
            {**draft["requirements"][0], "id": f"r{i}", "subject": subject}
            for i, subject in enumerate(["Школа", "Детский сад"])
        ],
    }
    fake_llm.json_responses = [json.dumps(draft), json.dumps(repaired)]
    goal = await GoalManager(fake_llm).create(
        "m", "Получи школы и детские сады.", [], 17
    )
    assert [r.subject for r in goal.requirements] == ["Школа", "Детский сад"]


def test_synthesis_view_includes_source_proof_without_manual_inspection():
    context = AnalysisContext()
    aid = context.add_artifact(
        {
            "type": "source_evidence",
            "content": {
                "source": "dvd",
                "document_name": "Example",
                "version": "2026",
                "text": "Minimum 50 m",
            },
        },
        1,
        "r",
    )
    context.finish(1, "Read source", None, "completed", "Read", "r")
    view = context.view()
    assert any(p.get("artifact_id") == aid for p in view["selected_evidence"])


def test_calculation_evidence_retains_scenario_and_rows_in_crowded_context():
    context = AnalysisContext()
    aid = context.add_artifact(
        {
            "type": "table",
            "content": {
                "name": "provision_summary",
                "title": "Обеспеченность",
                "columns": [{"key": "deficit", "label": "Дефицит"}],
                "rows": [{"deficit": 75}],
            },
        },
        1,
        "calculation",
    )
    context.finish(1, "Расчёт", 17, "completed", "Расчёт готов", "calculation")
    for i in range(35):
        context.add_artifact(
            {
                "type": "analysis_text",
                "content": {"text": "Ready", "title": f"Other {i}"},
            },
            2,
            "later",
        )
    context.finish(2, "Другая работа", 18, "completed", "Готово", "later")
    preview = next(
        p for p in context.view()["selected_evidence"] if p.get("artifact_id") == aid
    )
    assert preview["scenario_id"] == 17
    assert preview["rows"][0]["deficit"] == 75


async def test_followup_can_cite_prior_service_and_layer_conditions(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Повторный расчёт",
                "requirements": [
                    {
                        "id": "p",
                        "agent": "provision",
                        "description": "Рассчитать обеспеченность школами",
                        "source_ids": [1, 2],
                        "required_artifacts": ["table"],
                    }
                ],
            }
        )
    ]
    goal = await GoalManager(fake_llm).create(
        "m",
        "Теперь сценарий 18.",
        [],
        18,
        [
            {
                "role": "user",
                "content": "Рассчитай обеспеченность школами и верни расчётные слои.",
            }
        ],
    )
    assert "школами" in goal.requirements[0].source_quote
    assert "feature_collection" in goal.requirements[0].required_artifacts


async def test_zone_layer_resolves_required_source_and_year_from_catalogue(
    monkeypatch, fake_llm, fake_urban, state_store
):
    from unittest.mock import AsyncMock

    from src.agents.services.scenario_data.scenario_data_service import (
        ScenarioDataService,
    )
    from tests.unit.test_scenario_data_read import read_plan

    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **k: fake_llm,
    )
    source = make_tool(
        "GetScenarioFunctionalZoneSources",
        "projects",
        {"scenario_id": {"type": "integer"}},
    )
    zones = make_tool(
        "GetScenarioFunctionalZones",
        "projects",
        {
            "scenario_id": {"type": "integer"},
            "source": {"type": "string"},
            "year": {"type": "integer"},
        },
    )
    zones.input_schema["required"] = ["scenario_id", "source", "year"]
    fake_llm.json_responses = [
        read_plan(
            zones, {"scenario_id": 17, "source": "survey", "year": 2024}
        ).model_dump_json()
    ]
    mcp = AsyncMock()
    mcp.load_tools.return_value = [source, zones]
    layer = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [30, 60]},
                "properties": {"source": "survey", "year": 2024},
            }
        ],
    }
    mcp.execute_tool.side_effect = [[{"source": "survey", "year": 2024}], layer]
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    events = [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "t",
            "m",
            0,
            "Покажи функциональные зоны сценария 17",
            scenario_id=17,
            persist_history=False,
        )
    ]
    assert [c.args[1] for c in mcp.execute_tool.await_args_list] == [
        source.name,
        zones.name,
    ]
    assert any(
        e["type"] == "feature_collection"
        and e["content"]["feature_collection"] == layer
        for e in events
    )


async def test_goal_creation_does_not_copy_large_assistant_history(fake_llm):
    from src.agents.runtime.budget import RunBudget, budget_scope

    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Повторить расчёт",
                "requirements": [
                    {
                        "id": "p",
                        "agent": "provision",
                        "source_ids": [1],
                        "description": "Обеспеченность школами",
                        "required_artifacts": ["table"],
                    }
                ],
            }
        )
    ]
    with budget_scope(RunBudget()):
        result = await GoalManager(fake_llm).create(
            "m",
            "Рассчитай обеспеченность школами.",
            [],
            17,
            [{"role": "assistant", "content": "large saved artifact " * 4000}],
        )
    assert result.requirements[0].agent == "provision"


async def test_model_combined_physical_selection_is_split_without_extra_read(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Исходные объекты",
                "requirements": [
                    {
                        "id": "physical",
                        "agent": "scenario_data",
                        "source_ids": [1],
                        "description": "Получить физические объекты типов «Жилой дом» и «Парк».",
                        "required_artifacts": ["table", "feature_collection"],
                    }
                ],
            }
        )
    ]
    result = await GoalManager(fake_llm).create(
        "m", "Покажи физические объекты типов «Жилой дом» и «Парк».", [], 17
    )
    assert [(r.entity_kind, r.subject) for r in result.requirements] == [
        ("physical_objects", "Жилой дом"),
        ("physical_objects", "Парк"),
    ]


def test_provision_comparison_uses_real_scoped_cells_and_excludes_other_scenarios():
    context = AnalysisContext()
    for sid, deficit in [(17, 700), (18, 0), (19, 999)]:
        context.add_artifact(
            {
                "type": "table",
                "content": {
                    "name": "provision_summary",
                    "title": "Обеспеченность",
                    "columns": [
                        {"key": "service", "label": "Услуга"},
                        {"key": "deficit", "label": "Дефицит"},
                    ],
                    "rows": [{"service": "Школа", "deficit": deficit}],
                },
            },
            1,
            str(sid),
        )
        context.finish(1, "Расчёт", sid, "completed", "Расчёт", str(sid))
    specs = context.provision_comparisons("Сравни дефициты сценариев 17 и 18")
    result = context.compare(specs)
    assert len(result["content"]["rows"]) == 1
    row = result["content"]["rows"][0]
    assert (row["before"], row["after"], row["delta"]) == ("700", "0", "-700")
    assert context.provision_comparisons("Рассчитай сценарий 18") == []


def test_draft_calculation_domain_cannot_be_confused_with_retrieval_kind():
    from src.agents.services.orchestrator.analysis_goal import GoalDraftRequirement

    r = GoalDraftRequirement.model_validate(
        {
            "id": "p",
            "agent": "provision",
            "subject": "Школа",
            "entity_kind": "services",
            "description": "Рассчитать школы",
            "source_ids": [1],
            "required_artifacts": ["table"],
        }
    )
    assert r.agent == "provision" and r.entity_kind == "other"


async def test_russian_request_cannot_dispatch_an_english_indicator_task(fake_llm):
    draft = {
        "objective": "Получить население",
        "requirements": [
            {
                "id": "population",
                "agent": "scenario_data",
                "source_ids": [1],
                "description": "Retrieve the population indicator for scenario 17",
                "required_artifacts": ["table"],
            }
        ],
    }
    repaired = {
        **draft,
        "requirements": [
            {
                **draft["requirements"][0],
                "description": "Получи сохранённый показатель численности населения для сценария 17",
            }
        ],
    }
    fake_llm.json_responses = [json.dumps(draft), json.dumps(repaired)]
    goal = await GoalManager(fake_llm).create(
        "m", "Приведи сохранённую численность населения.", [], 17
    )
    assert "показатель" in goal.requirements[0].description


async def test_scoped_compliance_does_not_audit_other_documents_or_versions():
    from src.agents.services.normgraph.normgraph_restriction_retriever import (
        NormGraphRestrictionRetriever,
    )
    from tests.unit.test_normgraph_restriction_retriever import FakeClient

    hits = [
        {
            "id": rid,
            "kind": "минимальное_расстояние",
            "value": {"number": 50, "unit": "м"},
            "provenance": {"name": name, "numbering": "1.1", "version": year},
        }
        for rid, name, year in [
            ("wanted", "EXAMPLE", "2026"),
            ("old", "EXAMPLE", "2025"),
            ("other", "OTHER", "2026"),
        ]
    ]
    result = await NormGraphRestrictionRetriever(None).retrieve(
        FakeClient(hits),
        "m",
        "Проверь пункт 1.1 документа EXAMPLE, версия 2026: минимум 50 м.",
        retrieve_all=True,
        retain_unsupported=True,
    )
    assert [h["id"] for h in result.restrictions] == ["wanted"]


async def test_quoted_canonical_distance_uses_saved_compliance_plan(state_store):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from src.agents.services.restriction.restriction_parser_service import (
        RestrictionParserService,
    )

    service = object.__new__(RestrictionParserService)
    service.state_store = state_store
    service.compliance_result_harness = SimpleNamespace(
        prepare_follow_up=lambda *a, **k: None
    )
    hit = {
        "id": "r1",
        "kind": "минимальное_расстояние",
        "value": {"number": 50, "unit": "м"},
        "provenance": {"name": "EXAMPLE", "version": "2026", "numbering": "1.1"},
    }
    service.normgraph_retriever = SimpleNamespace(
        retrieve=AsyncMock(
            return_value=SimpleNamespace(
                restrictions=[hit], unsupported_count=0, tool_call={}
            )
        )
    )
    service._build_plan = AsyncMock(
        side_effect=AssertionError(
            "Canonical clause was routed into the free-form planner"
        )
    )

    async def canonical(**kwargs):
        yield {"type": "compliance_summary", "content": {"total_norms": 1}}

    service._run_executable_compliance = canonical
    events = [
        e
        async for e in service._run_restriction_execution_pipline(
            mcp_client=object(),
            temperature=0,
            model="m",
            user_query="Проверь пункт 1.1 документа EXAMPLE, версия 2026: расстояние не менее 50 м.",
            scenario_id=17,
            token_ref=["t"],
            persist_history=False,
            normgraph_mcp_client=object(),
            history_agent="compliance",
        )
    ]
    assert any(e["type"] == "compliance_summary" for e in events)
    service._build_plan.assert_not_awaited()


@pytest.mark.parametrize(
    "query,temporary",
    [
        ("Проверь пункт 1.1 документа EXAMPLE, версия 2026: не менее 50 м.", False),
        ("Проверь пункт 1.1 документа EXAMPLE, версия 2026: не менее 0.05 км.", False),
        (
            "Проверь пункт 1.1 документа EXAMPLE, версия 2026: вместо 50 м используй 70 м.",
            True,
        ),
        ("Проверь пункт 1.1 документа EXAMPLE, версия 2026: не менее 70 м.", True),
        ("Проверь расстояние 50 м между школами и стоянками.", True),
        ("Проверь пункт 1.1 документа OTHER, версия 2026: не менее 50 м.", True),
        ("Проверь пункт 1.1 документа EXAMPLE, версия 2025: не менее 50 м.", True),
    ],
)
def test_canonical_quote_does_not_swallow_temporary_user_conditions(query, temporary):
    from src.agents.services.normgraph.normgraph_restriction_retriever import (
        NormGraphRestrictionRetriever,
    )

    hit = {
        "id": "r1",
        "kind": "минимальное_расстояние",
        "value": {"number": 50, "unit": "м"},
        "provenance": {"name": "EXAMPLE", "numbering": "1.1", "version": "2026"},
    }
    assert (
        NormGraphRestrictionRetriever.requires_temporary_distance(query, [hit])
        == temporary
    )


async def test_exhaustive_scoped_retrieval_retains_canonical_value_for_override():
    from src.agents.services.normgraph.normgraph_restriction_retriever import (
        NormGraphRestrictionRetriever,
    )
    from tests.unit.test_normgraph_restriction_retriever import FakeClient

    hit = {
        "id": "r1",
        "kind": "минимальное_расстояние",
        "value": {"number": 50, "unit": "м"},
        "provenance": {"name": "EXAMPLE", "numbering": "1.1", "version": "2026"},
    }
    result = await NormGraphRestrictionRetriever(None).retrieve(
        FakeClient([hit]),
        "m",
        "В пункте 1.1 документа EXAMPLE, версия 2026, вместо 50 м используй 70 м.",
        retrieve_all=True,
    )
    assert result.restrictions == [hit]


def test_multiword_document_name_in_canonical_scope():
    from src.agents.services.normgraph.normgraph_restriction_retriever import (
        NormGraphRestrictionRetriever,
    )

    hit = {
        "id": "r1",
        "kind": "минимальное_расстояние",
        "value": {"number": 50, "unit": "м"},
        "provenance": {
            "name": "EXAMPLE TEST NORM",
            "numbering": "1.1",
            "version": "2026",
        },
    }
    query = "Используй только пункт 1.1 синтетического документа EXAMPLE TEST NORM, версия 2026: 50 м."
    assert NormGraphRestrictionRetriever._filter_explicit_references([hit], query) == [
        hit
    ]
    assert not NormGraphRestrictionRetriever.requires_temporary_distance(query, [hit])
    assert not NormGraphRestrictionRetriever.requires_temporary_distance(
        query + " Анализируй без создания или изменения объектов.", [hit]
    )


async def test_source_quote_supplies_entity_kind_when_description_omits_it(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Получить исходные объекты",
                "requirements": [
                    {
                        "id": "objects",
                        "agent": "scenario_data",
                        "entity_kind": "other",
                        "subject": "Жилой дом и Парк",
                        "source_ids": [1],
                        "description": "Исходные таблицы и слои выбранного сценария для типов «Жилой дом» и «Парк»",
                        "required_artifacts": ["table", "feature_collection"],
                    }
                ],
            }
        )
    ]
    goal = await GoalManager(fake_llm).create(
        "m",
        "Для физических объектов типов «Жилой дом» и «Парк» нужны только исходные таблицы и слои выбранного сценария.",
        [],
        17,
    )
    assert [(r.entity_kind, r.subject) for r in goal.requirements] == [
        ("physical_objects", "Жилой дом"),
        ("physical_objects", "Парк"),
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        {"mode": "needs_clarification"},
        {"mode": "provision", "service_name": None},
        {"mode": "effects", "service_name": None},
    ],
)
async def test_model_cannot_turn_an_empty_clarification_into_a_user_blocker(
    fake_llm, invalid
):
    from src.agents.services.provision.provision_plan_builder import (
        ProvisionPlanBuilder,
    )

    fake_llm.json_responses = [
        json.dumps(invalid),
        json.dumps(
            {"mode": "provision", "service_name": "Школа", "target_population": 12000}
        ),
    ]
    plan = await ProvisionPlanBuilder(fake_llm).build_plan(
        "m",
        "Расчёт обеспеченности и слои для услуги типа «Школа», население 12000.",
        ["Школа", "Детский сад"],
    )
    assert plan.mode == "provision"
    assert plan.service_name == "Школа"


async def test_ready_synthesis_cannot_restart_a_satisfied_requirement(fake_llm):
    fake_llm.json_responses = [
        json.dumps(
            {"action": "continue", "requirement_id": "done", "task": "Повторить"}
        ),
        json.dumps(
            {
                "action": "complete",
                "answer": "Обнаружено одно нарушение.",
                "evidence_ids": ["e1"],
            }
        ),
    ]
    view = {
        "goal": {
            "requirements": [
                {"id": "done", "status": "satisfied", "entity_kind": "other"}
            ]
        },
        "selected_evidence": [],
    }
    decision = await GoalManager(fake_llm).review("m", "Дай вывод", [], view, [], {})
    assert decision.action == "complete"


def test_large_compliance_geometry_cannot_hide_the_verdict_from_synthesis():
    context = AnalysisContext()
    result = {
        "restriction_id": "norm-1",
        "verification_status": "complete",
        "compliance_status": "violated",
        "coverage": {"checked_objects": 1, "unchecked_objects": 0},
        "summary": {"violated_objects": 1, "passed_objects": 0},
        "source": {"document_name": "TEST", "version": "2026", "clause_number": "1.1"},
        "violated_features": {
            "type": "FeatureCollection",
            "features": [{"geometry": {"coordinates": [[30, 60]] * 5000}}],
        },
    }
    aid = context.add_artifact(
        {
            "type": "compliance_summary",
            "content": {"total_norms": 1, "violated_norms": 1, "results": [result]},
        },
        1,
        "check",
    )
    context.finish(1, "Проверка", 17, "completed", "Выполнено", "check")
    for i in range(8):
        context.add_artifact(
            {
                "type": "source_evidence",
                "content": {
                    "system": "documents",
                    "sources": [{"text": "Документ " * 150, "id": str(i)}],
                },
            },
            1,
            "doc" + str(i),
        )
        context.finish(1, "Источник", 17, "completed", "Документ", "doc" + str(i))
    view = context.view(9000)
    assert len(json.dumps(view, ensure_ascii=False).encode()) <= 9000
    proof = next(p for p in view["selected_evidence"] if p["artifact_id"] == aid)
    assert proof["scenario_id"] == 17
    assert proof["content"]["violated_norms"] == 1
    assert proof["content"]["results"][0]["summary"]["violated_objects"] == 1
    assert "violated_features" not in proof["content"]["results"][0]
    assert (
        context.get(aid)["content"]["results"][0]["violated_features"]
        == result["violated_features"]
    )


def test_comparison_covers_individual_and_combined_provision_tables():
    from src.agents.services.provision.provision_context import ProvisionContextBuilder

    context = AnalysisContext()
    builder = ProvisionContextBuilder()
    for service in ("Кружок", "Секция"):
        request = "before-" + service
        context.add_artifact(
            {
                "type": "table",
                "content": builder.build_provision_metrics_table(
                    {"deficit": 40}, service
                ),
            },
            1,
            request,
        )
        context.finish(1, "Расчёт", 17, "completed", "Расчёт", request)
    context.add_artifact(
        {
            "type": "table",
            "content": builder.build_summary_table(
                {
                    "services": {
                        str(i): {"name": service, "summary": {"deficit": 10}}
                        for i, service in enumerate(("Кружок", "Секция"))
                    }
                }
            ),
        },
        1,
        "after",
    )
    context.finish(1, "Расчёт", 18, "completed", "Расчёт", "after")
    specs = context.provision_comparisons("Сравни дефициты сценариев 17 и 18")
    assert len(specs) == 2
    rows = context.compare(specs)["content"]["rows"]
    assert all(
        r["before"] == "40" and r["after"] == "10" and r["delta"] == "-30" for r in rows
    )
    assert all(r["source_before"]["column"] == "value" for r in rows)
    assert (
        len(
            context.provision_comparisons(
                "Сравни дефициты сценариев 17 и 18", specs[:1]
            )
        )
        == 2
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "Укажи оба источника и версию.\n",
        "Укажи версию документа и источник.\n",
        "Сопоставь источники и укажи версию.\n",
    ],
)
def test_request_to_report_version_does_not_filter_out_canonical_norm(prefix):
    from src.agents.services.normgraph.normgraph_restriction_retriever import (
        NormGraphRestrictionRetriever,
    )

    hit = {
        "provenance": {
            "name": "EXAMPLE TEST NORM",
            "version": "2026",
            "numbering": "1.1",
        }
    }
    query = (
        prefix
        + "Проверь пункт 1.1 документа EXAMPLE TEST NORM, версия 2026: не менее 50 м."
    )
    assert NormGraphRestrictionRetriever._filter_explicit_references(
        [hit], query, match_distance=False
    ) == [hit]


@pytest.mark.parametrize(
    "extra",
    [
        {
            "agent": "compliance",
            "description": "Проверить соответствие социальной инфраструктуры",
        },
        {
            "agent": "scenario_data",
            "description": "Оценка пространственной реализуемости готового варианта",
        },
    ],
)
async def test_provision_assessment_does_not_invent_a_new_spatial_audit(
    fake_llm, extra
):
    calculation = {
        "id": "p",
        "agent": "provision",
        "scenario_id": 17,
        "description": "Рассчитать обеспеченность школами",
        "source_ids": [1],
        "required_artifacts": ["table"],
    }
    redundant = {
        **calculation,
        **extra,
        "id": "extra",
        "required_artifacts": ["analysis_text"],
    }
    fake_llm.json_responses = [
        json.dumps(
            {
                "objective": "Оценить обеспеченность",
                "requirements": [calculation, redundant],
            }
        ),
        json.dumps(
            {"objective": "Оценить обеспеченность", "requirements": [calculation]}
        ),
    ]
    goal = await GoalManager(fake_llm).create(
        "m",
        "Проверь социальную инфраструктуру: рассчитай обеспеченность школами, спрос, вместимость и дефицит.",
        [],
        17,
    )
    assert [r.agent for r in goal.requirements] == ["provision"]


async def test_existing_scenario_provision_is_not_a_hypothetical_effect(fake_llm):
    from src.agents.services.provision.provision_plan_builder import (
        ProvisionPlanBuilder,
    )

    fake_llm.json_responses = [
        '{"mode":"effects","service_name":"Школа","target_population":25000}',
        '{"mode":"provision","service_name":"Школа","target_population":25000}',
    ]
    plan = await ProvisionPlanBuilder(fake_llm).build_plan(
        "m",
        "Рассчитать обеспеченность услугой «Школа» для готового сценария 17 при 25000 жителей; без создания или изменения объектов.",
        ["Школа"],
    )
    assert plan.mode == "provision"


async def test_goal_cannot_lose_explicit_buffer_behind_compliance_layer(fake_llm):
    check = {
        "id": "check",
        "agent": "compliance",
        "scenario_id": 17,
        "description": "Проверить пункт и вернуть слой проверки",
        "source_ids": [1],
        "required_artifacts": ["compliance_summary", "compliance_result"],
    }
    buffer = {
        **check,
        "id": "zone",
        "agent": "restriction",
        "description": "Построить 50-метровую зону ограничения вокруг стоянки",
        "required_artifacts": ["feature_collection"],
    }
    fake_llm.json_responses = [
        json.dumps({"objective": "Проверка и буфер", "requirements": [check]}),
        json.dumps({"objective": "Проверка и буфер", "requirements": [check, buffer]}),
    ]
    goal = await GoalManager(fake_llm).create(
        "m",
        "Проверь требование, верни число проверенных объектов, нарушений и слой проверки, а также 50-метровую зону ограничения вокруг стоянки.",
        [],
        17,
    )
    assert {r.agent for r in goal.requirements} == {"compliance", "restriction"}
