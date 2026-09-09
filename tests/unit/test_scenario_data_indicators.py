"""Indicator identity and scenario provenance must survive answer preparation."""

import json
from unittest.mock import AsyncMock

import pytest

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data.scenario_data_evaluator import (
    ScenarioDataEvaluator,
)
from src.agents.services.scenario_data.scenario_data_indicators import (
    IndicatorRequest,
    calculation_request,
    normalize_indicators,
    render_indicators,
    scenario_scope,
    validate_request,
)
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService


def indicator(iid=4, value=5.95, name="Площадь территории", unit="км2", sid=772):
    return {
        "indicator_value_id": 9000 + iid,
        "indicator": {
            "indicator_id": iid,
            "name_full": name,
            "measurement_unit": {"name": unit} if unit else None,
        },
        "scenario": {"id": sid, "name": f"Сценарий {sid}"},
        "territory": None,
        "hexagon_id": None,
        "value": value,
        "comment": None,
    }


def test_indicator_summary_retains_the_name_value_unit_and_scenario():
    summary = ScenarioDataService._result_summary([indicator()])
    assert "Площадь территории" in summary
    assert "км2" in summary
    assert "5.95" in summary
    assert "772" in summary


@pytest.mark.asyncio
async def test_table_error_code_cannot_override_a_negative_factual_judgement():
    evaluator = ScenarioDataEvaluator(None)
    evaluator._judge = AsyncMock(return_value=(False, "required_table_not_emitted"))
    result = await evaluator.evaluate(
        "model",
        "Сравни площадь сценариев 772 и 848",
        [{"retrieved": True, "table_count": 1, "table_complete": True}],
        "В обоих сценариях 33.2 км².",
    )
    assert not result.sufficient


@pytest.mark.parametrize(
    "query,selected,expected",
    [
        ("Сравни площадь сценариев 772 и 848", 772, [772, 848]),
        ("Сравни сценарий 848 со сценарием 772", 848, [848, 772]),
        ("Показатели сценария 848", 772, [848]),
        ("Численность населения, больше 100 человек?", 772, [772]),
        ("Сравни сценарии 772, 848 и 900", None, [772, 848, 900]),
    ],
)
def test_scope_comes_from_explicit_scenario_references(query, selected, expected):
    assert scenario_scope(query, selected) == expected


@pytest.mark.parametrize(
    "query,selected", [("Сравни сценарии", 772), ("Показатели", None)]
)
def test_missing_scope_requires_clarification(query, selected):
    with pytest.raises(ValueError):
        scenario_scope(query, selected)


@pytest.mark.parametrize(
    "result",
    [
        [indicator(sid=848)],
        [indicator(value=float("nan"))],
        [indicator(value=True)],
        [indicator(), indicator(value=9)],
        {"items": [indicator()], "total": 2},
        {"items": [indicator()], "complete": False},
    ],
)
def test_invalid_or_partial_evidence_cannot_support_an_answer(result):
    with pytest.raises(ValueError):
        normalize_indicators(result, 772)


def test_scope_and_deduplication_of_indicator_values():
    territorial = {**indicator(value=99), "territory": {"id": 1}}
    assert len(normalize_indicators([indicator(), indicator(), territorial], 772)) == 1


def test_saved_values_are_not_rounded_to_zero():
    facts = normalize_indicators([indicator(value=0.000012345)], 772)
    answer, rows = render_indicators(
        IndicatorRequest(operation="values", names=["Площадь территории"], missing=[]),
        {772: facts},
        query="Площадь территории",
    )
    assert "0,000012345 км²" in answer
    assert rows[0]["value"] == 0.000012345


def test_all_selected_names_does_not_expand_a_single_indicator_request():
    facts = normalize_indicators(
        [indicator(), indicator(28, 2, "Средняя этажность", "этажей")], 772
    )
    request = validate_request(
        IndicatorRequest(operation="all", names=["средняя этажность"], missing=[]),
        facts,
        "Какова средняя этажность как сохранённый показатель?",
    )
    assert request.operation == "values" and request.names == ["Средняя этажность"]


def test_integer_trailing_zeroes_are_preserved():
    facts = normalize_indicators([indicator(value=1000)], 772)
    answer, _ = render_indicators(
        IndicatorRequest(operation="values", names=["Площадь территории"], missing=[]),
        {772: facts},
        query="Площадь территории",
    )
    assert "1000 км²" in answer


def test_missing_is_distinct_from_zero_and_score():
    facts = normalize_indicators([indicator(197, 5, "Население", None, 848)], 848)
    request = IndicatorRequest(
        operation="values", names=[], missing=["Численность населения"]
    )
    answer, rows = render_indicators(
        request, {848: facts}, query="Численность населения в людях"
    )
    assert "данные отсутствуют" in answer
    assert "5 человек" not in answer
    assert not rows


def test_comparison_preserves_percentage_points_and_negative_signs():
    scenarios = {
        sid: normalize_indicators(
            [indicator(17, value, "Земли жилой застройки", "%", sid)], sid
        )
        for sid, value in [(772, 36.68), (848, 97.6)]
    }
    answer, rows = render_indicators(
        IndicatorRequest(
            operation="values", names=["Земли жилой застройки"], missing=[]
        ),
        scenarios,
        query="Сравни",
    )
    assert "60,92 процентного пункта" in answer
    assert [r["value"] for r in rows] == [36.68, 97.6]


def test_calculated_density_does_not_replace_the_saved_value():
    facts = normalize_indicators(
        [
            indicator(),
            indicator(1, 13016, "Численность населения", "человек"),
            indicator(37, 22, "Плотность населения", "чел/км2"),
        ],
        772,
    )
    answer, rows = render_indicators(
        IndicatorRequest(operation="density", names=[], missing=[]),
        {772: facts},
        query="Рассчитай плотность",
    )
    assert "2187,563" in answer and "22 чел/км²" in answer
    assert "Причина расхождения по этим данным не установлена" in answer
    assert next(r for r in rows if r["indicator_id"] == 37)["value"] == 22


def test_model_cannot_invent_an_indicator_or_erase_a_known_one():
    facts = normalize_indicators([indicator()], 772)
    with pytest.raises(ValueError):
        validate_request(
            IndicatorRequest(operation="values", names=["выдумка"], missing=[]), facts
        )
    with pytest.raises(ValueError):
        validate_request(
            IndicatorRequest(
                operation="values", names=[], missing=["Площадь территории"]
            ),
            facts,
        )


def test_literal_absent_indicator_is_not_replaced_by_a_score():
    facts = normalize_indicators([indicator(197, 5, "Население", None)], 772)
    request = validate_request(
        IndicatorRequest(
            operation="values", names=["Численность населения"], missing=[]
        ),
        facts,
        "Какова численность населения в людях?",
    )
    assert request.names == [] and request.missing == ["Численность населения"]


def test_explicit_density_formula_cannot_become_an_inventory():
    assert (
        calculation_request(
            "Какова плотность населения? Отдельно рассчитай её по численности и площади."
        ).operation
        == "density"
    )
    assert calculation_request("Какова плотность населения?") is None


def test_explicit_comparison_base_overrides_display_order():
    scenarios = {
        sid: normalize_indicators([indicator(value=value, sid=sid)], sid)
        for sid, value in [(848, 8.08), (772, 5.95)]
    }
    answer, _ = render_indicators(
        IndicatorRequest(operation="values", names=["Площадь территории"], missing=[]),
        scenarios,
        query="Сценарии 848 и 772, изменение относительно 772",
    )
    assert "разница 848 − 772 = 2,13 км²" in answer


@pytest.mark.parametrize(
    "first,second,unit1,unit2,expected",
    [
        (0, 0, "%", "%", "исходное значение равно нулю"),
        (-7.99, -8.64, None, None, "= -0,65 единица не указана"),
        (1, 2, "км2", "м2", "единицы не совпадают"),
    ],
)
def test_comparison_zero_negative_and_incompatible_units(
    first, second, unit1, unit2, expected
):
    scenarios = {
        sid: normalize_indicators([indicator(value=value, unit=unit, sid=sid)], sid)
        for sid, value, unit in [(772, first, unit1), (848, second, unit2)]
    }
    answer, _ = render_indicators(
        IndicatorRequest(operation="values", names=["Площадь территории"], missing=[]),
        scenarios,
        query="Сравни",
    )
    assert expected in answer


def test_population_question_cannot_be_answered_with_a_score():
    facts = normalize_indicators([indicator(197, 5, "Население", None)], 772)
    with pytest.raises(ValueError):
        validate_request(
            IndicatorRequest(operation="values", names=["Население"], missing=[]),
            facts,
            "Сколько жителей в сценарии?",
        )


def test_local_density_comparison_does_not_require_a_second_scenario():
    assert scenario_scope(
        "Сравни сохранённую плотность населения с расчётной", 772
    ) == [772]


class IndicatorMcp:
    def __init__(self, *, denied=None, wrong_scope=False):
        self.calls = []
        self.denied = denied
        self.wrong_scope = wrong_scope
        self.tools = [
            UrbanMcpTool(
                group=group,
                name=name,
                title=name,
                description="",
                tags=(),
                input_schema={
                    "type": "object",
                    "properties": {"scenario_id": {"type": "integer"}},
                    "required": ["scenario_id"],
                },
            )
            for group, name in [
                ("projects", "GetScenarioById"),
                ("indicators", "GetScenarioIndicatorsValues"),
            ]
        ]

    async def load_tools(self):
        return self.tools

    async def execute_tool(self, group, name, arguments, *, meta):
        sid = arguments["scenario_id"]
        assert meta == {"scenario_id": sid}
        self.calls.append((name, sid))
        if sid == self.denied:
            raise PermissionError("denied")
        if name == "GetScenarioById":
            return {
                "scenario_id": sid,
                "project": {"project_id": sid},
                "name": str(sid),
            }
        return [
            indicator(
                value=5.95 if sid == 772 else 8.08, sid=772 if self.wrong_scope else sid
            )
        ]


async def run_indicators(
    monkeypatch,
    fake_llm,
    fake_urban,
    state_store,
    mcp,
    query="Сравни площадь территории сценариев 772 и 848",
):
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter",
        lambda *a, **kw: fake_llm,
    )
    fake_llm.json_responses = [
        json.dumps(
            {"operation": "values", "names": ["Площадь территории"], "missing": []}
        )
    ]
    service = ScenarioDataService("http://llm", None, fake_urban, state_store)
    return [
        e
        async for e in service.run_scenario_data_pipeline(
            mcp,
            "caller-token",
            "model",
            0,
            query,
            scenario_id=772,
            persist_history=False,
        )
    ]


async def test_full_pipeline_fetches_each_authorized_scenario_and_computes_the_difference(
    monkeypatch, fake_llm, fake_urban, state_store
):
    mcp = IndicatorMcp()
    events = await run_indicators(monkeypatch, fake_llm, fake_urban, state_store, mcp)
    answer = "".join(
        e["content"].get("text", "") for e in events if e["type"] == "chunk"
    )
    assert "2,13 км²" in answer and "35,7983%" in answer
    assert mcp.calls == [
        ("GetScenarioById", 772),
        ("GetScenarioIndicatorsValues", 772),
        ("GetScenarioById", 848),
        ("GetScenarioIndicatorsValues", 848),
    ]


async def test_denied_second_scenario_cannot_publish_partial_comparison(
    monkeypatch, fake_llm, fake_urban, state_store
):
    with pytest.raises(PermissionError):
        await run_indicators(
            monkeypatch, fake_llm, fake_urban, state_store, IndicatorMcp(denied=848)
        )


async def test_wrong_scenario_result_does_not_emit_a_table_or_wrong_values(
    monkeypatch, fake_llm, fake_urban, state_store
):
    events = await run_indicators(
        monkeypatch, fake_llm, fake_urban, state_store, IndicatorMcp(wrong_scope=True)
    )
    assert not any(e["type"] == "table" for e in events)
    answer = "".join(
        e["content"].get("text", "") for e in events if e["type"] == "chunk"
    )
    assert "Не удалось подтвердить" in answer
