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
    base_comparison_requested,
    calculation_request,
    comparison_declined,
    grouped,
    indicator_comparison,
    names_indicator,
    normalize_indicators,
    render_indicators,
    scenario_labels,
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
    "query,selected",
    [
        ("Сравни сценарии, но без базового", 772),
        ("Сравни сценарии", None),
        ("Показатели", None),
    ],
)
def test_missing_scope_requires_clarification(query, selected):
    with pytest.raises(ValueError):
        scenario_scope(query, selected)


@pytest.mark.parametrize(
    "query",
    [
        "Сравни показатели с базовым сценарием",
        "Чем мой сценарий отличается от базового?",
        "Какая разница по численности населения?",
    ],
)
def test_comparison_without_a_second_id_falls_back_to_the_project_base(query):
    assert scenario_scope(query, 772) == [772]


@pytest.mark.parametrize(
    "query,default,expected",
    [
        ("Сравни показатели с базовым сценарием", False, True),
        ("Чем мой сценарий отличается от базового?", False, True),
        ("Покажи показатели сценария", False, False),
        ("Покажи показатели сценария", True, True),
        ("Сравни показатели, но без сравнения с базовым", True, False),
        ("Сравни с базовым не надо, дай только текущий сценарий", True, False),
        ("Покажи только мой сценарий", True, False),
        ("Не показывай гексагоны, сравни с базовым", False, True),
    ],
)
def test_base_comparison_follows_the_prompt_and_the_entry_point(
    query, default, expected
):
    assert base_comparison_requested(query, [772], default=default) is expected


@pytest.mark.parametrize(
    "query",
    [
        "Покажи показатели без сравнения",
        "Покажи показатели, не сравнивай ни с чем",
    ],
)
def test_declining_every_comparison_is_a_plain_summary_not_a_clarification(query):
    assert comparison_declined(query) is True
    assert scenario_scope(query, 772) == [772]


def test_declining_only_the_base_still_requires_explicit_ids():
    query = "Сравни сценарии, но без базового"
    assert comparison_declined(query) is False
    with pytest.raises(ValueError):
        scenario_scope(query, 772)


@pytest.mark.parametrize("ids", [[772, 848], []])
def test_base_is_never_added_to_an_explicit_multi_scenario_scope(ids):
    assert base_comparison_requested("Сравни с базовым", ids, default=True) is False


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
    assert "разница 848 − 772 = +2,13 км²" in answer


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


def test_scenarios_are_labelled_by_role_and_name():
    labels = scenario_labels(
        {846: "Исходный сценарий", 848: "Застройка у реки", 900: None},
        selected=848,
        base_id=846,
    )

    assert labels == {
        846: "Базовый сценарий «Исходный сценарий»",
        848: "Ваш сценарий «Застройка у реки»",
        900: "Сценарий 900",
    }


def test_labelled_comparison_names_scenarios_instead_of_ids():
    scenarios = {
        sid: normalize_indicators(
            [indicator(17, value, "Земли жилой застройки", "%", sid)], sid
        )
        for sid, value in [(846, 23.41), (848, 97.6)]
    }

    answer, rows = render_indicators(
        IndicatorRequest(
            operation="values", names=["Земли жилой застройки"], missing=[]
        ),
        scenarios,
        query="Покажи",
        labels=scenario_labels(
            {846: "Исходный", 848: "У реки"}, selected=848, base_id=846
        ),
    )

    assert "Базовый сценарий «Исходный»: «Земли жилой застройки» — 23,41 %." in answer
    assert (
        "разница ваш сценарий «У реки» − базовый сценарий «Исходный» "
        "= +74,19 процентного пункта" in answer
    )
    assert "(база — базовый сценарий «Исходный»;" in answer
    assert "846" not in answer and "848" not in answer
    assert [r["scenario"] for r in rows] == [
        "Базовый сценарий «Исходный»",
        "Ваш сценарий «У реки»",
    ]


def facts_of(sid, *items):
    return normalize_indicators(
        [indicator(iid, value, name, unit, sid) for iid, name, value, unit in items],
        sid,
    )


def compare(request, scenarios, **kwargs):
    return indicator_comparison(
        request,
        scenarios,
        query="",
        names={846: "Исходный", 848: "У реки"},
        selected=848,
        base_id=846 if len(scenarios) > 1 else None,
        **kwargs,
    )


ALL = IndicatorRequest(operation="all", names=[], missing=[])


@pytest.mark.parametrize(
    ("query", "named"),
    [
        ("Сравни с базовым сценарием", False),
        ("Что изменилось?", False),
        ("Покажи показатели", False),
        ("Покажи все показатели", False),
        ("Покажи численность населения", True),
        ("Покажи «Земли жилой застройки»", True),
        ("Какие показатели по площади территории?", True),
    ],
)
def test_a_generic_indicator_request_names_no_indicator(query, named):
    assert names_indicator(query) is named


def test_large_numbers_are_grouped_by_thousands():
    assert grouped(6949748051) == "6 949 748 051"
    assert grouped(-25809) == "-25 809"
    assert grouped(1000.5) == "1 000,5"
    assert grouped(97.6) == "97,6"


def test_summary_counts_ranks_changes_and_lists_gaps_without_repeating_the_table():
    base = facts_of(
        846,
        (17, "Земли жилой застройки", 23.41, "%"),
        (20, "Земли сельскохозяйственного назначения", 66.82, "%"),
        (21, "Земли промышленного назначения", 0.62, "%"),
        (299, "Срок рекультивации территории", 34175, "дней"),
        (198, "Транспортное обеспечение", 5, None),
        (1, "Численность населения", 499899, "человек"),
    )
    mine = facts_of(
        848,
        (17, "Земли жилой застройки", 97.6, "%"),
        (20, "Земли сельскохозяйственного назначения", 0, "%"),
        (21, "Земли промышленного назначения", 0, "%"),
        (299, "Срок рекультивации территории", 8366, "дней"),
        (198, "Транспортное обеспечение", 5, None),
    )

    text, rows, labels = compare(ALL, {846: base, 848: mine})

    assert "Сравниваются: базовый сценарий «Исходный» → ваш сценарий «У реки»." in text
    assert (
        "Показателей: в базовом — 6, в вашем — 5. Изменились — 4, "
        "без изменений — 1, нет значения в вашем — 1." in text
    )
    ranked = [
        "• Земли жилой застройки: 23,41 % → 97,6 % (+74,19 п. п.)",
        "• Земли сельскохозяйственного назначения: 66,82 % → 0 % (-66,82 п. п.)",
        "• Земли промышленного назначения: 0,62 % → 0 % (-0,62 п. п.)",
        "• Срок рекультивации территории: 34 175 → 8 366 дней (-75,5 %)",
    ]
    assert [text.index(line) for line in ranked] == sorted(
        text.index(line) for line in ranked
    )
    assert "Транспортное обеспечение" not in text
    assert "Нет значения в вашем сценарии: Численность населения." in text
    assert text.endswith("Все значения — в таблице.")
    assert labels == {
        "indicator": "Показатель",
        "unit": "Ед.",
        "scenario_846": "Базовый сценарий «Исходный»",
        "scenario_848": "Ваш сценарий «У реки»",
        "difference": "Разница",
        "change_percent": "Изменение, %",
    }
    by_name = {row["indicator"]: row for row in rows}
    assert len(rows) == 6
    assert by_name["Земли жилой застройки"] == {
        "indicator": "Земли жилой застройки",
        "unit": "% (разница — п. п.)",
        "scenario_846": 23.41,
        "scenario_848": 97.6,
        "difference": 74.19,
        "change_percent": None,
    }
    assert by_name["Срок рекультивации территории"]["difference"] == -25809
    assert by_name["Срок рекультивации территории"]["change_percent"] == -75.5
    assert by_name["Численность населения"]["scenario_848"] is None
    assert by_name["Численность населения"]["difference"] is None


def test_a_named_indicator_is_answered_alone():
    base = facts_of(
        846,
        (17, "Земли жилой застройки", 23.41, "%"),
        (4, "Площадь территории", 8.08, "км2"),
    )
    mine = facts_of(
        848,
        (17, "Земли жилой застройки", 97.6, "%"),
        (4, "Площадь территории", 8.08, "км2"),
    )

    text, rows, _ = compare(
        IndicatorRequest(
            operation="values", names=["Земли жилой застройки"], missing=["Шум"]
        ),
        {846: base, 848: mine},
    )

    assert [row["indicator"] for row in rows] == ["Земли жилой застройки"]
    assert "«Земли жилой застройки»: 23,41 % → 97,6 % (+74,19 п. п.)." in text
    assert "«Шум» — такого показателя нет в данных сценариев." in text
    assert "Площадь территории" not in text and "Показателей:" not in text


def test_units_that_differ_are_not_subtracted():
    text, rows, _ = compare(
        ALL,
        {
            846: facts_of(846, (4, "Площадь территории", 8.08, "км2")),
            848: facts_of(848, (4, "Площадь территории", 808, "га")),
        },
    )

    assert rows[0]["unit"] == "км² / га"
    assert rows[0]["difference"] is None
    assert "единицы не совпадают — 1" in text
    assert "Единицы не совпадают, разница не считается: Площадь территории." in text


def test_value_cells_drop_a_float_zero_tail_and_a_dash_unit_means_none():
    base = facts_of(
        846,
        (299, "Срок рекультивации территории", 34175.0, "дней"),
        (11, "Иные категории земель", 0.0, "%"),
        (30, "Коэффициент застройки", 0.1385, "-"),
    )
    mine = facts_of(
        848,
        (299, "Срок рекультивации территории", 8366.0, "дней"),
        (11, "Иные категории земель", 0.0, "%"),
        (30, "Коэффициент застройки", 0.2, None),
    )

    text, rows, _ = compare(ALL, {846: base, 848: mine})

    by_name = {row["indicator"]: row for row in rows}
    days = by_name["Срок рекультивации территории"]
    assert (days["scenario_846"], days["scenario_848"]) == (34175, 8366)
    assert type(days["scenario_846"]) is int
    assert type(by_name["Иные категории земель"]["scenario_848"]) is int
    ratio = by_name["Коэффициент застройки"]
    assert ratio["unit"] is None
    assert (ratio["scenario_846"], ratio["difference"]) == (0.1385, 0.0615)
    assert "единицы не совпадают" not in text


def test_a_single_scenario_gets_one_value_column():
    text, rows, labels = compare(
        ALL, {848: facts_of(848, (17, "Земли жилой застройки", 97.6, "%"))}
    )

    assert list(labels) == ["indicator", "unit", "scenario_848"]
    assert rows == [
        {"indicator": "Земли жилой застройки", "unit": "%", "scenario_848": 97.6}
    ]
    assert text == (
        "Ваш сценарий «У реки»: показателей уровня сценария — 1."
        "\n\nВсе значения — в таблице."
    )


def test_table_columns_take_labels_and_keep_unknown_keys():
    table = ScenarioDataService._table_from_result(
        [{"indicator": "A", "value": 1}],
        name="t",
        title="T",
        labels={"indicator": "Показатель"},
    )

    assert table["columns"] == [
        {"key": "indicator", "label": "Показатель"},
        {"key": "value", "label": "value"},
    ]
