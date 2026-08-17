"""The evaluator catches a fluent answer that does not actually answer.

The reported case in one line: aggregates were in the observations, and the answer still said
"types are not specified, so the exact distribution cannot be reported".
"""

from __future__ import annotations

import json

import pytest

from src.agents.services.scenario_data_evaluator import (
    ScenarioDataEvaluator,
    Verdict,
    deterministic_checks,
    wants_layers,
)

WITH_AGGREGATE = [
    {
        "tool": "projects.GetScenarioPhysicalObjects",
        "layer_count": 0,
        "aggregate": {
            "total_records": 924,
            "breakdown": {
                "physical_object_type.name": {
                    "distinct_values": 3,
                    "counts": {"Жилой дом": 900, "Банк": 20, "Школа": 4},
                }
            },
        },
    }
]

NO_AGGREGATE = [{"tool": "dictionaries.GetBufferTypes", "layer_count": 0}]


class FakeLlm:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    async def chat(self, **kwargs):
        self.calls += 1
        content = (
            self._payload
            if isinstance(self._payload, str)
            else json.dumps(self._payload)
        )
        return {"message": {"content": content}}


class TestDeterministicChecks:
    def test_pleading_ignorance_while_counts_exist_is_rejected(self):
        answer = "Типы объектов не указаны, поэтому распределение сообщить невозможно."

        reasons = deterministic_checks("Какие объекты?", WITH_AGGREGATE, answer)

        assert reasons and "aggregate" in reasons[0]

    def test_an_answer_without_a_single_number_is_rejected(self):
        answer = "На территории представлены жилые дома, банки и школы."

        reasons = deterministic_checks("Какие объекты?", WITH_AGGREGATE, answer)

        assert any("нет ни одного" in reason for reason in reasons)

    def test_a_concrete_answer_passes(self):
        answer = "Всего 924 объекта: жилых домов 900, банков 20, школ 4."

        assert deterministic_checks("Какие объекты?", WITH_AGGREGATE, answer) == []

    def test_an_empty_answer_is_rejected(self):
        assert deterministic_checks("Какие объекты?", WITH_AGGREGATE, "  ") != []

    def test_ignorance_is_allowed_when_nothing_was_counted(self):
        """With no aggregates, "unknown" may simply be the truth."""
        answer = "Данных по типам нет."

        assert deterministic_checks("Какие объекты?", NO_AGGREGATE, answer) == []

    def test_a_layer_request_that_produced_none_is_rejected(self):
        answer = "Всего 924 объекта: жилых домов 900."

        reasons = deterministic_checks(
            "Покажи объекты на карте", WITH_AGGREGATE, answer
        )

        assert any("FeatureCollection" in reason for reason in reasons)

    def test_a_layer_request_with_layers_passes(self):
        observations = [dict(WITH_AGGREGATE[0], layer_count=1)]
        answer = "Всего 924 объекта: жилых домов 900. Слой отправлен на карту."

        assert deterministic_checks("Покажи на карте", observations, answer) == []


class TestWantsLayers:
    @pytest.mark.parametrize(
        "query",
        [
            "Покажи на карте физические объекты",
            "Отобрази слои сценария",
            "Нужен GeoJSON зданий",
            "Покажи границы территории",
        ],
    )
    def test_layer_requests_are_detected(self, query):
        assert wants_layers(query) is True

    def test_a_plain_count_question_is_not_a_layer_request(self):
        assert wants_layers("Сколько объектов в сценарии?") is False

    def test_plain_show_word_without_spatial_context_is_not_a_layer_request(self):
        assert wants_layers("Покажи названия и идентификаторы типов") is False


class TestEvaluator:
    @pytest.mark.asyncio
    async def test_a_rule_rejection_skips_the_judge(self):
        llm = FakeLlm({"sufficient": True, "missing": ""})
        evaluator = ScenarioDataEvaluator(llm)

        verdict = await evaluator.evaluate(
            "m", "Какие объекты?", WITH_AGGREGATE, "Типы неизвестны."
        )

        assert verdict.sufficient is False
        assert llm.calls == 0

    @pytest.mark.asyncio
    async def test_the_judge_can_reject_an_otherwise_clean_answer(self):
        llm = FakeLlm(
            {"sufficient": False, "missing": "Не назван ни один тип объекта."}
        )
        evaluator = ScenarioDataEvaluator(llm)

        verdict = await evaluator.evaluate(
            "m", "Какие объекты?", WITH_AGGREGATE, "Всего 924 объекта."
        )

        assert verdict.sufficient is False
        assert verdict.hint == "Не назван ни один тип объекта."

    @pytest.mark.asyncio
    async def test_the_judge_can_accept(self):
        llm = FakeLlm({"sufficient": True, "missing": ""})
        evaluator = ScenarioDataEvaluator(llm)

        verdict = await evaluator.evaluate(
            "m", "Какие объекты?", WITH_AGGREGATE, "Всего 924: домов 900, банков 20."
        )

        assert verdict.sufficient is True

    @pytest.mark.asyncio
    async def test_a_broken_judge_never_blocks_the_answer(self):
        """The judge is an improvement, not a gate."""
        evaluator = ScenarioDataEvaluator(FakeLlm("not json at all"))

        verdict = await evaluator.evaluate(
            "m", "Какие объекты?", WITH_AGGREGATE, "Всего 924: домов 900."
        )

        assert verdict.sufficient is True

    @pytest.mark.asyncio
    async def test_a_verdict_without_a_boolean_is_treated_as_no_opinion(self):
        """Otherwise a malformed reply would burn the retry budget on a fine answer."""
        evaluator = ScenarioDataEvaluator(FakeLlm({"missing": "что-то"}))

        verdict = await evaluator.evaluate(
            "m", "Какие объекты?", WITH_AGGREGATE, "Всего 924: домов 900."
        )

        assert verdict.sufficient is True


class TestVerdict:
    def test_defaults(self):
        assert Verdict(sufficient=True).hint == ""
        assert Verdict(sufficient=True).reasons == []


UNRESOLVED = [
    {
        "tool": "projects.GetScenarioServices",
        "layer_count": 0,
        "aggregate": {
            "total_records": 900,
            "breakdown": {
                "service_type_id": {"distinct_values": 12, "counts": {"3": 90}}
            },
        },
        "unresolved_references": ["service_type_id"],
    }
]


class TestUnresolvedReferences:
    def test_an_answer_by_id_instead_of_name_is_rejected(self):
        """ "90 services of type 3" is not what "which services" asked for."""
        answer = "Всего 900 сервисов, из них 90 относятся к типу 3."

        reasons = deterministic_checks("Какие сервисы?", UNRESOLVED, answer)

        assert any("справочник" in reason for reason in reasons)

    def test_the_reason_names_the_field_to_resolve(self):
        reasons = deterministic_checks("Какие сервисы?", UNRESOLVED, "900 сервисов.")

        assert any("service_type_id" in reason for reason in reasons)

    def test_nothing_is_flagged_once_the_names_are_present(self):
        observations = [
            {
                "tool": "projects.GetScenarioServices",
                "layer_count": 0,
                "aggregate": {
                    "total_records": 900,
                    "breakdown": {
                        "service_type.name": {
                            "distinct_values": 2,
                            "counts": {"Школа": 500, "Банк": 400},
                        }
                    },
                },
            }
        ]
        answer = "Всего 900 сервисов: школ 500, банков 400."

        assert deterministic_checks("Какие сервисы?", observations, answer) == []
