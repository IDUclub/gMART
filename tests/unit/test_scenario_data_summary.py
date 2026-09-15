"""The model may rephrase a comparison, but only with numbers the code computed."""

import json
from decimal import Decimal

import pytest

from src.agents.services.scenario_data.scenario_data_summary import (
    SUMMARY_TAIL,
    comparison_facts,
    grounded,
    numbers_in,
    summarize_comparison,
)

ROWS = [
    {
        "indicator": "Земли жилой застройки",
        "unit": "% (разница — п. п.)",
        "scenario_846": 23.41,
        "scenario_848": 97.6,
        "difference": 74.19,
        "change_percent": None,
    },
    {
        "indicator": "Стоимость рекультивации",
        "unit": "руб",
        "scenario_846": 6949748051,
        "scenario_848": 2257190106,
        "difference": -4692557945,
        "change_percent": -67.5,
    },
    {
        "indicator": "Транспортное обеспечение",
        "unit": None,
        "scenario_846": 5,
        "scenario_848": 5,
        "difference": 0,
        "change_percent": 0,
    },
]
LABELS = {
    "indicator": "Показатель",
    "unit": "Ед.",
    "scenario_846": "Базовый сценарий «Исходный»",
    "scenario_848": "Ваш сценарий «Вариант 17»",
    "difference": "Разница",
    "change_percent": "Изменение, %",
}
COUNTS = {"scenario_846": 3, "scenario_848": 3}
FACTS = comparison_facts(ROWS, LABELS, COUNTS, [])


def test_russian_numbers_are_read_with_their_rounding():
    assert numbers_in("с 23,41 % до 97,6 %") == [
        (Decimal("23.41"), Decimal("0.005")),
        (Decimal("97.6"), Decimal("0.05")),
    ]
    assert numbers_in("стоимость 6 949 748 051 руб") == [
        (Decimal(6949748051), Decimal("0.5"))
    ]
    assert numbers_in("около 2,3 млрд руб") == [
        (Decimal("2.3") * 10**9, Decimal("0.05") * 10**9)
    ]
    assert numbers_in("площадь в км2") == []


def test_facts_carry_labels_and_counts_but_no_identifiers():
    assert "846" not in json.dumps(FACTS) and "848" not in json.dumps(FACTS)
    assert FACTS["изменились"] == 2 and FACTS["без_изменений"] == 1
    assert FACTS["показателей_в_сценарии"] == {
        "Базовый сценарий «Исходный»": 3,
        "Ваш сценарий «Вариант 17»": 3,
    }


@pytest.mark.parametrize(
    "text",
    [
        "Доля жилой застройки выросла на 74,19 п. п.: с 23,41 % до 97,6 %.",
        "Доля жилой застройки выросла почти до 98 %.",
        "Стоимость рекультивации снизилась на 67,5 %, примерно до 2,3 млрд руб.",
        "Стоимость снизилась с 6 949 748 051 до 2 257 190 106 руб.",
        "В сценарии «Вариант 17» изменились 2 показателя, 1 остался прежним.",
    ],
)
def test_numbers_from_the_facts_pass_as_is_or_rounded(text):
    assert grounded(text, FACTS)


@pytest.mark.parametrize(
    "text",
    [
        "Доля жилой застройки выросла в 4 раза.",
        "Стоимость снизилась на 4,7 млрд руб. в сценарии 848.",
        "Доля выросла на 316,9 %.",
    ],
)
def test_a_number_the_code_did_not_compute_is_rejected(text):
    assert not grounded(text, FACTS)


class Llm:
    def __init__(self, content="", *, done_reason="stop", error=None):
        self.content = content
        self.done_reason = done_reason
        self.error = error
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return {"message": {"content": self.content}, "done_reason": self.done_reason}


async def summarize(llm):
    return await summarize_comparison(
        llm,
        "model",
        rows=ROWS,
        column_labels=LABELS,
        counts=COUNTS,
        missing=["Шум"],
        fallback="FALLBACK",
    )


async def test_a_grounded_summary_sits_between_the_header_and_the_tail():
    llm = Llm(
        "Доля жилой застройки выросла на 74,19 п. п., стоимость рекультивации "
        "снизилась на 67,5 %. Все значения — в таблице."
    )

    text = await summarize(llm)

    assert text == "\n\n".join(
        [
            "Сравниваются: базовый сценарий «Исходный» → ваш сценарий «Вариант 17».",
            "Доля жилой застройки выросла на 74,19 п. п., стоимость рекультивации "
            "снизилась на 67,5 %.",
            "«Шум» — такого показателя нет в данных сценариев.",
            SUMMARY_TAIL,
        ]
    )
    call = llm.calls[0]
    assert call["options"]["temperature"] == 0 and call["think"] is False
    prompt = call["messages"][0]["content"]
    assert "846" not in prompt and "848" not in prompt


@pytest.mark.parametrize(
    "llm",
    [
        Llm("Доля жилой застройки выросла в 4 раза."),
        Llm(""),
        Llm("{}"),
        Llm("Доля жилой застройки выросла на 74,19 п. п.", done_reason="length"),
        Llm(error=RuntimeError("model is down")),
    ],
    ids=["foreign-number", "empty", "no-text", "truncated", "exception"],
)
async def test_an_untrustworthy_summary_falls_back_to_the_computed_one(llm):
    assert await summarize(llm) == "FALLBACK"
