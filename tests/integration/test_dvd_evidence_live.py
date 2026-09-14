"""Opt-in semantic regressions against VLLM_BASE_URL / VLLM_MODEL.

These are synthetic evidence fixtures, not statements of actual building law.
An available model must pass; an unset endpoint skips through the shared fixture.
"""

import pytest

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.dvd.context_reducer import DvdContextReducer
from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.dvd.dvd_reasoning import AnswerCritic

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "text,question,answer,expected",
    [
        (
            "Вместимость помещений при проектировании ЛИУ определяется заданием на проектирование.",
            "Как определяется вместимость?",
            "Для ЛИУ (лечебно-исцелительных учреждений) вместимость определяется заданием на проектирование [1].",
            False,
        ),
        (
            "Пункт 27.3. Вместимость определяется заданием на проектирование. Состав помещений указан в таблице 31.3.",
            "Как определяется вместимость?",
            "Таблица 27.3 устанавливает, что вместимость определяется заданием на проектирование [1].",
            False,
        ),
        (
            "Гостиницы для приезжих размещаются на неохраняемой территории учреждения.",
            "Где размещать городскую школу?",
            "Предоставленное правило относится к гостиницам учреждения. Данных о размещении городской школы в этом фрагменте нет.",
            True,
        ),
        (
            "Гостиницы для приезжих размещаются на неохраняемой территории учреждения.",
            "Где размещать городскую школу?",
            "Прямых норм о школах нет, но как общий ориентир школу следует размещать на неохраняемой территории [1].",
            False,
        ),
        (
            "Расстояние от объекта до границы участка должно быть не менее 15 м.",
            "Какое минимальное расстояние установлено?",
            "Минимум 10 м [1].",
            False,
        ),
        (
            "Расстояние от объекта до границы участка должно быть не менее 15 м.",
            "Какое минимальное расстояние установлено?",
            "Минимальное расстояние до границы участка — 15 м [1].",
            True,
        ),
        (
            "Оглавление: 14.7 Школы. 14.8 Мастерские.",
            "Каковы требования к школам?",
            "Фрагмент содержит лишь оглавление; текст требований не приведён.",
            True,
        ),
        (
            "Школа в исправительном учреждении размещается в жилой зоне учреждения.",
            "Как размещается школа в исправительном учреждении?",
            "В исправительном учреждении школа размещается в его жилой зоне [1].",
            True,
        ),
    ],
)
async def test_critic_preserves_scope_and_evidence(
    require_openai_backend, text, question, answer, expected
):
    url, model = require_openai_backend
    adapter = OpenAiCompatAdapter(url)
    context = DvdContextBuilder().build_context(
        [{"name": "Тестовый источник", "text": text}]
    )
    try:
        async with DvdContextReducer(adapter).model_window(model):
            verdict = await AnswerCritic(adapter).review(
                model, question, context, answer
            )
            assert verdict.satisfied is expected, verdict.critique
    finally:
        await adapter.client.close()


async def test_span_selection_keeps_original_table_spelling(require_openai_backend):
    url, model = require_openai_backend
    adapter = OpenAiCompatAdapter(url)
    reducer = DvdContextReducer(adapter)
    context = DvdContextBuilder().build_context(
        [
            {
                "name": "Тестовый источник",
                "text": "Т а б л и ц а 31.3\nКомнаты: 2–4-местные; 6,0 м2.\nСад: 100 м2.",
            }
        ]
    )
    try:
        async with reducer.model_window(model):
            result = await reducer._select_evidence(
                model, "Какова вместимость и площадь комнат?", context, 1500
            )
        assert "Комнаты: 2–4-местные; 6,0 м2." in result
        assert reducer._sources(result) == {"[1]"}
    finally:
        await adapter.client.close()


async def test_extracts_quotes_without_inventing_bibliographic_sources(
    require_openai_backend,
):
    url, model = require_openai_backend
    adapter = OpenAiCompatAdapter(url)
    reducer = DvdContextReducer(adapter)
    text = "Гостиницы для приезжих размещаются на неохраняемой территории учреждения. См. [10]."
    context = DvdContextBuilder().build_context(
        [{"name": "Тестовый источник", "text": text}]
    )
    try:
        async with reducer.model_window(model):
            result = await reducer._summarize(
                model, "Где размещаются гостиницы для приезжих?", context, 1500
            )
        assert reducer._sources(result) == {"[1]"}
        assert (
            "Гостиницы для приезжих размещаются на неохраняемой территории учреждения."
            in result
        )
    finally:
        await adapter.client.close()
