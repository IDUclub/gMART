"""Figure-free prompts retain the exact UTF-8 bytes of the dev prompts."""

import hashlib
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from src.agents.model_clients.context_budget import OutputBudget, estimated_input_tokens
from src.agents.services.dvd import answer_generation, context_reducer, dvd_reasoning
from src.agents.services.dvd.context_reducer import DvdContextReducer, PreparedContext
from src.agents.services.dvd.document_comparison import COMPARISON_PROMPT
from src.agents.services.dvd.dvd_context import (
    DvdContextBuilder,
    FigureContext,
    has_figures,
    source_records,
)
from src.agents.services.dvd.dvd_reasoning import AnswerCritic
from src.agents.services.dvd.evidence_set import cited_context
from src.agents.services.dvd.figures import FIGURE_CRITIC_RULE, FIGURE_RULE
from tests.helpers import verdict_json

# SHA-256 of the system prompts evaluated from dev before ICII-205. Reducer
# snapshots use budget=2000; comparison uses comparison_summary="summary".
DEV_PROMPT_SHA256 = {
    "draft_norm": "2d55ad375220f4ca3bb0135a213df7f183655dedc099816ec86df0ebbf3a4389",
    "draft_knowledge": "60ba58307d8c82e71db98a2dc5c988993c8298446fed1c372bd4e1a77274800a",
    "reducer_summary": "c65b3c861f09d101f34ee2ebe6ca6fd13825c13167e3efa4691b98adc8faec4f",
    "reducer_select": "1f7ab3d08f8650ef36e5202bc29373c7692f7795505344aec73124ec63e5eaac",
    "critic": "f1600534f2bf97ea0d393a6b50a180eb702ab5fe8573bff5736bed938fd856d7",
    "comparison_base": "21de61291737d5d1b73e031d86ab1b7323e2cf34a5cfd72a4696605a08d3dc52",
    "comparison": "e480cf9f6877ab2d36e48549aaae5d778e446f76eb17e68dd33b1d336c113a34",
}


def assert_dev_prompt(name, prompt):
    assert FIGURE_RULE not in prompt
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == DEV_PROMPT_SHA256[name]


def context_with_figure(figure):
    return DvdContextBuilder().build_context(
        [
            {
                "id": "source",
                "name": "Источник",
                "type": "figure" if figure else "text",
                "text": "Расстояние составляет 15 м.",
                **(
                    {
                        "figure": {
                            "caption": "Рисунок 1 — Схема",
                            "description": "Расстояние составляет 15 м.",
                        }
                    }
                    if figure
                    else {}
                ),
            }
        ]
    )


@pytest.mark.parametrize("figure", [False, True])
@pytest.mark.parametrize(
    "intent,knowledge",
    [("norm", False), ("norm", True), ("document_comparison", False)],
)
async def test_drafting_and_comparison_prompts_match_dev_without_figures(
    service, fake_llm, monkeypatch, figure, intent, knowledge
):
    monkeypatch.setenv("DVD_KNOWLEDGE_FALLBACK", str(knowledge).lower())
    fake_llm.answer_texts = ["Ответ [1]."]
    context = context_with_figure(figure)
    _ = [
        e
        async for e in service._generate_answer(
            "m",
            "question",
            context,
            0,
            [],
            1,
            intent=intent,
            comparison_summary="summary",
        )
    ]
    prompt = fake_llm.chat_calls[-1].messages[0]["content"]
    suffix = "\nФрагменты нормативных документов:\n" + context
    assert prompt.endswith(suffix)
    system = prompt[: -len(suffix)]
    assert (FIGURE_RULE in system) is figure
    if figure:
        system = system.replace(FIGURE_RULE + "\n", "")
    key = (
        "comparison"
        if intent == "document_comparison"
        else ("draft_knowledge" if knowledge else "draft_norm")
    )
    assert_dev_prompt(key, system)
    assert_dev_prompt("comparison_base", COMPARISON_PROMPT)


@pytest.mark.parametrize("figure", [False, True])
async def test_critic_prompt_and_audit_budget_use_only_visible_figures(
    fake_llm, monkeypatch, figure
):
    context = context_with_figure(figure)
    fake_llm.json_responses = [verdict_json()]
    original_budget = dvd_reasoning.output_budget
    counted = []

    async def budget(*args, **kwargs):
        counted.append(deepcopy(args[2]))
        return await original_budget(*args, **kwargs)

    monkeypatch.setattr(dvd_reasoning, "output_budget", budget)
    await AnswerCritic(fake_llm).review(
        "m", "question", context, "Расстояние составляет 15 м [1]."
    )
    system = fake_llm.chat_calls[-1].messages[0]["content"]
    assert counted[0] == fake_llm.chat_calls[-1].messages
    assert (FIGURE_CRITIC_RULE in system) is figure
    assert_dev_prompt("critic", system.replace(FIGURE_CRITIC_RULE, ""))
    assert_dev_prompt("critic", AnswerCritic._prompt())


@pytest.mark.parametrize("figure", [False, True])
@pytest.mark.parametrize(
    "method,key",
    [("_summarize", "reducer_summary"), ("_select_evidence", "reducer_select")],
)
async def test_reducer_prompt_is_conditional_for_each_part(
    service, fake_llm, monkeypatch, figure, method, key
):
    counter = AsyncMock(wraps=context_reducer.remaining_output_tokens)
    monkeypatch.setattr(context_reducer, "remaining_output_tokens", counter)
    mixed = DvdContextBuilder().build_context(
        [
            {"name": "Текст", "type": "text", "text": "Текст содержит рисунок."},
            {
                "name": "Схема",
                "type": "figure",
                "figure": {
                    "caption": "Рисунок 1",
                    "description": "Расстояние составляет 15 м.",
                },
            },
        ]
    )
    parts = service.context_reducer._parts(mixed, 2000)
    part = next(p for p in parts if has_figures(p) is figure)
    label, (_, body) = next(iter(source_records(part).items()))
    if method == "_summarize":
        result = {
            "evidence": [{"source_id": label, "quotes": [body]}],
            "complete": True,
        }
    else:
        result = {"selections": {label: [0]}, "complete": True}
    fake_llm.json_responses = [json.dumps(result)]
    reduced = await getattr(service.context_reducer, method)(
        "m", "question", part, 2000
    )
    assert has_figures(reduced) is figure
    system = fake_llm.chat_calls[-1].messages[0]["content"]
    assert counter.call_args.args[2] == fake_llm.chat_calls[-1].messages
    assert (FIGURE_RULE in system) is figure
    assert_dev_prompt(key, system.replace(FIGURE_RULE, ""))


async def test_evidence_reserve_counts_figure_rule_only_when_present(fake_llm):
    fake_llm.model_input_tokens = AsyncMock(
        side_effect=(lambda model, messages, **kwargs: estimated_input_tokens(messages))
    )
    reducer = DvdContextReducer(fake_llm, window_tokens=8192)
    assert reducer.budget("question") == reducer.budget(
        "question", context=context_with_figure(False)
    )
    assert reducer.budget("question") - reducer.budget(
        "question", context=context_with_figure(True)
    ) == len(FIGURE_CRITIC_RULE.encode("utf-8"))
    plain = await reducer._evidence_budget(
        "m", "question", [], context_with_figure(False)
    )
    plain_messages = fake_llm.model_input_tokens.call_args.args[1]
    assert plain_messages == [{"role": "user", "content": "question"}]
    figure = await reducer._evidence_budget(
        "m", "question", [], context_with_figure(True)
    )
    figure_messages = fake_llm.model_input_tokens.call_args.args[1]
    assert figure_messages[0] == {"role": "system", "content": FIGURE_CRITIC_RULE}
    assert plain - figure == (
        estimated_input_tokens(figure_messages) - estimated_input_tokens(plain_messages)
    )


@pytest.mark.parametrize("figure", [False, True])
async def test_drafting_fixed_reserve_retains_only_active_figure_rule(
    service, fake_llm, monkeypatch, figure
):
    monkeypatch.setenv("DVD_KNOWLEDGE_FALLBACK", "false")
    context = context_with_figure(figure)
    monkeypatch.setattr(
        answer_generation,
        "output_budget",
        AsyncMock(
            side_effect=[
                OutputBudget(window_rest=0, tokens=0),
                OutputBudget(window_rest=1000, tokens=256),
            ]
        ),
    )
    counter = AsyncMock(return_value=20000)
    monkeypatch.setattr(answer_generation, "remaining_output_tokens", counter)
    service.context_reducer.prepare = AsyncMock(return_value=PreparedContext(context))
    fake_llm.answer_texts = ["Ответ [1]."]
    _ = [e async for e in service._generate_answer("m", "question", context, 0, [], 1)]
    reserved = counter.call_args.args[2][0]["content"]
    assert (FIGURE_RULE in reserved) is figure
    assert reserved.endswith("\nФрагменты нормативных документов:\n")


def test_cited_context_drops_rule_when_only_text_is_selected():
    context = DvdContextBuilder().build_context(
        [
            {"name": "Текст", "text": "Требование."},
            {"name": "Схема", "type": "figure", "figure": {"caption": "Рисунок 1"}},
        ]
    )
    assert has_figures(context)
    selected = cited_context(context, "Требование [1].")
    assert not has_figures(selected)
    assert_dev_prompt("critic", AnswerCritic._prompt(selected))
    assert has_figures(cited_context(context, "Схема [2]."))
    # A word in untrusted source text cannot enable application instructions.
    assert not has_figures("[1] Рисунок 1\nОписание изображения")
    assert has_figures(deepcopy(context))
    assert isinstance(context, FigureContext)
