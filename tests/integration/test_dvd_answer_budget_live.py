"""Opt-in real-model continuation using synthetic evidence (not building law)."""

import pytest

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.dvd.answer_generation import DvdAnswerGenerator, message_cost
from src.agents.services.dvd.context_reducer import DvdContextReducer

pytestmark = pytest.mark.integration


async def test_small_initial_budget_recovers_complete_evidence_list(
    require_openai_backend,
    monkeypatch,
):
    url, model = require_openai_backend
    monkeypatch.setenv("DVD_ANSWER_MIN_TOKENS", "128")
    monkeypatch.setenv("DVD_ANSWER_MAX_TOKENS", "16384")
    monkeypatch.setenv("DVD_ANSWER_RETRIES", "4")
    adapter = OpenAiCompatAdapter(url)
    calls = []

    class RecordedModel:
        model_context_window = adapter.model_context_window

        async def chat(self, *args, **kwargs):
            call = {
                "messages": args[1],
                "budget": kwargs["options"]["num_predict"],
                "reason": None,
            }
            calls.append(call)
            stream = await adapter.chat(*args, **kwargs)

            async def record():
                async for part in stream:
                    if part.done_reason:
                        call["reason"] = part.done_reason
                    yield part

            return record()

    rules = [
        f"For test object ITEM-{i:02}, the minimum distance is {i + 10} m."
        for i in range(1, 81)
    ]
    context = "[1] Synthetic evidence\n" + "\n".join(rules)
    question = (
        "Copy every rule verbatim, in order, one rule per line. Include all 80 rules."
    )

    def messages(evidence):
        return [
            {
                "role": "system",
                "content": "Use only the evidence. No introduction or conclusion.\n"
                + evidence,
            },
            {"role": "user", "content": question},
        ]

    try:
        reducer = DvdContextReducer(RecordedModel())
        async with reducer.model_window(model):
            result = await DvdAnswerGenerator(reducer).generate(
                model,
                question,
                context,
                0,
                messages,
                iteration=1,
            )
            assert all(rule in result for rule in rules), result
            assert all(result.count(rule) == 1 for rule in rules), result
            assert any(c["reason"] in {"length", "max_tokens"} for c in calls), calls
            assert calls[-1]["reason"] == "stop"
            assert all(
                message_cost(c["messages"]) + c["budget"] <= reducer.window
                for c in calls
            )
            print(
                "Budget continuation calls:",
                [
                    (
                        c["budget"],
                        c["reason"],
                        any(m["role"] == "assistant" for m in c["messages"]),
                    )
                    for c in calls
                ],
            )
    finally:
        await adapter.client.close()
