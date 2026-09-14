import asyncio
import json
import re

import pytest

from src.agents.services.dvd.context_reducer import DvdContextReducer, cost, split_text


class Summarizer:
    def __init__(self, fail=False):
        self.active = self.peak = 0
        self.calls = []
        self.output_budgets = []
        self.fail = fail

    async def chat(self, model, messages, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append(messages)
        self.output_budgets.append(kwargs["options"]["num_predict"])
        try:
            await asyncio.sleep(0.002)
            sources = json.loads(messages[1]["content"].split("\nПроверь", 1)[0])[
                "sources"
            ]
            if self.fail and any("FAIL_SOURCE" in part["text"] for part in sources):
                raise RuntimeError("temporary failure")
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "evidence": [
                                {
                                    "source_id": part["source_id"],
                                    "quotes": sorted(
                                        set(re.findall(r"FACT\d+", part["text"]))
                                    ),
                                }
                                for part in sources
                            ],
                            "complete": True,
                        }
                    )
                }
            }
        finally:
            self.active -= 1


def test_unicode_split_preserves_every_character():
    text = "Длинный текст 🔥\n" * 1000
    parts = split_text(text, 101)
    assert "".join(parts) == text
    assert all(cost(p) <= 101 for p in parts)


async def test_parallel_map_and_audit_keep_late_facts_and_sources():
    llm = Summarizer()
    reducer = DvdContextReducer(llm, concurrency=3)
    context = "\n\n".join(
        f"[{i}] Doc, version 1, clause {i}\n" + "padding " * 400 + f"FACT{i}"
        for i in range(1, 8)
    )
    result = await reducer.prepare("m", "question", context)
    assert 1 < llm.peak <= 3
    assert not result.failed_parts
    assert all(
        f"FACT{i}" in result.text and f"[{i}]" in result.text for i in range(1, 8)
    )
    assert cost(result.text) <= reducer.budget("question")
    assert all(
        sum(cost(m["content"]) for m in call) + output_budget + 128 <= reducer.window
        for call, output_budget in zip(llm.calls, llm.output_budgets)
    )


async def test_failed_parts_are_reported_after_retries():
    llm = Summarizer(fail=True)
    reducer = DvdContextReducer(llm, retries=1)
    text = "[1] A\n" + "FAIL_SOURCE " * 700 + "\n[2] B\n" + "padding " * 300 + "FACT2"
    result = await reducer.prepare("m", "question", text)
    assert result.failed_parts
    assert any("[1]" in failure for failure in result.failed_parts)
    assert "FACT2" in result.text


async def test_small_context_does_not_call_model():
    llm = Summarizer()
    result = await DvdContextReducer(llm).prepare("m", "question", "[1] source\ntext")
    assert result.text == "[1] source\ntext" and not llm.calls


def test_long_source_header_retains_citation_in_every_part():
    reducer = DvdContextReducer(None)
    parts = reducer._parts("[7] " + "Заголовок " * 300 + "\nFACT7", 600)
    assert len(parts) > 1
    assert all(part.startswith("[7]") and cost(part) <= 600 for part in parts)
    assert "FACT7" in parts[-1]


async def test_fabricated_coverage_is_rejected():
    class Llm:
        async def chat(self, **kwargs):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "evidence": [{"source_id": "[99]", "quotes": ["fake"]}],
                            "complete": True,
                        }
                    )
                }
            }

    reducer = DvdContextReducer(Llm(), retries=0)
    result = await reducer.prepare("m", "question", "[1] A\n" + "text " * 2000)
    assert result.failed_parts and "fake" not in result.text


async def test_cancellation_propagates():
    class Llm:
        async def chat(self, **kwargs):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await DvdContextReducer(Llm()).prepare("m", "q", "x " * 6000)


async def test_summary_requests_structured_output_and_reasoning_budget():
    from unittest.mock import AsyncMock

    llm = AsyncMock()
    llm.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "evidence": [{"source_id": "[1]", "quotes": ["FACT1"]}],
                    "complete": True,
                }
            )
        }
    }
    reducer = DvdContextReducer(llm)
    assert "[1] Doc\nFACT1" in await reducer._summarize("m", "q", "[1] Doc\nFACT1", 800)
    call = llm.chat.call_args.kwargs
    assert call["format"]["properties"]["evidence"]["type"] == "array"
    assert call["options"]["num_predict"] >= 4096
    assert call["think"] is False


async def test_failed_summary_reports_reason_not_generic_value_error():
    class Truncated:
        async def chat(self, **kwargs):
            return {"message": {"content": ""}, "done_reason": "length"}

    result = await DvdContextReducer(Truncated(), retries=0).prepare(
        "m", "q", "[1] Doc\n" + "text " * 1500
    )
    assert result.failed_parts
    assert all("output_truncated" in reason for reason in result.failed_parts)


async def test_retry_tells_model_why_summary_was_rejected():
    class RetrySummarizer(Summarizer):
        async def chat(self, model, messages, **kwargs):
            response = await super().chat(model, messages, **kwargs)
            if len(self.calls) == 1:
                data = json.loads(response["message"]["content"])
                data["complete"] = False
                response["message"]["content"] = json.dumps(data)
            return response

    llm = RetrySummarizer()
    result = await DvdContextReducer(llm, concurrency=1).prepare(
        "m", "q", "[1] Doc\n" + "padding " * 650 + "FACT1"
    )
    assert not result.failed_parts
    assert "incomplete" in llm.calls[1][0]["content"]


async def test_reducer_and_openai_adapter_recover_reasoning_only_completion(
    monkeypatch,
):
    """Exercise the real reducer -> schema translation -> bounded retry seam."""
    from tests.unit.test_llm_adapters import _adapter_with, _Choice, _Completion, _Delta

    monkeypatch.setenv("DVD_SUMMARY_MAX_TOKENS", "1536")
    adapter, _ = _adapter_with(None)
    calls = []

    async def create(**request):
        calls.append(request)
        if request["max_tokens"] <= 1536:
            return _Completion([_Choice(message=_Delta(""), finish_reason="length")])
        assert request["response_format"]["type"] == "json_schema"
        return _Completion(
            [
                _Choice(
                    message=_Delta(
                        json.dumps(
                            {
                                "evidence": [
                                    {
                                        "source_id": "[1]",
                                        "quotes": ["School distance: 500 m."],
                                    }
                                ],
                                "complete": True,
                            }
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        )

    adapter.client.chat.completions.create = create
    summary = await DvdContextReducer(adapter)._summarize(
        "gpt-oss-20b", "School distance?", "[1] Standard\nSchool distance: 500 m.", 1200
    )
    assert "[1] Standard\nSchool distance: 500 m." in summary
    assert len(calls) == 2 and calls[1]["max_tokens"] > calls[0]["max_tokens"]
