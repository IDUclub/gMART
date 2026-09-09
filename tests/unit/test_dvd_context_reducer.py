import asyncio
import json
import re

import pytest

from src.agents.services.dvd.context_reducer import DvdContextReducer, cost, split_text


class Summarizer:
    def __init__(self, fail=False):
        self.active = self.peak = 0
        self.calls = []
        self.fail = fail

    async def chat(self, model, messages, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.calls.append(messages)
        try:
            await asyncio.sleep(0.002)
            source = (
                messages[1]["content"]
                .split("Текст:\n", 1)[1]
                .split("\nПроверь черновую", 1)[0]
            )
            if self.fail and "FAIL_SOURCE" in source:
                raise RuntimeError("temporary failure")
            ids = sorted(set(re.findall(r"\[\d+\]", source)))
            facts = sorted(set(re.findall(r"FACT\d+", source)))
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": " ".join(ids + facts),
                            "covered_sources": ids,
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
        sum(cost(m["content"]) for m in call) + reducer.output_tokens + 128
        <= reducer.window
        for call in llm.calls
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
                            "summary": "[99] fake",
                            "covered_sources": ["[99]"],
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
