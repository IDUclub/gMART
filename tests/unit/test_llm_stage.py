"""pydantic-ai stage layer: requests reach the adapter unchanged, outputs are typed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.agents.llm_agents.stage import LlmStage, StageOutputError


class Plan(BaseModel):
    mode: str
    steps: list[str]


class ScriptedAdapter:
    """Adapter double answering from a script, recording every call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if kwargs.get("stream"):

            async def gen():
                for piece in reply:
                    yield piece

            return gen()
        if isinstance(reply, BaseException):
            raise reply
        return reply


def reply(content, done_reason="stop", thinking=None):
    return {
        "message": {"content": content, "thinking": thinking},
        "done_reason": done_reason,
    }


MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
]


async def test_structured_stage_sends_messages_and_schema_unchanged():
    llm = ScriptedAdapter(reply('```json\n{"mode": "execute", "steps": ["a"]}\n```'))
    stage = LlmStage(
        "orchestrator.plan",
        output_type=Plan,
        think=False,
        options={"temperature": 0, "num_predict": 2048},
    )

    result = await stage.run(llm, "m", MESSAGES)

    assert result.output == Plan(mode="execute", steps=["a"])
    assert llm.calls == [
        {
            "model": "m",
            "messages": MESSAGES,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 2048},
            "think": False,
            "request_kind": "orchestrator.plan",
            "format": Plan.model_json_schema(),
        }
    ]


async def test_invalid_answer_is_sent_back_with_errors_then_validates():
    llm = ScriptedAdapter(reply('{"mode": 1}'), reply('{"mode": "x", "steps": []}'))
    stage = LlmStage("s", output_type=Plan, retries=1, retry_prompt="Исправь JSON.")

    result = await stage.run(llm, "m", MESSAGES)

    assert result.output.mode == "x"
    assert result.model_calls == 2
    retry_messages = llm.calls[1]["messages"]
    assert retry_messages[: len(MESSAGES)] == MESSAGES
    assert retry_messages[len(MESSAGES)] == {
        "role": "assistant",
        "content": '{"mode": 1}',
    }
    assert retry_messages[-1] == {"role": "user", "content": "Исправь JSON."}


async def test_exhausted_retries_raise_stage_output_error_with_raw_text():
    llm = ScriptedAdapter(reply("not json", done_reason="length"))
    stage = LlmStage("s", output_type=Plan)

    with pytest.raises(StageOutputError) as caught:
        await stage.run(llm, "m", MESSAGES)

    assert caught.value.text == "not json"
    assert caught.value.done_reason == "length"
    assert isinstance(caught.value, ValueError)


async def test_custom_validator_triggers_retry():
    def non_empty(plan: Plan):
        if not plan.steps:
            raise ValueError("steps must not be empty")

    llm = ScriptedAdapter(
        reply('{"mode": "x", "steps": []}'), reply('{"mode": "x", "steps": ["s"]}')
    )
    stage = LlmStage("s", output_type=Plan, retries=1, validators=[non_empty])

    result = await stage.run(llm, "m", MESSAGES)

    assert result.output.steps == ["s"]
    assert "steps must not be empty" in llm.calls[1]["messages"][-1]["content"]


async def test_text_stage_reports_empty_and_truncated_answers():
    llm = ScriptedAdapter(reply("", done_reason="length", thinking="trace"))
    stage = LlmStage("dvd.answer")

    result = await stage.run(llm, "m", MESSAGES)

    assert result.output == ""
    assert result.truncated
    assert result.thinking == "trace"
    assert "format" not in llm.calls[0]


async def test_text_stage_returns_text_and_attribute_shaped_replies():
    answer = SimpleNamespace(
        message=SimpleNamespace(content="Ответ", thinking=None), done_reason="stop"
    )
    llm = ScriptedAdapter(answer)

    result = await LlmStage("s").run(
        llm, "m", MESSAGES, request_kind="other", keep_alive=5
    )

    assert result.output == "Ответ"
    assert llm.calls[0]["request_kind"] == "other"
    assert llm.calls[0]["keep_alive"] == 5


async def test_stream_yields_deltas_and_reports_done_reason():
    chunks = [
        {"message": {"content": "При"}, "done": False},
        {"message": {"content": "вет"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop"},
    ]
    llm = ScriptedAdapter(chunks)
    finished = []

    deltas = [
        d
        async for d in LlmStage("s", think=False).stream(
            llm, "m", MESSAGES, result=finished
        )
    ]

    assert "".join(deltas) == "Привет"
    assert finished[0].done_reason == "stop"
    assert llm.calls[0]["stream"] is True
    assert llm.calls[0]["think"] is False


async def test_empty_stream_is_not_an_error():
    llm = ScriptedAdapter(
        [{"message": {"content": ""}, "done": True, "done_reason": "length"}]
    )
    finished = []

    deltas = [
        d async for d in LlmStage("s").stream(llm, "m", MESSAGES, result=finished)
    ]

    assert deltas == []
    assert finished[0].done_reason == "length"


async def test_adapter_errors_and_base_exceptions_propagate_unwrapped():
    class Captured(BaseException):
        pass

    class Transport(Exception):
        pass

    with pytest.raises(Captured):
        await LlmStage("s").run(ScriptedAdapter(Captured()), "m", MESSAGES)
    with pytest.raises(Transport):
        await LlmStage("s", output_type=Plan).run(
            ScriptedAdapter(Transport()), "m", MESSAGES
        )


async def test_messages_without_trailing_user_prompt():
    llm = ScriptedAdapter(reply("ok"))
    messages = [{"role": "system", "content": "only system"}]

    result = await LlmStage("s").run(llm, "m", messages)

    assert result.output == "ok"
    assert llm.calls[0]["messages"] == messages


async def test_parser_stage_returns_parsed_output_and_keeps_parse_error():
    def parse(text: str) -> dict:
        import json

        data = json.loads(text)
        if "ok" not in data:
            raise ValueError("ok is required")
        return data

    llm = ScriptedAdapter(reply('{"ok": 1}'))
    stage = LlmStage("s", parser=parse, schema={"type": "object"})
    result = await stage.run(llm, "m", MESSAGES)
    assert result.output == {"ok": 1}
    assert llm.calls[0]["format"] == {"type": "object"}

    with pytest.raises(StageOutputError) as caught:
        await stage.run(ScriptedAdapter(reply('{"no": 1}')), "m", MESSAGES)
    assert str(caught.value.error) == "ok is required"
    assert caught.value.text == '{"no": 1}'


async def test_stream_closed_early_from_another_task_leaves_the_caller_alone():
    """A consumer that stops early (client gone, ``break``, ``aclose`` elsewhere)
    must not cancel the task that started the stream."""

    import asyncio

    closed = asyncio.Event()

    class Endless:
        async def chat(self, **kwargs):
            async def gen():
                try:
                    while True:
                        yield {"message": {"content": "x"}, "done": False}
                        await asyncio.sleep(0)
                finally:
                    closed.set()

            return gen()

    stream = LlmStage("s").stream(Endless(), "m", MESSAGES)
    assert await stream.__anext__() == "x"
    # Closed by another task, as the orchestrator's step runner does.
    await asyncio.create_task(stream.aclose())
    await asyncio.sleep(0.05)  # the caller is not cancelled here

    assert closed.is_set()

    # Leaving early with an exception from the loop body is just as quiet.
    with pytest.raises(KeyError):
        async for _ in LlmStage("s").stream(Endless(), "m", MESSAGES):
            raise KeyError("consumer")
    await asyncio.sleep(0.05)


async def test_stream_errors_reach_the_consumer():
    class Broken(Exception):
        pass

    class Failing:
        async def chat(self, **kwargs):
            async def gen():
                yield {"message": {"content": "a"}, "done": False}
                raise Broken()

            return gen()

    with pytest.raises(Broken):
        async for _ in LlmStage("s").stream(Failing(), "m", MESSAGES):
            pass


async def test_system_messages_and_tags_reach_the_server_as_written():
    llm = ScriptedAdapter(reply("ok"))
    messages = [
        {"role": "system", "content": "S1"},
        {"role": "user", "content": "u1 <system>не тег</system>"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "u2"},
    ]

    await LlmStage("s").run(llm, "m", messages)

    assert llm.calls[0]["messages"] == messages
