"""GmartLlm: what an ADK model request becomes on a gMART adapter."""

from __future__ import annotations

import json

import pytest
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

from research.adk_pilot.gmart_adk.llm import (
    GmartGenerateContentConfig,
    GmartLlm,
    InvalidStructuredOutput,
)
from research.adk_pilot.tests.fakes import RecordingLlm


class Answer(BaseModel):
    value: int
    note: str | None = None


def _request(
    config: types.GenerateContentConfig, model: str | None = "m"
) -> LlmRequest:
    return LlmRequest(
        model=model,
        config=config,
        contents=[
            types.Content(role="user", parts=[types.Part(text="вопрос")]),
            types.Content(role="model", parts=[types.Part(text="ответ")]),
            types.Content(role="user", parts=[types.Part(text="ещё")]),
        ],
    )


async def _collect(llm: GmartLlm, request: LlmRequest, stream: bool = False):
    return [r async for r in llm.generate_content_async(request, stream=stream)]


async def test_options_think_and_roles():
    adapter = RecordingLlm()
    adapter.answer_texts = ["привет"]
    config = GmartGenerateContentConfig(
        system_instruction="система",
        temperature=0.3,
        max_output_tokens=256,
        think=False,
        backend_options={"num_ctx": 8192},
    )
    await _collect(
        GmartLlm(model="fallback", adapter=adapter), _request(config), stream=True
    )

    (request,) = adapter.requests()
    assert request["model"] == "m"
    assert request["think"] is False
    assert request["options"] == {
        "temperature": 0.3,
        "num_predict": 256,
        "num_ctx": 8192,
    }
    assert [m["role"] for m in request["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert request["messages"][0]["content"] == "система"


async def test_model_falls_back_to_the_agents_model():
    adapter = RecordingLlm()
    adapter.answer_texts = ["x"]
    request = _request(types.GenerateContentConfig(), model=None)
    await _collect(GmartLlm(model="fallback", adapter=adapter), request, stream=True)
    assert adapter.requests()[0]["model"] == "fallback"


async def test_streaming_yields_partials_then_the_aggregate():
    adapter = RecordingLlm()
    adapter.answer_texts = ["Буфер 100 м"]
    responses = await _collect(
        GmartLlm(model="m", adapter=adapter),
        _request(types.GenerateContentConfig()),
        stream=True,
    )
    assert [r.partial for r in responses] == [True, True, False]
    assert responses[-1].content.parts[0].text == "Буфер 100 м"


async def test_structured_output_sends_the_plain_pydantic_schema():
    adapter = RecordingLlm()
    adapter.json_responses = ['```json\n{"value": 3}\n```']
    config = GmartGenerateContentConfig(response_schema=Answer)
    (response,) = await _collect(
        GmartLlm(model="m", adapter=adapter), _request(config), stream=True
    )

    request = adapter.requests()[0]
    # Not rewritten to OpenAI strict mode: optional fields stay optional.
    assert request["format"] == Answer.model_json_schema()
    assert request["format"]["required"] == ["value"]
    assert request["stream"] is False
    assert json.loads(response.content.parts[0].text) == {"value": 3}


async def test_json_repair_keeps_the_conversation():
    adapter = RecordingLlm()
    adapter.json_responses = ["{broken", '{"value": 1}']
    config = GmartGenerateContentConfig(response_schema=Answer, json_repair_attempts=1)
    await _collect(GmartLlm(model="m", adapter=adapter), _request(config))

    first, retry = adapter.requests()
    assert retry["messages"][: len(first["messages"])] == first["messages"]
    assert retry["messages"][-2] == {"role": "assistant", "content": "{broken"}
    assert "невалидный" in retry["messages"][-1]["content"]


async def test_json_repair_gives_up_after_its_attempts():
    adapter = RecordingLlm()
    adapter.json_responses = ["{broken", "{still broken"]
    config = GmartGenerateContentConfig(response_schema=Answer, json_repair_attempts=1)
    with pytest.raises(InvalidStructuredOutput):
        await _collect(GmartLlm(model="m", adapter=adapter), _request(config))
    assert len(adapter.requests()) == 2
