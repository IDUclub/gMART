"""Adapters carry tool calls both ways, for the ADK orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

from src.agents.model_clients.ollama_adapter import _ollama_messages
from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter


def test_openai_completion_tool_calls_are_kept_in_wire_form():
    completion = SimpleNamespace(
        model="m",
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(
                                name="run_norms", arguments='{"task": "школы"}'
                            ),
                        )
                    ],
                ),
            )
        ],
    )

    response = OpenAiCompatAdapter._as_response(completion)

    assert response.done_reason == "tool_calls"
    assert response.message.tool_calls == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "run_norms", "arguments": '{"task": "школы"}'},
        }
    ]


def test_plain_completion_has_no_tool_calls():
    completion = SimpleNamespace(
        model="m",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="ok"),
            )
        ],
    )

    assert OpenAiCompatAdapter._as_response(completion).message.tool_calls is None


def test_ollama_gets_object_arguments_and_no_tool_call_ids():
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_norms", "arguments": '{"task": "t"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    ]

    spelled = _ollama_messages(messages)

    assert spelled[0] == messages[0]
    assert spelled[1]["tool_calls"] == [
        {"function": {"name": "run_norms", "arguments": {"task": "t"}}}
    ]
    assert spelled[2] == {"role": "tool", "content": "{}"}
