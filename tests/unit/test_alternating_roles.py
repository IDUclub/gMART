"""Chat turns must alternate before they reach the server.

Gemma's chat template answers anything else with a 400 —
``Conversation roles must alternate user/assistant/user/assistant/...`` — which
takes down the whole pipeline run. Verified against a live vLLM server:
``[system, user, assistant, user]`` passes, while ``[system, assistant, user]``
and two user turns in a row do not.
"""

from __future__ import annotations

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter as A


def test_a_well_formed_conversation_is_untouched():
    messages = [
        {"role": "system", "content": "правила"},
        {"role": "user", "content": "запрос"},
        {"role": "assistant", "content": "невалидный json"},
        {"role": "user", "content": "почини"},
    ]

    assert A._alternating(messages) == messages


def test_repair_pass_without_a_user_turn_is_repaired():
    """The plan repair prompts put everything in the system message and pass no
    user query; the model's answer then follows the system turn directly."""

    out = A._alternating(
        [
            {"role": "system", "content": "почини план"},
            {"role": "assistant", "content": "{"},
            {"role": "user", "content": "верни валидный JSON"},
        ]
    )

    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[0]["content"] == "почини план"


def test_consecutive_user_turns_are_merged():
    """A history ending with the question the caller appends again."""

    out = A._alternating(
        [
            {"role": "system", "content": "правила"},
            {"role": "user", "content": "первый"},
            {"role": "user", "content": "второй"},
        ]
    )

    assert [m["role"] for m in out] == ["system", "user"]
    assert out[1]["content"] == "первый\n\nвторой"


def test_a_lone_system_message_is_left_alone():
    """It is accepted as-is — relabelling it would change nothing and hide the
    prompt's intent."""

    assert A._alternating([{"role": "system", "content": "правила"}]) == [
        {"role": "system", "content": "правила"}
    ]


def test_other_message_fields_survive():
    out = A._alternating(
        [
            {"role": "user", "content": "a", "name": "tool"},
            {"role": "user", "content": "b"},
        ]
    )

    assert out[0]["name"] == "tool"


def test_no_messages():
    assert A._alternating(None) == []
    assert A._alternating([]) == []
