"""Unit tests for BaseLlmService.build_llm_history — history compaction and the
trailing-current-question dedup (the current question is persisted before the pipeline
runs, so a reconnect fetches it back as the last history message)."""

from __future__ import annotations

from src.agents.services.base_llm_service import BaseLlmService


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_extracts_text_and_skips_other_roles():
    messages = [
        _msg("system", "internal"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]
    assert BaseLlmService.build_llm_history(messages) == [
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]


def test_drops_trailing_copy_of_current_question():
    messages = [
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
        _msg("user", "Q2"),
    ]
    history = BaseLlmService.build_llm_history(messages, current_user_query="Q2")
    assert history == [_msg("user", "Q1"), _msg("assistant", "A1")]


def test_keeps_matching_question_when_not_trailing():
    # The same text earlier in the chat (followed by an answer) must stay.
    messages = [
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    ]
    history = BaseLlmService.build_llm_history(messages, current_user_query="Q1")
    assert history == [_msg("user", "Q1"), _msg("assistant", "A1")]


def test_no_dedup_without_current_query():
    messages = [_msg("user", "Q1")]
    assert BaseLlmService.build_llm_history(messages) == [_msg("user", "Q1")]


def test_dedup_applies_before_max_messages_slice():
    messages = [_msg("user", f"Q{i}") for i in range(5)] + [_msg("user", "current")]
    history = BaseLlmService.build_llm_history(
        messages, max_messages=3, current_user_query="current"
    )
    assert history == [_msg("user", "Q2"), _msg("user", "Q3"), _msg("user", "Q4")]
