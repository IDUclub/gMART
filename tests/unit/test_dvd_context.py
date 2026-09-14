"""Unit tests for DvdContextBuilder — formatting IDU_DVD hits into a cited LLM context."""

from __future__ import annotations

from src.agents.services.dvd.dvd_context import DvdContextBuilder


def test_empty_hits_returns_empty_string():
    assert DvdContextBuilder().build_context([]) == ""


def test_hit_is_numbered_and_cited():
    ctx = DvdContextBuilder().build_context(
        [
            {
                "name": "СП 42",
                "version": "ред. 2018",
                "numbering": "7.5",
                "breadcrumb": "Раздел 7",
                "text": "Норма.",
            }
        ]
    )
    assert ctx.startswith("[1] СП 42")
    assert "ред. 2018" in ctx
    assert "п. 7.5" in ctx
    assert "Раздел 7" in ctx
    assert "Норма." in ctx


def test_multiple_hits_numbered_sequentially():
    ctx = DvdContextBuilder().build_context(
        [{"name": "A", "text": "one"}, {"name": "B", "text": "two"}]
    )
    assert "[1] A" in ctx and "[2] B" in ctx


def test_table_html_preferred_over_text():
    ctx = DvdContextBuilder().build_context(
        [
            {
                "name": "A",
                "table_html": "<table>x</table>",
                "context": "ctx",
                "text": "txt",
            }
        ]
    )
    assert "<table>x</table>" in ctx
    assert "txt" not in ctx


def test_context_keeps_target_even_if_upstream_context_omits_it():
    ctx = DvdContextBuilder().build_context(
        [{"name": "A", "context": "expanded", "text": "raw"}]
    )
    assert "expanded" in ctx and "raw" in ctx


def test_long_body_is_preserved_for_bounded_context_processing():
    long = "ё" * 2000
    ctx = DvdContextBuilder().build_context([{"name": "A", "text": long}])
    assert long in ctx


def test_long_preceding_context_cannot_hide_target():
    target = "3.3 Полный текст определения."
    context = "Предыдущий пункт. " * 200 + target
    result = DvdContextBuilder().build_context(
        [{"name": "СП", "numbering": "3.3", "text": target, "context": context}]
    )
    assert target in result


def test_missing_name_has_fallback():
    ctx = DvdContextBuilder().build_context([{"text": "body"}])
    assert "[1]" in ctx and "body" in ctx
