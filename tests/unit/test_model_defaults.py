"""The default model is read from the connected provider, not hardcoded.

Covers the selection order (explicit env > hint match > first served), the process-wide cache,
and the failure when a provider serves nothing. Regression guard for the class of bug where an
Ollama-style id (``gpt-oss:20b``) reached a vLLM serving ``gpt-oss-20b`` and every agent 404'd.
"""

from __future__ import annotations

import asyncio

import pytest

from src.agents.model_clients import model_defaults
from src.agents.model_clients.model_defaults import (
    DEFAULT_MODEL_ENV,
    DEFAULT_MODEL_HINT_ENV,
    NoModelsAvailable,
    invalidate_default_model,
    resolve_default_model,
)


class FakeAdapter:
    """Minimal stand-in for BaseLlmAdapter: only ``list`` is used here."""

    def __init__(self, models: list[str]):
        self._models = models
        self.list_calls = 0

    async def list(self):
        self.list_calls += 1
        return {"models": [{"model": name, "name": name} for name in self._models]}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)
    monkeypatch.delenv(DEFAULT_MODEL_HINT_ENV, raising=False)
    invalidate_default_model()
    yield
    invalidate_default_model()


class TestSelection:
    @pytest.mark.asyncio
    async def test_prefers_a_gpt_oss_id_over_the_first_served(self):
        adapter = FakeAdapter(["llama3.1:8b", "gpt-oss-20b", "qwen2.5:7b-instruct"])
        assert await resolve_default_model(adapter) == "gpt-oss-20b"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "served",
        ["gpt-oss-20b", "gpt-oss:20b", "openai/gpt-oss-20b", "GPT-OSS-20B"],
        ids=["vllm", "ollama", "namespaced", "uppercase"],
    )
    async def test_hint_matches_every_spelling_of_the_same_family(self, served):
        adapter = FakeAdapter(["mistral:7b", served])
        assert await resolve_default_model(adapter) == served

    @pytest.mark.asyncio
    async def test_falls_back_to_the_first_served_model(self):
        adapter = FakeAdapter(["mistral:7b", "llama3.1:8b"])
        assert await resolve_default_model(adapter) == "mistral:7b"

    @pytest.mark.asyncio
    async def test_hint_is_configurable(self, monkeypatch):
        monkeypatch.setenv(DEFAULT_MODEL_HINT_ENV, "qwen")
        adapter = FakeAdapter(["gpt-oss-20b", "qwen2.5:7b-instruct"])
        assert await resolve_default_model(adapter) == "qwen2.5:7b-instruct"

    @pytest.mark.asyncio
    async def test_an_empty_hint_takes_the_first_served(self, monkeypatch):
        monkeypatch.setenv(DEFAULT_MODEL_HINT_ENV, "")
        adapter = FakeAdapter(["mistral:7b", "gpt-oss-20b"])
        assert await resolve_default_model(adapter) == "mistral:7b"


class TestExplicitOverride:
    @pytest.mark.asyncio
    async def test_env_wins_and_skips_the_provider_call(self, monkeypatch):
        monkeypatch.setenv(DEFAULT_MODEL_ENV, "my-model")
        adapter = FakeAdapter(["gpt-oss-20b"])
        assert await resolve_default_model(adapter) == "my-model"
        assert adapter.list_calls == 0

    @pytest.mark.asyncio
    async def test_env_is_used_even_when_the_provider_does_not_list_it(
        self, monkeypatch
    ):
        """An aliased or hidden id must stay usable; a wrong one 404s at the provider."""
        monkeypatch.setenv(DEFAULT_MODEL_ENV, "hidden-alias")
        adapter = FakeAdapter(["gpt-oss-20b"])
        assert await resolve_default_model(adapter) == "hidden-alias"


class TestCaching:
    @pytest.mark.asyncio
    async def test_the_provider_is_asked_once(self):
        adapter = FakeAdapter(["gpt-oss-20b"])
        for _ in range(3):
            assert await resolve_default_model(adapter) == "gpt-oss-20b"
        assert adapter.list_calls == 1

    @pytest.mark.asyncio
    async def test_concurrent_first_calls_share_one_lookup(self):
        adapter = FakeAdapter(["gpt-oss-20b"])
        results = await asyncio.gather(
            *(resolve_default_model(adapter) for _ in range(5))
        )
        assert results == ["gpt-oss-20b"] * 5
        assert adapter.list_calls == 1

    @pytest.mark.asyncio
    async def test_invalidate_forces_a_fresh_lookup(self):
        adapter = FakeAdapter(["gpt-oss-20b"])
        await resolve_default_model(adapter)
        invalidate_default_model()
        await resolve_default_model(adapter)
        assert adapter.list_calls == 2

    @pytest.mark.asyncio
    async def test_an_expired_cache_is_refreshed(self, monkeypatch):
        adapter = FakeAdapter(["gpt-oss-20b"])
        await resolve_default_model(adapter)
        monkeypatch.setattr(model_defaults, "CACHE_TTL_SECONDS", -1.0)
        await resolve_default_model(adapter)
        assert adapter.list_calls == 2

    @pytest.mark.asyncio
    async def test_a_failed_lookup_is_not_cached(self):
        """A provider that is still loading must not poison the default for good."""

        class FlakyAdapter(FakeAdapter):
            async def list(self):
                self.list_calls += 1
                if self.list_calls == 1:
                    raise RuntimeError("connection refused")
                return await super().list()

        adapter = FlakyAdapter(["gpt-oss-20b"])
        with pytest.raises(RuntimeError):
            await resolve_default_model(adapter)
        assert await resolve_default_model(adapter) == "gpt-oss-20b"


class TestNoModels:
    @pytest.mark.asyncio
    async def test_an_empty_list_raises_instead_of_guessing(self):
        """Guessing would resurface later as a confusing provider-side 404."""
        with pytest.raises(NoModelsAvailable):
            await resolve_default_model(FakeAdapter([]))

    @pytest.mark.asyncio
    async def test_entries_without_a_model_key_are_ignored(self):
        class OddAdapter(FakeAdapter):
            async def list(self):
                self.list_calls += 1
                return {"models": [{"name": "no-model-key"}, {"model": "gpt-oss-20b"}]}

        assert await resolve_default_model(OddAdapter([])) == "gpt-oss-20b"
