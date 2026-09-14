from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.encoders import jsonable_encoder

from src.agents.model_clients.llm_base import LlmChatResponse, LlmMessage
from src.agents.services.simple_llm_service import SimpleLlmService


async def test_rest_payload_does_not_serialize_provider_usage_models(monkeypatch):
    class ProviderUsage(SimpleNamespace):
        def model_dump(self, **kwargs):
            raise TypeError("Provider serializer unavailable")

    response = LlmChatResponse(
        model="test",
        message=LlmMessage(content="Grounded answer"),
        usage=ProviderUsage(prompt_tokens=12, completion_tokens=7, total_tokens=19),
    )
    service = object.__new__(SimpleLlmService)
    service.llm_client = object()
    service.resolve_model = AsyncMock(return_value="test")
    service.validate_model = AsyncMock()
    monkeypatch.setattr(
        "src.agents.services.simple_llm_service.run_completion",
        AsyncMock(return_value=response),
    )
    payload = jsonable_encoder(await service.generate_message("Question", "test"))
    assert payload["message"]["content"] == "Grounded answer"
    assert payload["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    }
