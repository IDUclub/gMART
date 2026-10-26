"""ADK model that runs on gMART's own LLM adapters.

``GmartLlm`` is a :class:`google.adk.models.base_llm.BaseLlm` over
:class:`src.agents.model_clients.llm_base.BaseLlmAdapter`, so an ``LlmAgent``
reaches Ollama or any OpenAI-compatible server (vLLM, a provider) exactly the
way the current services do. Why not ADK's own ``OpenAILlm``:

* it always sends ``strict: true`` and rewrites the schema so every property
  is required, which changes what a local model is asked to produce;
* it has no ``extra_body``, so ``think=False`` cannot be spelled as
  ``chat_template_kwargs`` the way ``openai_adapter`` does;
* the ``google-adk[openai]`` extra pins ``openai<3`` while gMART needs
  ``openai>=3.3.1``.

Per-agent knobs ride on :class:`GmartGenerateContentConfig`, passed as the
agent's ``generate_content_config``.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.services.restriction.restriction_catalog import strip_json_fence

# The instruction the current planner sends after an invalid JSON answer
# (RestrictionPlanBuilder._request_plan); kept word for word for parity.
DEFAULT_JSON_REPAIR_INSTRUCTION = (
    "Твой предыдущий ответ содержит невалидный или неполный JSON. "
    "Верни тот же план целиком в виде валидного JSON без markdown и без пояснений. "
    "Убедись, что JSON полный — все скобки и кавычки закрыты."
)


class GmartGenerateContentConfig(types.GenerateContentConfig):
    """``GenerateContentConfig`` with the knobs gMART's backends understand."""

    think: bool | None = Field(
        default=None, description="Forwarded as the adapter's ``think`` flag."
    )
    backend_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra adapter options, e.g. Ollama's ``num_ctx``.",
    )
    json_repair_attempts: int = Field(
        default=0,
        ge=0,
        description=(
            "How many times an answer that fails the output schema is sent back"
            " with the repair instruction, keeping the whole conversation."
        ),
    )
    json_repair_instruction: str = DEFAULT_JSON_REPAIR_INSTRUCTION


class InvalidStructuredOutput(ValueError):
    """The model kept answering outside the output schema."""


class GmartLlm(BaseLlm):
    """``BaseLlm`` over a gMART ``BaseLlmAdapter`` (or anything chat-compatible)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: Any = Field(
        description="A BaseLlmAdapter: ``await chat(model=..., messages=..., ...)``."
    )

    @property
    def capabilities(self) -> LlmCapabilities:
        # Structured output and tools are never mixed in a single gMART step.
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        config = llm_request.config or types.GenerateContentConfig()
        model = llm_request.model or self.model
        messages = self._messages(llm_request)
        kwargs: dict[str, Any] = {"options": self._options(config)}
        think = getattr(config, "think", None)
        if think is not None:
            kwargs["think"] = think

        schema = self._schema_dict(config.response_schema)
        if schema is not None:
            text = await self._structured(model, messages, schema, config, kwargs)
            yield self._response(text)
            return

        if not stream:
            response = await self.adapter.chat(
                model=model, messages=messages, stream=False, **kwargs
            )
            yield self._response(response["message"]["content"] or "")
            return

        parts: list[str] = []
        async for part in await self.adapter.chat(
            model=model, messages=messages, stream=True, **kwargs
        ):
            text = part.message.content
            if text:
                parts.append(text)
                yield self._response(text, partial=True)
        yield self._response("".join(parts), partial=False)

    async def _structured(
        self,
        model: str,
        messages: list[dict],
        schema: dict,
        config: types.GenerateContentConfig,
        kwargs: dict[str, Any],
    ) -> str:
        validator = config.response_schema
        attempts = getattr(config, "json_repair_attempts", 0)
        instruction = getattr(
            config, "json_repair_instruction", DEFAULT_JSON_REPAIR_INSTRUCTION
        )
        while True:
            response = await self.adapter.chat(
                model=model, messages=messages, format=schema, **kwargs
            )
            content = response["message"]["content"] or ""
            text = strip_json_fence(content)
            try:
                self._validate(validator, text)
                return text
            except (ValidationError, json.JSONDecodeError) as exc:
                if attempts <= 0:
                    raise InvalidStructuredOutput(
                        "Model returned output outside the schema"
                    ) from exc
                attempts -= 1
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": instruction},
                ]

    @staticmethod
    def _validate(schema: Any, text: str) -> None:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema.model_validate_json(text)
        else:
            json.loads(text)

    @staticmethod
    def _schema_dict(schema: Any) -> dict | None:
        if schema is None:
            return None
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
        raise TypeError(f"Unsupported output schema for GmartLlm: {schema!r}")

    @staticmethod
    def _options(config: types.GenerateContentConfig) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if config.temperature is not None:
            options["temperature"] = config.temperature
        if config.top_p is not None:
            options["top_p"] = config.top_p
        if config.max_output_tokens is not None:
            options["num_predict"] = config.max_output_tokens
        options.update(getattr(config, "backend_options", None) or {})
        return options

    @staticmethod
    def _messages(llm_request: LlmRequest) -> list[dict]:
        messages: list[dict] = []
        system = llm_request.config.system_instruction if llm_request.config else None
        if system:
            messages.append({"role": "system", "content": _text(system)})
        for content in llm_request.contents or []:
            role = "assistant" if content.role == "model" else content.role or "user"
            messages.append({"role": role, "content": _text(content)})
        return messages

    @staticmethod
    def _response(text: str, partial: bool | None = None) -> LlmResponse:
        return LlmResponse(
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            partial=partial,
        )


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    parts = getattr(value, "parts", None) or []
    return "".join(part.text or "" for part in parts if not part.thought)
