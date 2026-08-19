from pydantic import BaseModel, Field


class SimpleRequestDTO(BaseModel):
    """
    Simple Request DTO to LLM.
    Attributes:
        model (str | None): Model name; None resolves to the provider's default.
        request (str): Request text.
    """

    model: str | None = Field(
        default=None,
        examples=["gpt-oss-20b"],
        description=(
            "Model name to generate request on. Omit it to use whatever the connected "
            "provider serves — the id is read from its own model list, so it is correct "
            "for both an OpenAI-compatible server and Ollama. GET /llm/available_models "
            "lists the valid values."
        ),
    )
    temperature: float = Field(
        default=1.0,
        examples=[0.75],
        description="Model temperature for pipeline generation.",
    )
    request: str = Field(examples=["Почему небо синее?"], description="Request message")
