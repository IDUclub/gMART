from pydantic import BaseModel, Field


class SimpleRequestDTO(BaseModel):
    """
    Simple Request DTO to LLM.
    Attributes:
        model (str): Model name.
        request (str): Request text.
    """

    model: str = Field(
        default="gpt-oss:20b",
        examples=["gpt-oss:20b"],
        description="Model name to generate request on",
    )
    temperature: float = Field(
        default=1.0,
        examples=[0.75],
        description="Model temperature for pipeline generation.",
    )
    request: str = Field(examples=["Почему небо синее?"], description="Request message")
