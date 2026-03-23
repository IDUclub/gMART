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
    request: str = Field(examples=["Почему небо синее?"], description="Request message")
