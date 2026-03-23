from typing import Literal

from pydantic import BaseModel, Field


class ChunkLlmResponse(BaseModel):

    type: Literal["Text"] = Field(examples=["Text"], description="Chunk response type")
    content: str = Field(examples=["Re"], description="Chunk of llm response")
