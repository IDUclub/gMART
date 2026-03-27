from typing import Annotated, AsyncIterable

from fastapi import APIRouter, Depends, Query
from fastapi.sse import EventSourceResponse

from src.agents.dependencies.dependencies import get_simple_llm_service
from src.agents.dto.llm_request_dto import SimpleRequestDTO
from src.agents.schema.chunk_llm_response import ChunkLlmResponse
from src.agents.services.simple_llm_service import SimpleLlmService

llm_router = APIRouter(prefix="/llm", tags=["simple_llm"])


@llm_router.get("/available_models", response_model=list[str])
async def get_list_of_available_models(
    only_active: bool = Query(
        default=False,
        examples=[True, False],
        description="Weather to return loaded in server vram",
    ),
    simple_llm_service: SimpleLlmService = Depends(get_simple_llm_service),
) -> list[str]:

    return await simple_llm_service.get_models(only_active)


@llm_router.get("/message")
async def get_llm_message(
    request_data: Annotated[SimpleRequestDTO, Depends(SimpleRequestDTO)],
    simple_llm_service: SimpleLlmService = Depends(get_simple_llm_service),
):

    return await simple_llm_service.generate_message(
        request_data.request,
        request_data.model,
    )


@llm_router.get("/message/stream", response_class=EventSourceResponse)
async def get_llm_stream_message(
    request_data: Annotated[SimpleRequestDTO, Depends(SimpleRequestDTO)],
    simple_llm_service: SimpleLlmService = Depends(get_simple_llm_service),
) -> AsyncIterable[ChunkLlmResponse]:

    async for chunk in simple_llm_service.generate_stream_message(
        request_data.request,
        request_data.model,
    ):
        yield ChunkLlmResponse(**chunk)


@llm_router.get("/chat")
async def get_llm_chat():
    pass


@llm_router.get("/chat/stream")
async def get_llm_stream_chat():
    pass
