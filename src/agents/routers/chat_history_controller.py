from fastapi import APIRouter

chat_history_router = APIRouter(prefix="/chat_history", tags=["chat_history"])


@chat_history_router.get("/{user_id}/{chat_id}")
async def get_chat(user_id: int, chat_id: int):
    pass


@chat_history_router.get("/{user_id}/chats")
async def get_user_chats(
    user_id: int,
):
    pass


@chat_history_router.delete("{user_id}/{chat_id}")
async def delete_chat(user_id: int, chat_id: int):
    pass
