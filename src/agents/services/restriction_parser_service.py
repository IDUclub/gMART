from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from loguru import logger
from ollama import AsyncClient as AsyncOllamaClient
from ollama import ChatResponse

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    StatusPartRequest,
    StatusPayload,
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.restriction_catalog import RestrictionPlanBuilder
from src.agents.services.restriction_context import RestrictionContextBuilder
from src.agents.services.restriction_tool_executor import RestrictionToolExecutor
from src.agents.services.service_entities.restriction_plan import (
    RestrictionPlan,
    RestrictionTaskMode,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


class RestrictionParserService(BaseLlmService):
    """
    Service for running restriction execution pipelines. Inherits from BaseLlmService.
    Attributes:
        host (str): Ollama host.
        chat_storage_client (ChatStorageApiClient)
        llm_client (AsyncOllamaClient): Asynchronous ollama client.
    """

    def __init__(
        self, ollama_host: str, chat_storage_client: ChatStorageApiClient
    ) -> None:
        """
        Initialization function for RestrictionParserService.
        Args:
            ollama_host (str): Ollama host.
            chat_storage_client (ChatStorageApiClient): ChatStorageApiClient instance for saving chat content.
        """

        super().__init__(ollama_host, chat_storage_client)
        self.plan_builder = RestrictionPlanBuilder(self.llm_client)
        self.tool_executor = RestrictionToolExecutor()
        self.context_builder = RestrictionContextBuilder()

    async def run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        temperature: float,
        model: str,
        user_query: str,
        scenario_id: int,
        chat_id: str | None = None,
    ) -> AsyncGenerator:
        token = mcp_client.mcp_client.transport.auth.token.get_secret_value()
        text_buffer: list[str] = []

        async for item in self._run_restriction_execution_pipline(
            mcp_client=mcp_client,
            temperature=temperature,
            model=model,
            user_query=user_query,
            scenario_id=scenario_id,
            chat_id=chat_id,
        ):
            chat_id = self._chat_id_from_storage_event(item) or chat_id
            if item.get("type") == "tool_call":
                await self._flush_text_buffer(
                    token,
                    chat_id,
                    text_buffer,
                    scenario_id=scenario_id,
                )
                await self._add_tool_calls_to_chat(
                    token,
                    chat_id,
                    item.get("content", {}).get("tool_calls", []),
                    execution_mode=item.get("content", {}).get("execution_mode", ""),
                    scenario_id=scenario_id,
                )
                continue

            if item.get("type") == "chunk":
                content = item.get("content", {})
                if content.get("text"):
                    text_buffer.append(content["text"])
                if content.get("done"):
                    await self._flush_text_buffer(
                        token,
                        chat_id,
                        text_buffer,
                        scenario_id=scenario_id,
                    )
                yield item
                continue

            await self._flush_text_buffer(
                token,
                chat_id,
                text_buffer,
                scenario_id=scenario_id,
            )
            await self._add_pipeline_item_to_chat(
                token,
                chat_id,
                item,
                scenario_id=scenario_id,
            )
            yield item

        if text_buffer:
            await self._flush_text_buffer(
                token,
                chat_id,
                text_buffer,
                scenario_id=scenario_id,
            )

    async def _run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        temperature: float,
        model: str,
        user_query: str,
        scenario_id: int,
        chat_id: str | None = None,
    ) -> AsyncGenerator:

        token = mcp_client.mcp_client.transport.auth.token.get_secret_value()
        if not chat_id:
            logger.info(f"No chat id provided in request, creating a new chat.")
            chat_id, title = await self.create_chat(
                token,
                model,
                user_query,
                additional_instructions="""Первый запрос пользователя был отправлен к сервису 
                            создания слоёв с ограничениями ихз запроса пользователя.
                            """,
                scenario_id=scenario_id,
            )
            yield self._chat_created_event(chat_id, title)

        logger.info(
            f"Starting restriction execution for request {user_query} for chat id {chat_id}"
        )

        yield self._status(
            "data_retrievement",
            "Получаю каталоги сервисов и физических объектов",
        )
        plan = await self._build_plan(mcp_client, model, user_query, scenario_id)
        if plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION:
            yield self._status(
                "context_preparation", "Нужно уточнение параметров запроса."
            )
            yield self._chunk(
                plan.clarification_question or "Уточните параметры запроса.", done=True
            )
            return

        yield self._status("plan_explanation", "Объясняю, почему выбраны эти параметры")
        async for chunk in self.generate_plan_explanation(
            model, user_query, plan, temperature
        ):
            yield chunk
        yield self._chunk("\n\n", done=False)

        yield self._status(
            "data_retrievement", "Получаю необходимые слои по утвержденному плану"
        )
        layers = await self.tool_executor.retrieve_layers_for_plan(
            mcp_client,
            plan,
            scenario_id,
        )
        for item in self._feature_collections(layers):
            yield item

        yield self._status(
            "buffer_creation", "Начинаю построение буферов зон с ограничениями"
        )
        buffers_result = await self.tool_executor.run_buffer_plan(
            mcp_client, plan, layers
        )
        yield self._tool_call("buffer_creation", buffers_result.tool_calls)
        yield self._status(
            "buffer_creation", "Построил необходимые буферы с ограничениями."
        )
        for item in self._feature_collections(buffers_result.tool_result):
            yield item

        if plan.mode == RestrictionTaskMode.BUFFERS_ONLY:
            context = await self.context_builder.generate_buffers_context(
                buffers_result.tool_result
            )
        else:
            yield self._status(
                "restriction_formation",
                "Начинаю извлечение нормативных ограничений.",
            )
            restriction_result = await self.tool_executor.run_restriction_plan(
                mcp_client,
                plan,
                layers,
                buffers_result.tool_result,
            )
            yield self._tool_call(
                "restriction_formation", restriction_result.tool_calls
            )
            yield self._status(
                "restriction_formation",
                "Извлечение нормативных ограничений завершено.",
            )
            for item in self._feature_collections(restriction_result.tool_result):
                yield item
            context = await self.context_builder.generate_restrictions_context(
                restriction_result.tool_result["generators"],
                restriction_result.tool_result["objects"],
            )

        async for chunk in self.generate_final_response(
            model, user_query, context, temperature
        ):
            yield chunk

    async def _add_pipeline_item_to_chat(
        self,
        token: str,
        chat_id: str | None,
        item: dict,
        **metadata,
    ) -> None:
        part = self._pipeline_item_to_chat_part(item)
        if not chat_id or part is None:
            return
        await self.add_complex_message(
            token,
            chat_id,
            RoleEnum.ASSISTANT,
            [part],
            **metadata,
        )

    async def _add_text_to_chat(
        self,
        token: str,
        chat_id: str | None,
        text: str,
        **metadata,
    ) -> None:
        if not chat_id or not text:
            return
        await self.add_complex_message(
            token,
            chat_id,
            RoleEnum.ASSISTANT,
            [TextPartRequest(kind="text", payload=TextPayload(text=text))],
            **metadata,
        )

    async def _flush_text_buffer(
        self,
        token: str,
        chat_id: str | None,
        text_buffer: list[str],
        **metadata,
    ) -> None:
        if not text_buffer:
            return
        await self._add_text_to_chat(
            token,
            chat_id,
            "".join(text_buffer),
            **metadata,
        )
        text_buffer.clear()

    async def _add_tool_calls_to_chat(
        self,
        token: str,
        chat_id: str | None,
        tool_calls: list[dict],
        execution_mode: str,
        **metadata,
    ) -> None:
        if not chat_id or not tool_calls:
            return

        calls = [
            self._tool_call_to_chat_storage_call(step, tool_call)
            for step, tool_call in enumerate(tool_calls, start=1)
        ]
        await self.add_complex_message(
            token,
            chat_id,
            RoleEnum.ASSISTANT,
            [
                ToolCallPartRequest(
                    kind="tool_call",
                    payload=ToolCallPayload(
                        execution_mode=execution_mode,
                        calls=calls,
                    ),
                )
            ],
            **metadata,
        )

    @staticmethod
    def _pipeline_item_to_chat_part(
        item: dict,
    ) -> TextPartRequest | StatusPartRequest | None:
        item_type = item.get("type")
        content = item.get("content") or {}

        if item_type == "status":
            return StatusPartRequest(
                kind="status",
                payload=StatusPayload(
                    status=content.get("status", ""),
                    text=content.get("text", ""),
                ),
            )

        if item_type == "chunk":
            text = content.get("text") or ""
            if not text:
                return None
            return TextPartRequest(kind="text", payload=TextPayload(text=text))

        return None

    @staticmethod
    def _tool_call_to_chat_storage_call(step: int, tool_call: dict) -> ToolCall:
        function_call = tool_call.get("function") or {}
        tool_name = (
            tool_call.get("tool_name")
            or tool_call.get("name")
            or function_call.get("name")
        )
        arguments = tool_call.get("arguments") or function_call.get("arguments") or {}
        if not tool_name:
            raise ValueError(f"Tool call without tool name: {tool_call}")
        return ToolCall(step=step, tool_name=tool_name, arguments=arguments)

    @staticmethod
    def _chat_id_from_storage_event(item: dict) -> str | None:
        event_container = item.get("content") or item
        event = event_container.get("event") or {}
        if event.get("storage_event_type") == "chat_created":
            return event.get("chat_id")
        return None

    async def _build_plan(
        self,
        mcp_client: IduMcpClient,
        model: str,
        user_query: str,
        scenario_id: int,
    ) -> RestrictionPlan:
        services_catalog, physical_objects_catalog = (
            await self.plan_builder.get_entity_catalogs(
                mcp_client,
                scenario_id,
            )
        )
        return await self.plan_builder.build_plan(
            model,
            user_query,
            scenario_id,
            services_catalog,
            physical_objects_catalog,
        )

    async def generate_plan_explanation(
        self,
        model: str,
        user_query: str,
        plan: RestrictionPlan,
        temperature: float,
    ) -> AsyncGenerator[dict[str, str | dict[str, str | None | bool]], None]:
        messages = [
            {
                "role": "system",
                "content": f"""Коротко и дружелюбно объясни пользователю, почему для его запроса выбраны такие параметры.
                Пиши обычным человеческим языком, без технических терминов.
                Не упоминай JSON, модель, инструмент, пайплайн, схему, поля или внутренние названия.
                Не спорь с пользователем и не перегружай деталями.
                Объясни:
                - что выбрано как источник построения зон;
                - какой радиус используется и откуда он взят;
                - будут ли строиться только буферы или также ограничения для других объектов;
                - если есть целевые объекты, почему они выбраны.

                Данные для объяснения:
                {self._plan_summary(plan)}
                """,
            },
            {"role": "user", "content": user_query},
        ]
        async for part in await self.llm_client.chat(
            model,
            messages,
            options={"temperature": min(temperature, 0.4)},
            stream=True,
        ):
            part: ChatResponse
            if part.message.content:
                yield self._chunk(part.message.content, done=False)

    async def generate_final_response(
        self,
        model: str,
        user_query: str,
        context: str,
        temperature: float,
    ) -> AsyncGenerator[dict[str, str | dict[str, str | None | bool]], None]:
        messages = [
            {
                "role": "system",
                "content": f"""Дай комментарий к запросу пользователя на основе контекста статистики сгенерированных слоёв.
                Ответ давай только в виде обычного текста. Внимательно анализируй предоставленную в контексте информацию.
                В качестве отсылок используй только названия ограничений.

                Контекст для ответа:

                {context}
                """,
            },
            {"role": "user", "content": user_query},
        ]
        async for part in await self.llm_client.chat(
            model,
            messages,
            options={"temperature": temperature},
            stream=True,
        ):
            part: ChatResponse
            yield self._chunk(part.message.content or "", done=part.done)

    @staticmethod
    def _chat_created_event(chat_id: str, chat_title: str) -> dict:
        return {
            "type": "service_event",
            "content": {
                "event_type": "storage_event",
                "event": {
                    "storage_event_type": "chat_created",
                    "chat_id": chat_id,
                    "chat_title": chat_title,
                },
            },
        }

    @staticmethod
    def _status(status: str, text: str) -> dict:
        return {
            "type": "status",
            "content": {
                "status": status,
                "text": text,
            },
        }

    @staticmethod
    def _chunk(text: str, done: bool) -> dict:
        return {
            "type": "chunk",
            "content": {
                "text": text,
                "done": done,
            },
        }

    @staticmethod
    def _tool_call(execution_mode: str, tool_calls: list[dict]) -> dict:
        return {
            "type": "tool_call",
            "content": {
                "execution_mode": execution_mode,
                "tool_calls": tool_calls,
            },
        }

    @staticmethod
    def _feature_collections(layers: dict[str, dict]):
        for name, feature_collection in layers.items():
            yield {
                "type": "feature_collection",
                "content": {
                    "name": name,
                    "feature_collection": feature_collection,
                },
            }

    @staticmethod
    def _plan_summary(plan: RestrictionPlan) -> dict:
        return {
            "mode": plan.mode.value,
            "sources": [entity.name for entity in plan.source_entities],
            "targets": [entity.name for entity in plan.target_entities],
            "buffers": [
                {
                    "source": rule.source_name,
                    "distance_m": rule.buffer_size,
                    "title": rule.title,
                }
                for rule in plan.buffer_rules
            ],
            "restrictions": [
                {
                    "source": rule.source_name,
                    "targets": rule.target_names,
                    "title": rule.title,
                    "description": rule.description,
                }
                for rule in plan.restriction_rules
            ],
            "selection_reasons": [
                {"step": reason.step, "reason": reason.reason}
                for reason in plan.selection_reasons
            ],
        }
