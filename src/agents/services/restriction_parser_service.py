from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

from loguru import logger
from ollama import ChatResponse

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
    def __init__(self, ollama_host: str):
        super().__init__(ollama_host)
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
    ) -> AsyncGenerator:
        logger.info(f"Starting restriction execution for request {user_query}")

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
