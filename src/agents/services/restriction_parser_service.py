import json
from typing import AsyncGenerator
from dataclasses import is_dataclass, asdict

import pandas as pd
from loguru import logger
import geopandas as gpd

from ollama import ChatResponse

from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from .base_llm_service import BaseLlmService
from .service_entities import GeometryToolCallResult


#TODO add planning system
#TODO add full streaming for all llm responses
#TODO add union tool execution function
class RestrictionParserService(BaseLlmService):

    def __init__(self, ollama_host: str):
        """
        Initialization function for SimpleLlmService. Inherits from BaseLlmService.
        Args:
            ollama_host (str): Ollama host.
        """

        super().__init__(ollama_host)

    @staticmethod
    async def execute_tool(mcp_client: IduMcpClient, tool_call: dict, meta: dict) -> dict[str, dict]:
        """
        Function executes  tool and returns possible result.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            tool_call (dict): Dict mcp tool call info computable with Ollama.
            meta (dict): Metadata for tool.
        Returns:
            dict[str, dict]: Dict with name - FeatureCollection info.
        """

        tool_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        return await mcp_client.execute_tool(tool_name, args, meta=meta)

    async def execute_one_response_tool(self, mcp_client: IduMcpClient, tool_call: dict, meta: dict) -> dict[str, dict]:
        """
        Function executes urban api tool and returns possible result
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            tool_call (dict): Dict mcp tool call info computable with Ollama.
            meta (dict): Metadata for tool.
        Returns:
            dict[str, dict]: Dict with name - FeatureCollection info.
        """

        result = await self.execute_tool(mcp_client, tool_call, meta)
        if isinstance(result, dict):
            for key, value in result.items():
                if is_dataclass(value):
                    result[key] = asdict(value)
        return result

    async def execute_restriction_tool(self, mcp_client: IduMcpClient, tool_call: dict, meta: dict) -> dict[str | dict]:
        """
        Function executes restriction formation tool and returns possible result.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            tool_call (dict): Dict mcp tool call info computable with Ollama.
            meta (dict): Metadata for tool.
        Returns:
            dict[str, dict]: Dict with name - FeatureCollection info.
        """

        result = await self.execute_tool(mcp_client, tool_call, meta)
        objects = json.loads(result.content[0].text)[0]
        generators = json.loads(result.content[0].text)[1]
        return {
            "objects": objects,
            "generators": generators
        }

    #TODO enhance data getter to async gather pipeline
    async def run_services_retrieval(
            self,
            mcp_client: IduMcpClient,
            user_prompt: dict[str, str],
            model: str,
            scenario_id: int,
    ) -> dict[str, dict] | str:
        """
        Function runs service data extraction from Urban API.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            user_prompt (dict[str, str]): User message info.
            model (str): Model name to run extraction on.
            scenario_id (int): Scenario ID from Urban API.
        Returns:
            dict[str, dict] | str: Either dict with layer name as ley and FeatureCollection as value,
            message with info why no service was retrieved.
        """

        form_request_instructions = """Ты должен определить, какие сервисы из предложенного списка необходимы для 
            обработки запроса пользователя.

            Тебе будут переданы:
            - запрос пользователя;
            - список сервисов.
            
            Правила:
            1. Проанализируй каждый сервис из списка отдельно.
            2. Если сервис необходим для дальнейшей обработки запроса, верни для него true.
            3. Если сервис не нужен, верни false.
            4. Используй только сервисы из переданного списка.
            5. Нельзя добавлять новые сервисы.
            6. Нельзя удалять сервисы из ответа.
            7. Нельзя переименовывать сервисы.
            8. Для каждого сервиса нужно вернуть только булево значение: true или false.
            9. Не выбирай сервис только из-за тематической близости слов.
            10. Выбирай объект только если он действительно нужен для выполнения запроса.
            
            Верни строго JSON-объект следующего вида:
            {
              "объект_1": true,
              "объект_2": false
            }
            \n
            """
        available_services_prompt = await mcp_client.get_available_services_prompt(scenario_id)
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": form_request_instructions + available_services_prompt
                    },
                    user_prompt
                ],
            )
        except Exception as e:
            logger.exception(e)
            raise
        instructions = f"""Извлеки инструмент с необходимыми сервисами. 
        Если true, то сервис нужен, если false, то сервис не нужен.
        Сервисы:
        {response.message.content}
        """
        services_prompts = await mcp_client.get_services_example_prompts()
        system_prompt = {"role": "system", "content": instructions}
        urban_service_tools = await mcp_client.get_urban_api_service_tool()
        meta = {"scenario_id": scenario_id}
        try:
            response = await self.llm_client.chat(
                model = model,
                messages=[
                    system_prompt,
                    *services_prompts,
                    user_prompt,
                ],
                tools=urban_service_tools
            )
            if tool_calls:=response["message"].get("tool_calls"):
                return await self.execute_one_response_tool(
                    mcp_client,
                    tool_calls[0],
                    meta
                )
            return response["message"]["content"]
        except Exception as e:
            logger.exception(e)
            raise

    async def run_physical_objects_retrieval(
            self,
            mcp_client: IduMcpClient,
            user_prompt: dict[str, str],
            model: str,
            scenario_id: int,
    ) -> str | dict:
        """
        Function runs physical objects data extraction from Urban API.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            user_prompt (dict[str, str]): User message info.
            model (str): Model name to run extraction on.
            scenario_id (int): Scenario ID from Urban API.
        Returns:
            dict[str, dict] | str: Either dict with layer name as ley and FeatureCollection as value,
            message with info why no physical objects was retrieved.
        """

        form_request_instructions = """Ты должен определить, какие физические объекты из предложенного списка необходимы для обработки запроса пользователя.

            Тебе будут переданы:
            - запрос пользователя;
            - список физических объектов.
            
            Правила:
            1. Проанализируй каждый объект из списка отдельно.
            2. Если объект необходим для дальнейшей обработки запроса, верни для него true.
            3. Если объект не нужен, верни false.
            4. Используй только объекты из переданного списка.
            5. Нельзя добавлять новые объекты.
            6. Нельзя удалять объекты из ответа.
            7. Нельзя переименовывать объекты.
            8. Для каждого объекта нужно вернуть только булево значение: true или false.
            9. Не выбирай объект только из-за тематической близости слов.
            10. Выбирай объект только если он действительно нужен для выполнения запроса.
            
            Верни строго JSON-объект следующего вида:
            {
              "объект_1": true,
              "объект_2": false
            }
            \n
            """
        available_physical_objects_prompt = await mcp_client.get_available_physical_objects_prompt(scenario_id)
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": form_request_instructions + available_physical_objects_prompt
                    },
                    user_prompt
                ],
            )
        except Exception as e:
            logger.exception(e)
            raise
        instructions = f"""Извлеки инструмент с необходимыми физическими объектами. 
        Если true, то физический объект нужен, если false, то физический объект не нужен.
        Физические объекты:
        {response.message.content}
        """
        # physical_objects_prompts = await mcp_client.get_physical_objects_example_prompts()
        system_prompt = {"role": "system", "content": instructions}
        urban_service_tools = await mcp_client.get_urban_api_physical_objects_tool()
        meta = {"scenario_id": scenario_id}
        try:
            response = await self.llm_client.chat(
                model = model,
                messages=[
                    system_prompt,
                    # *physical_objects_prompts,
                    user_prompt,
                ],
                tools=urban_service_tools
            )
            if tool_calls:=response["message"].get("tool_calls"):
                return await self.execute_one_response_tool(
                    mcp_client,
                    tool_calls[0],
                    meta
                )
            else:
                return response["message"]["content"]
        except Exception as e:
            logger.exception(e)
            raise

    async def run_buffer_construction(
            self,
            mcp_client: IduMcpClient,
            user_prompt: dict[str, str],
            model: str,
            objects: dict[str, dict]
    ) -> GeometryToolCallResult | str:
        """
        Function runs building buffers.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            user_prompt (dict[str, str]): User prompt info.
            model (str): Model name to run generation on.
            objects (dict[str, dict]): Layers with layer name as key and FeatureCollection as value.
        Returns:
            GeometryToolCallResult | str:  Instance of GeometryToolCallResult with first tool_result as a result from
            tool call as dict with name as keys and values as FeatureCollections, tool_calls as list on dict with info
            about provided params to tool call and third messages as formed messages to LLM.
            If no tool was called returns explanation why."""

        instructions = f"""Определи, какие слои нужно передать в инструмент построения буферов ограничений.

        Выбирай только слои-источники ограничения — то есть объекты, от которых должна строиться буферная зона.
        Не выбирай слои, которые являются объектами воздействия ограничения.
        
        Правила:
        - используй только названия из списка допустимых слоёв;
        - не добавляй новые названия;
        - не переименовывай названия;
        - не включай слой, если он не является источником ограничения;
        - если подходящих слоёв нет, верни пустой список;
        - ответ должен быть пригоден для прямой передачи в инструмент.
        
        Допустимые слои:
        {list(objects.keys())}
        
        Обязательно вызови соответствующий инструмент.
        """
        system_prompt = {"role": "system", "content": instructions}
        create_buffers_tools = await mcp_client.get_create_buffer_tool()
        meta = {"objects": objects}
        messages = [system_prompt, user_prompt,]
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=messages,
                tools=create_buffers_tools
            )
            if tool_calls := response["message"].get("tool_calls"):
                tool_result = await self.execute_one_response_tool(
                    mcp_client,
                    tool_calls[0],
                    meta
                )
                return GeometryToolCallResult(tool_result, tool_calls, messages)
            else:
                return response["message"]["content"]
        except Exception as e:
            logger.exception(e)
            raise

    async def explain_tool_call(self, model: str, tool_call: list, query: list[dict]) -> str:
        """
        Function executes llm explanation for called tool.
        Args:
            model (str): Model name to run generation on.
            tool_call (list): Info about called tools.
            query (list[dict]): Info with provided query for llm.
        Returns:
            str: explanation why tool was called with this params.
        """

        instructions = f"""Объясни, почему ты выбрал именно такие параметры:{tool_call}
        Для следующего запроса: {query}.
        Объясни только логику выбора.
        """

        result = await self.llm_client.chat(
            model=model,
            messages=[{"role": "system", "content": instructions}]
        )
        return result["message"]["content"]

    async def run_restriction_execution(
            self,
            mcp_client: IduMcpClient,
            user_prompt: dict[str, str],
            model: str,
            layers: dict[str, dict]
    ) -> GeometryToolCallResult | str:
        """
        Function runs building buffers.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            user_prompt (dict[str, str]): User prompt info.
            model (str): Model name to run generation on.
            layers (dict[str, dict]): Layers with layer name as key and FeatureCollection as value.
        Returns:
            [dict[str, dict] | str, list[dict]] | str:  Tuple with first value as a result value as dict with
            name as keys and values as FeatureCollections and third value as formed messages to LLM.
            If no tool was called returns explanation why.
        """

        final_names = []
        original_layers_names = [i for i in layers]
        for i in original_layers_names:
            if i.islower():
                final_names.append(i)
            else:
                if i.lower() in original_layers_names:
                    continue
                else:
                    final_names.append(i)
        instructions = f"""
        Буферы необходимых слоёв были сгенерированы были сгенерированы. 
        Сгенерируй ограничения исходя из запроса пользователя.
        Название функции передавай точно также, как оно определeно.
        Доступные названия слоёв для формирования ограничений: {final_names}. 
        Используй эти названия именно так как они указаны как ключ для создания ограничений именно в том виде, в котором они представлены. 
        Используй все возможные объекты, которые могут относится к этим ограничениям.
        Если в запросе пользователя нет прямого указания на отношения объектов (какие объекты оказывают 
        какие ограничения на какие объекты), то вместо вызова инструмента верни пустой ответ.
        """
        system_prompt = {"role": "system", "content": instructions}
        create_restriction_tools = await mcp_client.get_create_restriction_tool()
        messages = [system_prompt, user_prompt,]
        meta = {"layers": layers}
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=messages,
                tools=create_restriction_tools
            )
            if tool_calls := response["message"].get("tool_calls"):
                tool_result = await self.execute_one_response_tool(
                    mcp_client,
                    tool_calls[0],
                    meta
                )
                return GeometryToolCallResult(tool_result, tool_calls, messages)
            else:
                return response["message"]["content"]
        except Exception as e:
            logger.exception(e)
            raise

    @staticmethod
    async def generate_generators_summary(generators: gpd.GeoDataFrame) -> str:
        """
        Function forms summary for generators GeoDataFrame. Generators should be passed in local crs.
        Args:
            generators (gpd.GeoDataFrame): Layer with objects containing attributes "name", "restriction_title",
            "geometry".
        Returns:
            str: json repr of context as json str
        """

        generators["area"] = generators.area
        generators["num"] = 1
        return json.dumps(
            generators.groupby("name", as_index=False).agg(
                {
                    "name": "first",
                    "area": "sum",
                    "num": "sum"
                }
            ).rename(
                columns={
                    "name": "Название",
                    "area": "Площадь кв.м",
                    "num": "Количество"
                }
            ).to_dict(orient="records")
        )

    @staticmethod
    async def generate_objects_summary(objects: gpd.GeoDataFrame) -> str:
        """
        Function forms summary for objects GeoDataFrame. Objects should be passed in local crs.
        Args:
            objects (gpd.GeoDataFrame): Layer with objects containing attributes "restriction_name",
            "restriction_description", "geometry".
        Returns:
            str: json repr of context as json str
        """

        objects["area"] = objects.area
        objects["num"] = 1
        return json.dumps(
            objects.groupby("restriction_name", as_index=False).agg(
                {
                    "restriction_description": "first",
                    "area": "sum",
                    "num": "sum"
                }
            ).rename(
                columns={
                    "restriction_name": "Наименование ограничения",
                    "restriction_description": "Описание ограничения",
                    "area": "Площадь объектов кв.м",
                    "num": "Количество объектов"
                }
            ).to_dict(orient="records")
        )

    async def generate_buffers_context(self, buffers: dict) -> str:
        """
        Function forms context for response based only on generated buffers.
        Args:
            buffers (dict): Dict with key as buffer layer name and FeatureCollection as value.
        Returns:
            str: formed context for generated buffers.
        """

        gdf_list = []
        for name, buffer in buffers.items():
            current_buffer_gdf = gpd.GeoDataFrame.from_features(buffer, crs=4326)
            current_buffer_gdf["name"] = name
            gdf_list.append(current_buffer_gdf)
        buffers_gdf = pd.concat(gdf_list)
        buffers_summary = await self.generate_generators_summary(buffers_gdf.to_crs(buffers_gdf.estimate_utm_crs()))
        return f"""Сводная информация по сгенерированным буферам ограничений: 
        \n{buffers_summary}
        """

    async def generate_restrictions_context(self, generators: dict, objects: dict) -> str:
        """
        Function generates context for restriction llm response from generated layers.
        Args:
            generators (dict): Layer of generating restriction geometries as FeatureCollection.
            objects (dict): Layer of objects restricted by generators objects.
        Returns:
            str: Stats from provided layers
        """
        if generators["features"]:
            generators_gdf = gpd.GeoDataFrame.from_features(generators, crs=4326)
            generators_summary = await self.generate_generators_summary(generators_gdf.to_crs(generators_gdf.estimate_utm_crs()))
        else:
            generators_summary = ""
        if objects["features"]:
            objects_gdf = gpd.GeoDataFrame.from_features(objects, crs=4326)
            objects_summary = await self.generate_objects_summary(objects_gdf.to_crs(generators_gdf.estimate_utm_crs()))
        else:
            objects_summary = ""
        return f"""Сводная информация по сформированным ограничениям:\n
        Генераторы ограничений:
        \n{generators_summary}
        \nОбъекты, подверженные ограничениям:
        \n{objects_summary}"""

    async def generate_final_response(self,model: str, user_query: str, context: str) -> AsyncGenerator[dict[str, str], None]:
        """
        Generate final response for user request based on generated context.
        Args:
            model (str): Model name to run generation on.
            user_query (str): Original user_request.
            context (str): Generated context for user request based on tools results.
        Returns:
            AsyncGenerator[dict[str, Any], None]: generator for chunks from ollama api.
        """

        messages = [
            {
                "role": "system",
                "content": f"Дай комментарий к запросу пользователя на основе контекста статистики сгенерированных слоёв"
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
        async for part in await self.llm_client.chat(model, messages, stream=True):
            part: ChatResponse
            yield {
                "type": "chunk",
                "content": {
                    "text": part.message.content,
                    "done": part.done
                }
            }

    async def run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        model: str,
        user_query: str,
        scenario_id: int,
    ) -> AsyncGenerator:
        """
        Run pipline fo forming restrictions.
        Returns:
            AsyncGenerator
        """

        user_prompt = {"role": "user", "content": user_query}
        yield {
            "type": "status",
            "content": {
                "status": "data_retrievement",
                "text": "Получаю необходимые сервисы и физические объекты"
            }
        }
        services = await self.run_services_retrieval(mcp_client, user_prompt, model, scenario_id)
        if isinstance(services, dict):
            yield {
                "type": "status",
                "content": {
                    "status": "data_retrievement",
                    "text": "Были получены следующие сервисы."
                }
            }
            for name, service_layer in services.items():
                yield {
                    "type": "feature_collection",
                    "content": {
                        "name": name,
                        "feature_collection": service_layer
                    },
                }
        physical_objects = await self.run_physical_objects_retrieval(mcp_client, user_prompt, model, scenario_id)
        if isinstance(physical_objects, dict):
            yield {
                "type": "status",
                "content": {
                    "status": "data_retrievement",
                    "text": "Были получены следующие физические объекты."
                }
            }
            for name, physical_object_layer in physical_objects.items():
                yield {
                    "type": "feature_collection",
                    "content": {
                        "name": name,
                        "feature_collection": physical_object_layer
                    },
                }
        layers = {
            **(services if isinstance(services, dict) else {}),
            **(physical_objects if isinstance(physical_objects, dict) else {}),
        }
        yield {
            "type": "status",
            "content": {
                "status": "data_retrievement",
                "text": f"Получил необходимые сервисы и физические объекты {list(layers.keys())}"
            }
        }
        yield {
            "type": "status",
            "content": {
                "status": "buffer_creation",
                "text": "Начинаю построение буферов зон с ограничениями"
            }
        }
        buffers_result: GeometryToolCallResult = await self.run_buffer_construction(mcp_client, user_prompt, model, layers)
        yield {
            "type": "status",
            "content": {
                "status": "buffer_creation",
                "text": "Построил необходимые буферы с ограничениями."
            }
        }
        explanation = await self.explain_tool_call(model, buffers_result.tool_calls, buffers_result.messages)
        yield {
            "type": "chunk",
            "content": {
                "text": explanation,
                "done": False
            }
        }
        for name, buffer in buffers_result.tool_result.items():
            yield {
                "type": "feature_collection",
                "content": {
                    "name": name,
                    "feature_collection": buffer
                }
            }
        layers.update(buffers_result.tool_result)
        yield {
            "type": "status",
            "content": {
                "status": "restriction_formation",
                "text": "Начинаю извлечение нормативных ограничений."
            }
        }
        restriction_result: GeometryToolCallResult | str = await self.run_restriction_execution(mcp_client, user_prompt, model, layers)
        if isinstance(restriction_result, GeometryToolCallResult):
            explanation = await self.explain_tool_call(model ,restriction_result.tool_calls, restriction_result.messages)
            yield {
                "type": "status",
                "content": {
                    "status": "restriction_formation",
                    "text": "Извлечение нормативных ограничений завершено."
                }
            }
            yield {
                "type": "chunk",
                "content": {
                    "text": explanation,
                    "done": False
                }
            }
            for name, restriction in restriction_result.tool_result.items():
                yield {
                    "type": "feature_collection",
                    "content": {
                        "name": name,
                        "feature_collection": restriction
                    }
                }
            generators = restriction_result.tool_result["generators"]
            objects = restriction_result.tool_result["objects"]
            context = await self.generate_restrictions_context(generators, objects)
        else:
            context = await self.generate_buffers_context(buffers_result.tool_result)
        async for chunk in self.generate_final_response(model, user_query, context):
            yield chunk
