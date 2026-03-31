import json
from typing import AsyncGenerator
from dataclasses import is_dataclass, asdict

from loguru import logger

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

        instructions = """Исходя из запроса пользователя получи данные по сервисам для формирования ограничений\n" \
                       \nВыбирай из списка сервисов: школа, поликлиника."
                       Используй именно предложенное сочетание как название сервиса. Ты можешь выбирать несколько
                       сервисов, если они подходят по смыслу.
                       Старайся выбрать как можно больше подходящих по смыслу сервисов.
                       Если исходя из запроса пользователя получение сервисов не требуется, 
                       то не вызывай инструмент, а просто верни сообщение, что извлечение сервисов не требуется
                       и почему оно не требуется.\n"""
        services_prompts = await mcp_client.get_services_example_prompts()
        available_services_prompt = await mcp_client.get_available_services_prompt(scenario_id)
        system_prompt = {"role": "system", "content": instructions + available_services_prompt}
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

        instructions = """Исходя из запроса пользователя получи данные по физическим объектам для формирования ограничений.\n
                       Используй именно предложенное сочетание как название физического объекта. 
                       Ты можешь выбирать несколько физических объектов, если они подходят по смыслу под запрос пользователя.
                       Старайся выбрать как можно больше подходящих по смыслу физических объектов.
                       Если исходя из запроса пользователя получение физических объектов не требуется, 
                       то не вызывай инструмент, а просто верни сообщение.\n
                       """
        physical_objects_prompts = await mcp_client.get_physical_objects_example_prompts()
        available_physical_objects_prompt = await mcp_client.get_available_physical_objects_prompt(scenario_id)
        system_prompt = {"role": "system", "content": instructions + available_physical_objects_prompt}
        urban_service_tools = await mcp_client.get_urban_api_physical_objects_tool()
        meta = {"scenario_id": scenario_id}
        try:
            response = await self.llm_client.chat(
                model = model,
                messages=[
                    system_prompt,
                    *physical_objects_prompts,
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

        instructions = f"""Сгенерируй нужные буферы для запроса пользователя для слоёв c именами: 
        {list(objects.keys())}. Выбирай только из этих типов объектов.\n
        Используй все типы объектов, которые могут относится к объектам, создающим ограничения.
        Не передавай в генерацию объекты, на которые воздействуют ограничения.
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
        physical_objects = await self.run_physical_objects_retrieval(mcp_client, user_prompt, model, scenario_id)
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
        yield {"type": "chunk", "content": {"text": "Слои сформированы.", "done": True}}
