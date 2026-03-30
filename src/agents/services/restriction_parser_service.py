import json
from typing import AsyncGenerator
from dataclasses import is_dataclass, asdict

from loguru import logger

from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from .base_llm_service import BaseLlmService


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
                       Используй именно предложенное сочетание как название сервиса. 
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

        instructions = """Исходя из запроса пользователя получи данные по физическим объектам для формирования ограничений.
                       \nВыбирай из списка сервисов: школа, поликлиника."
                       Используй именно предложенное сочетание как название сервиса. 
                       Если исходя из запроса пользователя получение физических объектов не требуется, 
                       то не вызывай инструмент, а просто верни сообщение.\n
                       Используй максимально подходящие по смыслу к запросу пользователя физические объекты.\n
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
    ) -> dict[str, dict]:
        """
        Function runs building buffers.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            user_prompt (dict[str, str]): User prompt info.
            model (str): Model name to run generation on.
            objects (dict[str, dict]): Layers with layer name as key and FeatureCollection as value.
        Returns:
            dict[str, dict]:  Dict with layer name as ley and FeatureCollection as value.
        """

        instructions = f"Сгенерируй нужные буферы для запроса пользователя для слоёв c именами: {list(objects.keys())}"
        system_prompt = {"role": "system", "content": instructions}
        create_buffers_tools = await mcp_client.get_create_buffer_tool()
        meta = {"objects": objects}
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=[
                    system_prompt,
                    user_prompt,
                ],
                tools=create_buffers_tools
            )
            if tool_calls := response["message"].get("tool_calls"):
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


    async def run_restriction_execution(
            self,
            mcp_client: IduMcpClient,
            user_prompt: dict[str, str],
            model: str,
            layers: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Function runs building buffers.
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            user_prompt (dict[str, str]): User prompt info.
            model (str): Model name to run generation on.
            layers (dict[str, dict]): Layers with layer name as key and FeatureCollection as value.
        Returns:
            dict[str, dict]:  Dict with layer name as ley and FeatureCollection as value.
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
        Используй все возможные объекты, которые могут относится к этим ограничениям."""
        system_prompt = {"role": "system", "content": instructions}
        create_restriction_tools = await mcp_client.get_create_restriction_tool()
        meta = {"layers": layers}
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=[
                    system_prompt,
                    user_prompt,
                ],
                tools=create_restriction_tools
            )
            if tool_calls := response["message"].get("tool_calls"):
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
        buffers = await self.run_buffer_construction(mcp_client, user_prompt, model, layers)
        yield {
            "type": "status",
            "content": {
                "status": "buffer_creation",
                "text": "Построил необходимые буферы с ограничениями."
            }
        }
        for name, buffer in buffers.items():
            yield {
                "type": "feature_collection",
                "content": {
                    "name": name,
                    "feature_collection": buffer
                }
            }
        layers.update(buffers)
        yield {
            "type": "status",
            "content": {
                "status": "restriction_formation",
                "text": "Начинаю извлечение нормативных ограничений."
            }
        }
        restrictions = await self.run_restriction_execution(mcp_client, user_prompt, model, layers)
        yield {
            "type": "status",
            "content": {
                "status": "restriction_formation",
                "text": "Извлечение нормативных ограничений завершено."
            }
        }
        for name, restriction in restrictions.items():
            yield {
                "type": "feature_collection",
                "content": {
                    "name": name,
                    "feature_collection": restriction
                }
            }
        yield {"type": "chunk", "content": {"text": "Слои сформированы.", "done": True}}
