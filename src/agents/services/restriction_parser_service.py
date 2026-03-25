import json
from typing import AsyncGenerator

from loguru import logger
from pyexpat import features

from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from .base_llm_service import BaseLlmService


#TODO add full streaming for all llm responses
class RestrictionParserService(BaseLlmService):

    def __init__(self, ollama_host: str):
        """
        Initialization function for SimpleLlmService. Inherits from BaseService.
        Args:
            ollama_host (str): Ollama host.
        """

        super().__init__(ollama_host)

    @staticmethod
    async def execute_urban_api_tool(mcp_client: IduMcpClient, tool_call: dict, scenario_id: int) -> dict[str | dict]:
        """
        Function executes urban api tool and returns possible result
        Args:
            mcp_client (IduMcpClient): IduMcpClient instance.
            tool_call (dict): Dict mcp tool call info computable with Ollama.
            scenario_id (int): Scenario ID from Urban API.
        Returns:
            dict[str | dict]: dict with name - FeatureCollection info.
        """

        tool_name = tool_call["function"]["name"]
        meta = {"scenario_id": scenario_id}
        args = tool_call["function"]["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        result = await mcp_client.execute_tool(tool_name, args, meta=meta)
        dict_res = json.loads(result.content[0].text)
        return dict_res

    #TODO enhance data getter to async gather pipeline
    async def run_services_retrieval(
            self,
            mcp_client: IduMcpClient,
            user_prompt: dict[str, str],
            model: str,
            scenario_id: int,
    ) -> dict | str:

        instructions = """Исходя из запроса пользователя получи данные по сервисам для формирования ограничений\n" \
                       \nВыбирай из списка сервисов: школа, поликлиника."
                       Используй именно предложенное сочетание как название сервиса. 
                       Если исходя из запроса пользователя получение сервисов не требуется, 
                       то не вызывай инструмент, а просто верни сообщение.\n"""
        services_prompts = await mcp_client.get_services_example_prompts()
        available_services_prompt = await mcp_client.get_available_services_prompt(scenario_id)
        system_prompt = {"role": "system", "content": instructions + available_services_prompt}
        urban_service_tools = await mcp_client.get_urban_api_service_tool()
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
                return await self.execute_urban_api_tool(
                    mcp_client,
                    tool_calls[0],
                    scenario_id
                )
            return response
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
                return await self.execute_urban_api_tool(
                    mcp_client,
                    tool_calls[0],
                    scenario_id
                )
            else:
                return response["message"]["content"]
        except Exception as e:
            logger.exception(e)
            raise

    async def run_buffer_construction(self):
        pass

    async def run_restriction_execution(self):
        pass

    async def run_restriction_execution_pipline(
        self,
        mcp_client: IduMcpClient,
        model: str,
        user_query: str,
        scenario_id: int,
    ) -> AsyncGenerator:
        """
        Run pipline fo forming restrictions
        Returns:
            AsyncGenerator
        """

        user_prompt = {"role": "user", "content": user_query}
        services = await self.run_services_retrieval(mcp_client, user_prompt, model, scenario_id)
        physical_objects = await self.run_physical_objects_retrieval(mcp_client, user_prompt, model, scenario_id)
        layers = {**services, **physical_objects}
        pass
