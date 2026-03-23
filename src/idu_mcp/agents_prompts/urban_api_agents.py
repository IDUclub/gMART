from fastmcp import FastMCP

urban_api_prompts_mcp = FastMCP("Urban API prompts")


@urban_api_prompts_mcp.prompt
async def get_urban_api_prompt(scenario_id: int, user_request: str) -> str:

    return f"""
    Получи данные для запроса пользователя по сценарию {scenario_id} | Запрос пользователя: {user_request}
    """


@urban_api_prompts_mcp.prompt
async def get_restriction_generation_prompt():

    pass
