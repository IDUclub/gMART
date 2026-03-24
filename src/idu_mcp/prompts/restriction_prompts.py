from fastmcp import FastMCP
from fastmcp.prompts import Message

mcp = FastMCP(name="RestrictionsPromptService")

@mcp.prompt

@mcp.prompt(name="GetServicesPrompt")
async def get_services_system_prompt():

    return Message(role="system")