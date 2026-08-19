from src.agents.common.config.app_config import AgentsAppConfig


def make_config(**kwargs):
    return AgentsAppConfig(
        ollama_api_url="http://llm",
        idu_mcp_url="http://idu/mcp",
        effects_mcp_url="http://effects/mcp",
        chat_storage_url="http://chat",
        urban_api_url="http://urban",
        **kwargs,
    )


def test_dvd_api_url_is_derived_from_mcp_url():
    config = make_config(dvd_mcp_url="http://dvd:8100/mcp/")

    assert config.DVD_API_URL == "http://dvd:8100"
    assert config.to_dict()["DVD_API_URL"] == "http://dvd:8100"


def test_explicit_dvd_api_url_wins():
    config = make_config(
        dvd_mcp_url="http://dvd-mcp/mcp", dvd_api_url="http://dvd-rest/api/"
    )

    assert config.DVD_API_URL == "http://dvd-rest/api"
