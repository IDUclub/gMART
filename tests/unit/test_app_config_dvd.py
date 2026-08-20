import pytest

from src.agents.common.config.app_config import AgentsAppConfig


def make_config(**kwargs):
    values = {
        "ollama_api_url": "http://localhost:11434",
        "idu_mcp_url": "http://idu/mcp",
        "effects_mcp_url": "http://effects/mcp",
        "chat_storage_url": "http://chat",
        "urban_api_url": "http://urban",
    }
    values.update(kwargs)
    return AgentsAppConfig(**values)


def test_dvd_api_url_is_derived_from_mcp_url():
    config = make_config(dvd_mcp_url="http://dvd:8100/mcp/")

    assert config.DVD_API_URL == "http://dvd:8100"
    assert config.to_dict()["DVD_API_URL"] == "http://dvd:8100"


def test_explicit_dvd_api_url_wins():
    config = make_config(
        dvd_mcp_url="http://dvd-mcp/mcp", dvd_api_url="http://dvd-rest/api/"
    )

    assert config.DVD_API_URL == "http://dvd-rest/api"


def test_remote_ollama_is_rejected_even_when_openai_backend_is_selected():
    with pytest.raises(ValueError, match="must point to local Ollama"):
        make_config(ollama_api_url="http://a.dgx:11434", llm_backend="openai")


def test_language_model_endpoint_on_a_dgx_is_rejected():
    with pytest.raises(ValueError, match="must not target 'a.dgx'"):
        make_config(openai_base_url="http://a.dgx:8001/v1")


def test_remote_openai_compatible_endpoint_on_another_host_is_allowed():
    config = make_config(openai_base_url="http://a6k4.dgx:8001/v1")

    assert config.OPENAI_BASE_URL == "http://a6k4.dgx:8001/v1"
