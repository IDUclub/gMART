"""All configured sinks must suppress traceback local variable values."""

import sys
from unittest.mock import Mock

import pytest

from src.agents.common.logging import log_config as agents_logging
from src.idu_mcp.common.logging import log_config as mcp_logging


@pytest.mark.parametrize(
    "config", [agents_logging, mcp_logging], ids=["agents", "idu_mcp"]
)
def test_sinks_disable_diagnose(config, tmp_path, monkeypatch):
    add = Mock()
    rotate = Mock()
    monkeypatch.setattr(config.logger, "add", add)
    monkeypatch.setattr(config.logger, "remove", Mock())
    monkeypatch.setattr(config, "_get_project_root", lambda: tmp_path)
    monkeypatch.setattr(config, "start_daily_log_rotation", rotate)
    if config is agents_logging:
        monkeypatch.setattr(config, "_intercept_stdlib_logging", Mock())

    log_path = config.config_logger("synthetic.log")

    assert log_path == tmp_path / "synthetic.log"
    assert [call.args[0] for call in add.call_args_list] == [sys.stderr, log_path]
    assert all(call.kwargs.get("diagnose") is False for call in add.call_args_list)
    assert "backtrace" not in add.call_args_list[0].kwargs
    assert add.call_args_list[1].kwargs["backtrace"] is True
    rotate.assert_called_once_with(log_path)
    assert not list(tmp_path.iterdir())
