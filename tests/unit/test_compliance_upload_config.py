"""Upload budgets follow deployment settings before parsing any request."""

import json
import os
import subprocess
import sys

import pytest

from src.agents.common.config.app_config import ComplianceUploadLimits

SETTINGS = {
    "COMPLIANCE_MAX_UPLOAD_MB": 50,
    "COMPLIANCE_MAX_ARCHIVE_MEMBERS": 2000,
    "COMPLIANCE_MAX_UNPACKED_MB": 200,
    "COMPLIANCE_MAX_LAYERS": 500,
}


def test_default_upload_limits(monkeypatch):
    for variable in SETTINGS:
        monkeypatch.delenv(variable, raising=False)
    # Isolate defaults from a developer's optional dotenv file.
    monkeypatch.setattr(
        "src.agents.common.config.app_config_loader.try_load", lambda _: False
    )
    assert ComplianceUploadLimits.from_env() == ComplianceUploadLimits()


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_upload_limits_fail_at_startup(monkeypatch, value):
    monkeypatch.setenv("COMPLIANCE_MAX_LAYERS", value)
    with pytest.raises(ValueError):
        ComplianceUploadLimits.from_env()


def test_env_limits_reach_parser_and_router():
    # A fresh process verifies import-time settings without replacing dataclasses
    # and constants in modules already shared by the rest of the unit suite.
    env = {**os.environ, **{key: str(value + 1) for key, value in SETTINGS.items()}}
    script = """
import json
from src.agents.common.config.app_config import ComplianceUploadLimits
from src.agents.services.compilance import compliance_uploads as uploads
from src.agents.routers import compliance_controller as router
limits = ComplianceUploadLimits.from_env()
print(json.dumps([
    uploads.MAX_UPLOAD_BYTES, uploads.MAX_ARCHIVE_MEMBERS,
    uploads.MAX_UNPACKED_BYTES, uploads.MAX_LAYERS, router.MAX_UPLOAD_BYTES,
    limits.max_upload_mb,
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.splitlines()[-1]) == [
        51 * 1024 * 1024,
        2001,
        201 * 1024 * 1024,
        501,
        51 * 1024 * 1024,
        51,
    ]
