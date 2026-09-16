from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.agents.dependencies.dependencies import get_system_service
from src.agents.main import app
from src.agents.schema.app_config_response import AppConfigResponse
from src.agents.services.synapse.system_service import SystemService


@pytest.mark.parametrize("body", [None, {}, {"password": "incorrect"}])
def test_config_and_logs_require_no_credentials(tmp_path, body):
    log_file = tmp_path / "agents.log"
    log_file.write_text("diagnostic log\n", encoding="utf-8")
    config = {name: "http://localhost" for name in AppConfigResponse.model_fields}
    service = SystemService(
        log_file,
        SimpleNamespace(SYSTEM_PASSWORD="configured-password", to_dict=lambda: config),
    )
    app.dependency_overrides[get_system_service] = lambda: service
    try:
        client = TestClient(app)
        response = client.post("/system/config", json=body)
        assert response.status_code == 200
        assert response.json() == config
        response = client.get("/system/logs")
        assert response.status_code == 200
        assert response.text == "diagnostic log\n"
    finally:
        app.dependency_overrides.pop(get_system_service, None)
