"""Public HTTP/MCP transport used by industrial acceptance and its preflight."""

import base64
import json

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


def save(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


async def headers_for(http, config):
    response = await http.post(
        f"{config['SERVICE_AUTH_SERVER_URL']}/realms/{config['SERVICE_AUTH_REALM']}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": config["SERVICE_AUTH_CLIENT_ID"],
            "client_secret": config["SERVICE_AUTH_CLIENT_SECRET"],
        },
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
    return {"Authorization": "Bearer " + token, "X-User-Id": claims["sub"]}


async def provision(headers, sid):
    transport = StreamableHttpTransport(
        "http://localhost:18080/effects/mcp", headers=headers
    )
    async with Client(transport) as client:
        result = await client.call_tool(
            "CalculateServicesProvision",
            {
                "scenario_id": sid,
                "services": {
                    "22": {"name": "Школа", "as_layer": True},
                    "21": {"name": "Детский сад", "as_layer": True},
                },
            },
        )
    data = result.data
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    if not isinstance(data, dict):
        data = json.loads(result.content[0].text)
    return data


def parse_events(response):
    response.raise_for_status()
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def stored_context(
    http,
    headers,
    events,
    final,
    directory,
    origin="http://localhost:18000",
    chat_id=None,
):
    from urllib.parse import quote

    chat_id = chat_id or next(
        e["content"]["event"]["chat_id"]
        for e in events
        if e["type"] == "service_event"
        and e["content"].get("event", {}).get("storage_event_type") == "chat_created"
    )
    response = await http.get(
        f"http://localhost:18010/api/v1/chat_history/{chat_id}", headers=headers
    )
    response.raise_for_status()
    history = response.json()
    save(directory / "history.json", history)
    saved = next(
        p["payload"]["content"]
        for m in reversed(history["messages"])
        for p in m["parts"]
        if p["kind"] == "data" and p["payload"].get("event_type") == "analysis_context"
    )
    if {a["id"] for a in saved["artifacts"]} != {a["id"] for a in final["artifacts"]}:
        raise ValueError("Final/history artifact IDs differ")
    for event in events:
        if event["type"] != "step_event":
            continue
        item = event["content"].get("event", {})
        if item.get("type") not in {"table", "feature_collection"}:
            continue
        content = dict(item["content"])
        aid = content.pop("artifact_id")
        if not any(
            a["id"] == aid and a["content"] == content for a in saved["artifacts"]
        ):
            raise ValueError("Emitted artifact payload differs from history")
    for artifact in saved["artifacts"]:
        if not artifact.get("confirmed"):
            continue
        response = await http.get(
            f"{origin}/orchestrator/runs/{quote(final['continue_from'], safe='')}/artifacts/{quote(artifact['id'], safe='')}",
            headers=headers,
        )
        response.raise_for_status()
        if response.json() != artifact:
            raise ValueError("Downloaded artifact differs from history")
    return saved
