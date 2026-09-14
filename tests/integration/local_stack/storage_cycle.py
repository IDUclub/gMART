"""Exercise all compliance storage parts over HTTP using synthetic events, no LLM."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values
from stability import headers_for, save

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.agents.api_clients.chat_storage_client.request_models import (
    StructuredPartRequest,
)
from src.agents.services.orchestrator.analysis import artifact_parts
from src.agents.services.orchestrator.analysis_context import AnalysisContext


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = dotenv_values(args.env_file)
    events = json.loads(
        (ROOT / "tests/fixtures/compliance_storage_events.json").read_text(
            encoding="utf-8"
        )
    )
    context = AnalysisContext()
    request_id = str(uuid4())
    for event in events:
        context.add_artifact(event, 1, request_id)
    context.finish(
        1,
        "Synthetic storage contract",
        772,
        "completed",
        "Contract fixture",
        request_id,
    )
    parts = [
        *artifact_parts(context),
        StructuredPartRequest(
            kind="data",
            payload={
                "event_type": "analysis_context",
                "content": {**context.dump(), "continue_from": request_id},
            },
        ),
    ]
    wire = [part.model_dump(mode="json", exclude_none=True) for part in parts]
    async with httpx.AsyncClient(timeout=30, trust_env=False) as http:
        headers = await headers_for(http, config)
        response = await http.post(
            "http://localhost:18010/api/v1/chat_history/create_chat",
            headers=headers,
            json={
                "title": "Synthetic compliance storage contract",
                "metadata": {"agent_id": "orchestrator"},
            },
        )
        response.raise_for_status()
        chat_id = response.json()["chat_id"]
        url = f"http://localhost:18010/api/v1/chat_history/{chat_id}/message"
        invalid = await http.post(
            url,
            headers=headers,
            json={
                "role": "assistant",
                "parts": [
                    {"kind": event["type"], "payload": event["content"]}
                    for event in events
                ],
            },
        )
        assert (
            invalid.status_code == 422
        ), "The incompatible wrapped check_plan was not rejected"
        response = await http.post(
            url,
            headers=headers,
            json={"role": "assistant", "parts": wire, "source_event_id": request_id},
        )
        response.raise_for_status()
        history = await http.get(
            f"http://localhost:18010/api/v1/chat_history/{chat_id}", headers=headers
        )
        history.raise_for_status()
        saved = history.json()["messages"][-1]["parts"]
        assert [
            {"kind": p["kind"], "payload": p["payload"]} for p in saved
        ] == wire, "Stored artifact content differs"
        result = {
            "passed": True,
            "chat_id": chat_id,
            "invalid_status": invalid.status_code,
            "valid_status": response.status_code,
            "parts": [p["kind"] for p in saved],
            "artifacts_retained": len(context.artifacts),
            "model_calls": 0,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save(args.output, result)
        print(json.dumps(result))


if __name__ == "__main__":
    asyncio.run(main())
