"""Opt-in live parks check: real ParkPlanner, LLM, Redis and optional ChatStorage.

No model or provider responses are stubbed. Credentials stay in process memory.
Run from the repository root: python -m evaluation.parks.run_live --help.
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from loguru import logger
from redis.asyncio import Redis

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.parks_api_client import ParksApiClient
from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.services.parks.parks_service import ParksService
from src.agents.services.pipeline_state import PipelineStateStore
from src.common.service_auth import (
    build_service_auth,
    internal_user_context_jwt,
    user_id_from_jwt,
)


class RecordedParks(ParksApiClient):
    def __init__(self, url):
        super().__init__(url)
        self.calls = []

    async def _request(self, method, path, **kwargs):
        self.calls.append({"method": method, "path": path, "at": time.time()})
        return await super()._request(method, path, **kwargs)


def save(directory, name, value):
    (directory / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8"
    )


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redis-url", default="redis://localhost:16385/0")
    parser.add_argument("--model")
    parser.add_argument(
        "--base-url", help="Override the configured vision-model endpoint"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume-task",
        type=Path,
        help="Resume an existing task.json without generating again",
    )
    parser.add_argument(
        "--chat-storage",
        action="store_true",
        help="Create a test chat with configured service credentials",
    )
    args = parser.parse_args()
    load_dotenv(".env.agents")
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    logger.remove()  # Never log credential-bearing stack locals in a live report.
    args.out.mkdir(parents=True, exist_ok=True)
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    await redis.ping()
    auth = build_service_auth() if args.chat_storage else None
    handler = (
        JsonApiHandler(os.environ["CHAT_STORAGE"], service_auth=auth) if auth else None
    )
    chat = ChatStorageApiClient(handler) if handler else None
    api = RecordedParks(
        os.getenv("PARKS_API_URL") or "https://api.1.urban.actcognitive.org"
    )
    service = ParksService(
        "http://localhost:11434", chat, None, PipelineStateStore(redis), api
    )
    conversation = str(uuid4())
    chat_id = None
    report = {
        "started_at": time.time(),
        "upstream": api.base_url,
        "assertions": [],
        "mode": "existing-task-recovery" if args.resume_task else "full",
        "chat_storage": bool(chat),
    }
    traces = []
    try:
        token = internal_user_context_jwt("parks-live-check")
        if auth:
            for attempt in range(3):
                try:
                    token = await auth.get_access_token()
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1)
        model = args.model or await service.resolve_model(None)
        report["model"] = model
        if chat:
            created = await chat.create_chat(
                token, "ICII-234 live integration check", test_run=conversation
            )
            chat_id = created.chat_id
            conversation = chat_id
            report["chat_id"] = chat_id

        async def collect(query, **kwargs):
            events = []
            async for event in service.run_parks_pipeline(
                token=token,
                model=model,
                temperature=0,
                user_query=query,
                chat_id=chat_id,
                conversation_key=conversation,
                persist_history=bool(chat),
                **kwargs,
            ):
                events.append(event)
                print(event["type"], event["content"].get("status", ""), flush=True)
            traces.append({"query": query, "events": events})
            save(args.out, "events.json", traces)
            return events

        if args.resume_task:
            task = json.loads(args.resume_task.read_text(encoding="utf8"))
            request_id = str(uuid4())
            state = dict(
                request_id=request_id,
                owner=user_id_from_jwt(token),
                status="running",
                phase="generating",
                model=model,
                temperature=0,
                query="Проанализируй тестовый парк",
                scenario_id=None,
                chat_id=chat_id,
                conversation=conversation,
                persist_history=bool(chat),
                task_id=task["task_id"],
                generation_started=task["created_at"],
                parameters={
                    "park_geojson": json.loads(task["payload"]["parkGeoJson"]),
                    "tags": task["payload"]["tags"],
                    "location_requests": task["payload"]["locationRequests"],
                },
            )
            await service.store.create(state)
            final = await collect("", request_id=request_id)
        else:
            boundary = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"type": "boundary"},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [30.30, 59.97],
                                    [30.302, 59.97],
                                    [30.302, 59.971],
                                    [30.30, 59.971],
                                    [30.30, 59.97],
                                ]
                            ],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {"type": "entry"},
                        "geometry": {"type": "Point", "coordinates": [30.30, 59.9705]},
                    },
                    {
                        "type": "Feature",
                        "properties": {"type": "entry"},
                        "geometry": {"type": "Point", "coordinates": [30.302, 59.9705]},
                    },
                ],
            }
            first = await collect(
                "Создай парк с одной детской площадкой и местами отдыха.",
                inputs={"park_geojson": boundary},
            )
            draft = next(e["content"] for e in first if e["type"] == "clarification")
            assert (
                draft["ready"] and not api.calls
            ), "Generation must not start before confirmation"
            report["assertions"].append("draft without upstream submission")
            edited = await collect(
                "Добавь ещё одну детскую площадку. Места отдыха оставь."
            )
            revised = next(e["content"] for e in edited if e["type"] == "clarification")
            assert revised["revision"] > draft["revision"] and not api.calls
            assert revised["parameters"]["tags"].count("CHILDREN") == 2
            report["assertions"].append(
                "natural-language edit preserved and not submitted"
            )
            final = await collect(
                "Подтверждаю", inputs={"confirmed_revision": revised["revision"]}
            )
            request_id = final[0]["content"]["request_id"]

        state = await service.get_run(request_id, token)
        report.update(
            request_id=request_id,
            task_id=state.get("task_id"),
            status=state["status"],
            phase=state["phase"],
        )
        layers = [event for event in final if event["type"] == "feature_collection"]
        assert (
            layers and layers[0]["content"]["feature_collection"]["features"]
        ), "Missing nonempty GeoJSON"
        report["assertions"].append("real upstream result delivered")
        if chat:
            history = await chat.get_chat(token, chat_id)
            save(args.out, "history.json", history.messages)
            assert state.get(
                "history_result"
            ), "Result must persist even if vision is unavailable"
            parts = [
                part
                for message in history.messages
                for part in message.get("parts", [])
            ]
            assert any(
                part.get("payload", {}).get("type") == "feature_collection"
                for part in parts
            )
            assert any(
                part.get("payload", {}).get("type") == "parks_economy" for part in parts
            )
            report["assertions"].append("result saved to real ChatStorage")
        cursor = final[-1]["content"]["event_id"]
        # A fresh service instance reads the same real Redis checkpoint.
        await service.close()
        close_llm = getattr(service.llm_client.client, "close", None)
        if close_llm:
            await close_llm()
        service = ParksService(
            "http://localhost:11434", chat, None, PipelineStateStore(redis), api
        )
        if state["status"] == "done":
            tail = await collect("", request_id=request_id, after=cursor)
            assert not tail
            report["assertions"].append("fresh-instance cursor replay is empty")
        report["generate_calls"] = sum(
            call["path"] == "/map/generate" for call in api.calls
        )
        assert report["generate_calls"] == (0 if args.resume_task else 1)
        if state["status"] != "done":
            raise RuntimeError(
                "Generation recovered, but the real vision stage did not complete"
            )
        assert any(e["type"] == "chunk" and e["content"].get("text") for e in final)
        report["assertions"].append("real model analysis completed")
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["calls"] = api.calls
        report["finished_at"] = time.time()
        save(args.out, "report.json", report)
        await service.close()
        close_llm = getattr(service.llm_client.client, "close", None)
        if close_llm:
            await close_llm()
        if handler:
            await handler.close()
        if auth:
            await auth.aclose()
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(main())
