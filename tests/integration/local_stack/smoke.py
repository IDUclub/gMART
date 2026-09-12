"""Live HTTP/MCP acceptance checks. Run only against the isolated compose stack."""

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from dotenv import dotenv_values
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "health",
            "effects",
            "analysis",
            "data",
            "documents",
            "artifacts",
            "llm",
            "seed-documents",
        ],
        default="health",
    )
    parser.add_argument("--scenario", type=int, default=772)
    args = parser.parse_args()
    config = dotenv_values(args.env_file)
    args.output.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=660, trust_env=False) as http:
        token_url = f"{config['SERVICE_AUTH_SERVER_URL']}/realms/{config['SERVICE_AUTH_REALM']}/protocol/openid-connect/token"
        auth = await http.post(
            token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": config["SERVICE_AUTH_CLIENT_ID"],
                "client_secret": config["SERVICE_AUTH_CLIENT_SECRET"],
            },
        )
        auth.raise_for_status()
        token = auth.json()["access_token"]
        claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
        headers = {"Authorization": f"Bearer {token}", "X-User-Id": claims["sub"]}
        if args.mode == "seed-documents":
            from document_cycle import document_cycle

            await document_cycle(http, headers, args.output)
            return
        if args.mode == "llm":
            response = await http.get(
                "http://localhost:18000/llm/message/stream",
                params={
                    "request": "Reply with exactly LOCAL_SDK_OK",
                    "model": config["LLM_MODEL"],
                },
                headers=headers,
            )
            response.raise_for_status()
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            assert events and not any(e["type"] == "error" for e in events), events
            assert "LOCAL_SDK_OK" in "".join(
                e["content"] for e in events if isinstance(e.get("content"), str)
            ), events
            print(
                "PASS local Agents HTTP/SSE -> Agents SDK -> remote local-gpu vLLM",
                flush=True,
            )
            return
        if args.mode == "artifacts":
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from src.agents.services.orchestrator.analysis import artifact_parts

            layer = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [30, 60]},
                        "properties": {"name": "Synthetic school"},
                    }
                ],
            }
            artifacts = [
                {
                    "id": "test-table",
                    "kind": "table",
                    "confirmed": True,
                    "content": {
                        "name": "synthetic",
                        "title": "Synthetic table",
                        "columns": [{"key": "count", "label": "Count"}],
                        "rows": [{"count": 1}],
                        "complete": True,
                        "total_rows": 1,
                    },
                },
                {
                    "id": "test-layer",
                    "kind": "feature_collection",
                    "confirmed": True,
                    "content": {"name": "Synthetic layer", "feature_collection": layer},
                },
            ]
            parts = [
                part.model_dump(mode="json", exclude_none=True)
                for part in artifact_parts(SimpleNamespace(artifacts=artifacts))
            ]
            origin = "http://localhost:18010/api/v1/chat_history"
            response = await http.post(
                origin + "/create_chat",
                headers=headers,
                json={"title": "Local SDK synthetic artifact test"},
            )
            response.raise_for_status()
            chat_id = response.json()["chat_id"]
            response = await http.post(
                f"{origin}/{chat_id}/message",
                headers=headers,
                json={"role": "assistant", "parts": parts},
            )
            response.raise_for_status()
            response = await http.get(f"{origin}/{chat_id}", headers=headers)
            response.raise_for_status()
            stored = response.json()
            (args.output / "artifacts-chat.json").write_text(
                json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            loaded = [
                part for message in stored["messages"] for part in message["parts"]
            ]
            assert [part["payload"] for part in loaded] == [
                part["payload"] for part in parts
            ]
            print(
                "PASS gMART artifact serializer -> ChatStorage HTTP -> MongoDB -> chat history: table and nonempty GeoJSON",
                flush=True,
            )
            return
        if args.mode == "health":
            failures = []
            for name, url in [
                ("agents", "http://localhost:18000/llm/available_models"),
                ("idu_mcp", "http://localhost:18002/health"),
                ("effects", "http://localhost:18080/status"),
                ("dvd", "http://localhost:18100/ping"),
                ("normgraph", "http://localhost:18020/ping"),
            ]:
                try:
                    response = await http.get(url, headers=headers, timeout=10)
                    print(name, response.status_code, flush=True)
                    response.raise_for_status()
                except httpx.HTTPError as error:
                    print(name, type(error).__name__, flush=True)
                    failures.append(name)
            assert not failures, f"Unavailable services: {failures}"
            return
        if args.mode == "effects":
            transport = StreamableHttpTransport(
                "http://localhost:18080/effects/mcp", headers=headers
            )
            async with Client(transport) as client:
                result = await client.call_tool(
                    "CalculateServicesProvision",
                    {
                        "scenario_id": args.scenario,
                        "services": {"22": {"name": "Школа", "as_layer": True}},
                    },
                )
            data = result.structured_content
            if not data:
                data = json.loads(result.content[0].text)
            (args.output / "effects.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            text = json.dumps(data, ensure_ascii=False)
            assert "KeyError" not in text, text
            assert "missing_service_normative" in text, text
            assert "required_action" in text, text
            print(
                "PASS effects: missing normative has actionable domain error",
                flush=True,
            )
            return
        if args.mode == "data":
            query = f"В сценарии {args.scenario} посчитай школы и детские сады, сравни их количество в таблице и верни слои этих объектов. Не оценивай обеспеченность."
        elif args.mode == "analysis":
            query = f"В сценарии {args.scenario} посчитай школы и детские сады, сравни их количество в таблице и верни слои объектов. Затем оцени обеспеченность школами. Если нормативы не заданы, сохрани результаты подсчёта и прямо объясни, каких данных не хватает."
        else:
            query = (
                "Сравни исходный текст пункта 1.1 документа LOCAL SDK TEST из DVD "
                "с ограничением, извлечённым в NormGraph из этого документа. "
                "Проверь совпадение числового требования и объектов, объясни результат "
                "и приложи ссылки на исходный пункт и запись ограничения. Используй оба "
                "источника. Это синтетические тестовые данные, не действующий норматив."
            )
        params = {
            "request": query,
            "scenario_id": args.scenario,
            "model": config["LLM_MODEL"],
            "temperature": 0,
        }
        if args.mode == "documents":
            params.pop("scenario_id")
        response = await http.get(
            "http://localhost:18000/orchestrator/route/stream",
            params=params,
            headers=headers,
        )
        response.raise_for_status()
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        (args.output / f"{args.mode}-events.json").write_text(
            json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        assert events, "Empty SSE response"
        assert not any(event["type"] == "error" for event in events), events[-1]
        final = next(
            event["content"]
            for event in reversed(events)
            if event["type"] == "orchestrator_final"
        )
        print(
            json.dumps(
                {
                    k: final[k]
                    for k in ("status", "budget", "continue_from")
                    if k in final
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.mode == "documents":
            from document_cycle import verify_analysis

            await verify_analysis(http, headers, args.output, events, final)
        request_id = final["continue_from"]
        replay = await http.get(
            "http://localhost:18000/orchestrator/route/stream",
            params={**params, "request_id": request_id},
            headers=headers,
        )
        replay.raise_for_status()
        replay_events = [
            json.loads(line[6:])
            for line in replay.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events == replay_events, "Terminal replay differs"
        print("PASS terminal SSE replay", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
