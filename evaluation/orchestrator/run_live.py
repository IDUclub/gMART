"""Local in-process orchestrator + real MCP/REST/LLM; isolated Redis, no chat writes.

Traces are evidence for review, not an automatic factual-accuracy score.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from contextlib import AsyncExitStack
from pathlib import Path

import fakeredis.aioredis
from dotenv import dotenv_values
from idu_service_auth import KeycloakTokenClient, KeycloakTokenConfig
from loguru import logger
from run_planner import ROOT, save, score, source_hash

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.common.config.app_config import AgentsAppConfig
from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient
from src.agents.services.dvd.dvd_rag_service import DvdRagService
from src.agents.services.normgraph.normgraph_rag_service import NormGraphRagService
from src.agents.services.orchestrator.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.provision.provsion_service import ProvisionService
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.scenario_data.scenario_data_service import ScenarioDataService
from src.common.service_auth import service_mcp_client, user_id_from_jwt


class TracedTools:
    async def execute_tool(self, *args, **kwargs):
        call = dict(args=args, meta=kwargs.get("meta"))
        self.trace.append(call)
        started = time.monotonic()
        try:
            value = await super().execute_tool(*args, **kwargs)
            call["result"] = NormGraphMcpClient._to_dict(value)
            return value
        except Exception as exc:
            call["error"] = type(exc).__name__
            raise
        finally:
            call["seconds"] = round(time.monotonic() - started, 3)


def traced(cls, *args, **kwargs):
    obj = type("Traced" + cls.__name__, (TracedTools, cls), {})(*args, **kwargs)
    obj.trace = []
    return obj


class RecordedPlanner:
    def __init__(self, inner, case):
        self.inner, self.case, self.plan = inner, case, None

    async def build_plan(self, model, query, agents, history=None, **kwargs):
        agents = [a for a in agents if a.key not in self.case["disabled_agents"]]
        self.plan = await self.inner.build_plan(
            model, query, agents, self.case["history"] or history, **kwargs
        )
        return self.plan


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ids", nargs="*")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=240)
    p.add_argument("--document-context-window", type=int, default=8192)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    args = p.parse_args()
    if not 1 <= args.concurrency <= 16:
        p.error("concurrency must be 1..16")
    if not 4096 <= args.document_context_window <= 65536:
        p.error("document context must be 4096..65536; verify the model server limit")
    logger.remove()
    args.out.mkdir(parents=True, exist_ok=True)
    logger.add(args.out / "errors.log", level="ERROR", backtrace=False, diagnose=False)
    provider = json.loads((Path.home() / ".graphify/providers.json").read_text())
    provider = provider.get("local_gpu") or provider["local-gpu"]
    os.environ.update(
        LLM_BACKEND="openai",
        OPENAI_BASE_URL=provider["base_url"],
        OPENAI_THINK_EFFORT="low",
        DVD_CONTEXT_WINDOW_TOKENS=str(args.document_context_window),
    )
    env = dotenv_values(ROOT / "env/.env.agents.dev")
    config = AgentsAppConfig(
        ollama_api_url="http://localhost:11434",
        llm_backend="openai",
        openai_base_url=provider["base_url"],
        idu_mcp_url="http://localhost:8002/mcp",
        effects_mcp_url="http://localhost:8080/effects/mcp",
        dvd_mcp_url="http://localhost:8100/mcp",
        norm_graph_mcp_url="http://localhost:8020/mcp",
        urban_mcp_url=env["URBAN_MCP_SERVER"],
        urban_api_url=env["URBAN_API_URL"],
        chat_storage_url="http://localhost:8010",
    )
    frozen = source_hash()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    manifest_path = args.out / "manifest.json"
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert (
            previous["source_hash"] == frozen
        ), "Cannot resume against different source"
        assert (
            previous.get("document_context_window", 8192)
            == args.document_context_window
        ), "Cannot mix context-window configurations"
        assert (
            previous.get("dataset_sha256")
            == hashlib.sha256(args.cases.read_bytes()).hexdigest()
        ), "Cannot mix datasets"
    save(
        manifest_path,
        dict(
            source_hash=frozen,
            dataset_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            model="gpt-oss-20b",
            document_context_window=args.document_context_window,
            timeout_seconds=args.timeout,
            concurrency=args.concurrency,
            cases=[c["id"] for c in cases],
            boundaries="real local MCPs + real Urban MCP/REST + real LLM; fakeredis; ChatStorage disabled; provided history injected into actual planner",
            accuracy="No completion event alone proves factual accuracy. Review traces against case answer_rubric.",
        ),
    )
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    def auth(client):
        return KeycloakTokenClient(
            KeycloakTokenConfig(
                auth_server_url="http://localhost:8085",
                realm="local",
                client_id=client,
                client_secret="local-integration-only",
                background_refresh=True,
            )
        )

    async with AsyncExitStack() as stack:
        service_auth = await stack.enter_async_context(auth("gmart"))
        internal_auth = await stack.enter_async_context(auth("gmart-internal"))
        await service_auth.get_access_token()
        await internal_auth.get_access_token()

        async def one(case):
            nonlocal done
            path = args.out / "runs" / f"{case['id']}.json"
            if args.resume and path.exists():
                return
            async with sem:
                started = time.monotonic()
                events = []
                clients = {}
                result = dict(
                    id=case["id"],
                    group=case["group"],
                    query=case["query"],
                    scenario_id=case["scenario_id"],
                )
                redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
                recorder = None
                try:
                    token = await service_auth.get_access_token()
                    uid = user_id_from_jwt(token)
                    for name, cls, url, a in [
                        ("idu", IduMcpClient, config.IDU_MCP_URL, internal_auth),
                        (
                            "effects",
                            EffectsMcpClient,
                            config.EFFECTS_MCP_URL,
                            service_auth,
                        ),
                        ("dvd", DvdMcpClient, config.DVD_MCP_URL, service_auth),
                        (
                            "norms",
                            NormGraphMcpClient,
                            config.NORM_GRAPH_MCP_URL,
                            service_auth,
                        ),
                    ]:
                        clients[name] = traced(
                            cls,
                            await service_mcp_client(url, a, uid),
                            **({"mcp_url": url} if name in {"dvd", "norms"} else {}),
                            **({"user_id": uid} if name == "dvd" else {}),
                        )
                    clients["urban"] = traced(
                        UrbanMcpClient, config.URBAN_MCP_URL, token
                    )
                    state = PipelineStateStore(redis)
                    urban = UrbanApiClient(
                        JsonApiHandler(config.URBAN_API_URL, service_auth=service_auth)
                    )
                    chat = ChatStorageApiClient(
                        JsonApiHandler(
                            config.CHAT_STORAGE_URL, service_auth=service_auth
                        )
                    )
                    common = (config.OLLAMA_URL, chat, urban, state)
                    service = OrchestratorService(
                        *common,
                        RestrictionParserService(*common),
                        ProvisionService(*common),
                        DvdRagService(*common),
                        NormGraphRagService(*common),
                        config,
                        scenario_data_service=ScenarioDataService(
                            *common,
                            idu_mcp_url=config.IDU_MCP_URL,
                            service_auth=internal_auth,
                        ),
                    )
                    recorder = RecordedPlanner(service.plan_builder, case)
                    service.plan_builder = recorder
                    async with asyncio.timeout(args.timeout):
                        async for event in service.run_orchestration_pipeline(
                            idu_mcp_client=clients["idu"],
                            effects_mcp_client=clients["effects"],
                            dvd_mcp_client=clients["dvd"],
                            normgraph_mcp_client=clients["norms"],
                            urban_mcp_client=clients["urban"],
                            token=token,
                            model="gpt-oss-20b",
                            temperature=0,
                            user_query=case["query"],
                            scenario_id=case["scenario_id"],
                            persist_history=False,
                        ):
                            events.append(event)
                except Exception as exc:
                    result["error"] = type(exc).__name__
                finally:
                    await redis.aclose()
                if recorder and recorder.plan:
                    result["plan"] = recorder.plan.model_dump(mode="json")
                    result["routing_checks"] = score(case, result["plan"])
                result["events"] = events
                result["tools"] = {k: v.trace for k, v in clients.items()}
                result["seconds"] = round(time.monotonic() - started, 3)
                result["source_unchanged"] = source_hash() == frozen
                finals = [
                    e["content"] for e in events if e["type"] == "orchestrator_final"
                ]
                result["final"] = finals[-1] if finals else None
                result["factual_verdict"] = "not_reviewed"
                save(path, result)
                done += 1
                statuses = (
                    [s["status"] for s in result["final"]["steps"]]
                    if result["final"]
                    else []
                )
                print(
                    json.dumps(
                        dict(
                            id=case["id"],
                            done=done,
                            seconds=result["seconds"],
                            error=result.get("error"),
                            statuses=statuses,
                        )
                    ),
                    flush=True,
                )

        await asyncio.gather(*(one(c) for c in cases))


if __name__ == "__main__":
    asyncio.run(main())
