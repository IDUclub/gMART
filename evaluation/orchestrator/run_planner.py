"""Run the production planner against a live configured LLM; never fake its output."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from loguru import logger

from src.agents.model_clients.factory import build_llm_adapter
from src.agents.services.orchestrator.orchestrator_catalog import available_agents
from src.agents.services.orchestrator.orchestrator_plan_builder import (
    OrchestratorPlanBuilder,
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def score(case, plan):
    agents = [s["agent"] for s in plan.get("steps", [])]
    text = " ".join(s["task"] for s in plan.get("steps", [])).lower()
    return {
        "mode": plan.get("mode") == case["expected_mode"],
        "agents": set(agents) == set(case["expected_agents"]),
        "no_duplicates": len(agents) == len(set(agents)),
        "task_details": all(t.lower() in text for t in case["required_task_terms"]),
        "question": case["expected_mode"] != "needs_clarification"
        or bool((plan.get("clarification_question") or "").strip()),
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--split", choices=["development", "holdout"])
    p.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    p.add_argument("--base-url")
    p.add_argument("--model", default="gpt-oss-20b")
    args = p.parse_args()
    if not 1 <= args.concurrency <= 32:
        p.error("concurrency must be 1..32")
    logger.remove()
    profile = (
        json.loads((Path.home() / ".graphify/providers.json").read_text())
        if not args.base_url
        else {}
    )
    provider = profile.get("local_gpu") or profile.get("local-gpu", {})
    base_url = args.base_url or provider["base_url"]
    os.environ["OPENAI_THINK_EFFORT"] = "low"
    llm = build_llm_adapter(
        "http://localhost:11434", backend="openai", base_url=base_url, timeout=90
    )
    models = await llm.list()
    assert args.model in [
        m["model"] for m in models["models"]
    ], "Requested model unavailable"
    cases_path = args.cases
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if args.split:
        cases = [c for c in cases if c["split"] == args.split]
    if args.limit:
        cases = cases[: args.limit]
    frozen = source_hash()
    save(
        args.out / "manifest.json",
        dict(
            model=args.model,
            base_url=base_url,
            source_hash=frozen,
            dataset_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            cases=[c["id"] for c in cases],
            evaluation="live_llm_routing_only; does not measure answer factual accuracy",
        ),
    )
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def one(case):
        async with sem:
            started = time.monotonic()
            config = SimpleNamespace(
                DVD_MCP_URL="configured",
                NORM_GRAPH_MCP_URL="configured",
                URBAN_MCP_URL="configured",
            )
            agents = [
                a
                for a in available_agents(config, case["scenario_id"])
                if a.key not in case["disabled_agents"]
            ]
            result = dict(
                id=case["id"],
                group=case["group"],
                split=case["split"],
                query=case["query"],
            )
            try:
                plan = await OrchestratorPlanBuilder(llm).build_plan(
                    args.model,
                    case["query"],
                    agents,
                    history=case["history"],
                    scenario_id=case["scenario_id"],
                )
                result["plan"] = plan.model_dump(mode="json")
                result["checks"] = score(case, result["plan"])
                result["passed"] = all(result["checks"].values())
            except Exception as exc:
                result.update(error=type(exc).__name__, passed=False)
            result["seconds"] = round(time.monotonic() - started, 3)
            save(args.out / "runs" / f"{case['id']}.json", result)
            results.append(result)
            if len(results) % 20 == 0 or len(results) == len(cases):
                print(
                    json.dumps(
                        dict(
                            done=len(results),
                            total=len(cases),
                            passed=sum(r["passed"] for r in results),
                        )
                    ),
                    flush=True,
                )

    await asyncio.gather(*(one(c) for c in cases))
    summary = dict(
        total=len(results),
        passed=sum(r["passed"] for r in results),
        errors=dict(Counter(r["error"] for r in results if "error" in r)),
        source_unchanged=source_hash() == frozen,
    )
    summary["groups"] = {
        g: dict(
            total=sum(r["group"] == g for r in results),
            passed=sum(r["group"] == g and r["passed"] for r in results),
        )
        for g in sorted({r["group"] for r in results})
    }
    save(args.out / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
