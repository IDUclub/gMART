"""Contention probe on synthetic Redis keys with a five-minute cleanup TTL."""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import redis.asyncio as redis
from loguru import logger

from src.agents.services.pipeline_state import PipelineStateStore


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redis-url", default="redis://localhost:16389/0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.disable("src.agents")
    db = redis.from_url(args.redis_url, decode_responses=True)
    store = PipelineStateStore(db)
    results = []
    try:
        for repeat in range(3):
            for concurrency in (1, 4, 8):
                scope = f"stability-synthetic-{uuid4()}"
                gate = asyncio.Semaphore(concurrency)

                async def write(i):
                    async with gate:
                        await store.save_analysis_context(
                            scope,
                            {
                                "artifacts": [
                                    {
                                        "id": str(i),
                                        "content": {"synthetic": True, "value": i},
                                    }
                                ],
                                "completed": [{"request_id": str(i), "step": 1}],
                            },
                        )

                started = time.monotonic()
                outcomes = await asyncio.gather(
                    *(write(i) for i in range(80)), return_exceptions=True
                )
                saved = await store.get_analysis_context(scope)
                results.append(
                    {
                        "repeat": repeat + 1,
                        "concurrency": concurrency,
                        "writes": 80,
                        "errors": [
                            type(x).__name__
                            for x in outcomes
                            if isinstance(x, Exception)
                        ],
                        "saved_artifacts": len(saved.get("artifacts", [])),
                        "saved_operations": len(saved.get("completed", [])),
                        "seconds": round(time.monotonic() - started, 3),
                    }
                )
                await db.expire("analysis:" + scope, 300)
    finally:
        await db.aclose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results))
    return int(any(r["errors"] for r in results))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
