"""Run cases x profiles x repeats and write one record per run."""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from benchmarks.harness.cases import Case
from benchmarks.harness.checks import evaluate_check, score
from benchmarks.harness.config import BenchConfig, ProfileConfig
from benchmarks.harness.llm import RecordingLlm, build_adapter
from benchmarks.harness.stage import Stage, StageContext
from src.agents.model_clients import llm_usage


def to_jsonable(value: Any) -> Any:
    """Plain JSON data out of pydantic models, dataclasses, enums and tuples."""
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    return value


class Runner:
    def __init__(
        self,
        config: BenchConfig,
        stages: dict[str, Stage],
        *,
        save_prompts: bool = False,
        case_timeout_s: float = 900.0,
    ) -> None:
        self.config = config
        self.stages = stages
        self.save_prompts = save_prompts
        self.case_timeout_s = case_timeout_s
        self._adapters: dict[str, Any] = {}

    def _adapter(self, endpoint_name: str):
        if endpoint_name not in self._adapters:
            self._adapters[endpoint_name] = build_adapter(
                self.config.endpoints[endpoint_name]
            )
        return self._adapters[endpoint_name]

    async def run_one(
        self, case: Case, profile: ProfileConfig, repeat: int
    ) -> dict[str, Any]:
        stage = self.stages.get(case.stage)
        model_name = profile.model_for(case.stage)
        record: dict[str, Any] = {
            "profile": profile.name,
            "stage": case.stage,
            "case": case.id,
            "tags": list(case.tags),
            "repeat": repeat,
            "model": model_name,
            "kind": stage.kind if stage else None,
            "input": case.input,
            "review": case.review,
        }
        if stage is None:
            return {**record, "status": "error", "error": "unknown stage", "score": 0.0}
        if model_name is None:
            return {**record, "status": "skipped", "error": "no model for stage"}
        model_cfg = self.config.models[model_name]
        llm = RecordingLlm(self._adapter(model_cfg.endpoint), model_cfg)
        record["model_id"] = model_cfg.model
        record["endpoint"] = model_cfg.endpoint
        usage = llm_usage.meter()
        started = time.perf_counter()
        output: Any = None
        try:
            output = await asyncio.wait_for(
                stage.fn(
                    case.input,
                    StageContext(
                        llm=llm,
                        model=model_cfg.model,
                        temperature=self.config.temperature,
                    ),
                ),
                timeout=self.case_timeout_s,
            )
            output = to_jsonable(output)
            error = None
        except (
            Exception
        ) as exc:  # noqa: BLE001 - a failed case is a result, not a crash
            error = f"{type(exc).__name__}: {exc}"
            logger.debug(traceback.format_exc())
        elapsed = time.perf_counter() - started

        checks = (
            [evaluate_check(output, check) for check in case.expect]
            if error is None
            else []
        )
        if error is not None:
            status, case_score = "error", 0.0
        else:
            case_score = score(checks)
            passed = all(check.passed for check in checks)
            status = "pass" if passed else "fail"
        return {
            **record,
            "status": status,
            "needs_review": stage.kind == "manual",
            "score": round(case_score, 4),
            "checks": [check.to_dict() for check in checks],
            "output": output,
            "error": error,
            "latency_s": round(elapsed, 3),
            "llm_calls": len(llm.calls),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "unmeasured_calls": usage.unmeasured_calls,
            "calls": [
                call.to_dict(with_prompt=self.save_prompts) for call in llm.calls
            ],
        }

    async def run(
        self,
        cases: list[Case],
        profiles: list[ProfileConfig],
        out_file: Path,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        results: list[dict[str, Any]] = []
        total = len(cases) * len(profiles) * self.config.repeat
        done = 0
        lock = asyncio.Lock()

        async def job(case: Case, profile: ProfileConfig, repeat: int) -> None:
            nonlocal done
            async with semaphore:
                result = await self.run_one(case, profile, repeat)
            async with lock:
                results.append(result)
                done += 1
                with out_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                logger.info(
                    f"[{done}/{total}] {result['profile']} {case.key} "
                    f"#{repeat}: {result['status']} score={result.get('score')} "
                    f"{result.get('latency_s', 0)}s"
                    + (f" — {result['error']}" if result.get("error") else "")
                )

        # Profiles run one after another: a shared server under two models at once
        # would make each one's latency meaningless.
        for profile in profiles:
            await asyncio.gather(
                *(
                    job(case, profile, repeat)
                    for repeat in range(1, self.config.repeat + 1)
                    for case in cases
                )
            )
        return results
