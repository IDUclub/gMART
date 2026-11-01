"""Freeze the request every case makes into the standalone benchmark's dataset.

For each case the stage runs gMART's own code against :class:`CaptureLlm` and the
first model request is written to ``data/requests.jsonl`` of the benchmark checkout.
The runner there sends these requests to any model and checks the answers without
gMART. ``input_sha`` ties a request to the case input it was built from, so the
runner can refuse a request whose case was edited after the export.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmarks.harness.capture import Captured, CaptureLlm
from benchmarks.harness.cases import Case
from benchmarks.harness.stage import Stage, StageContext

#: Model id the stages see while exporting. The prompts do not depend on it.
EXPORT_MODEL = "benchmark-model"


def input_sha(stage_id: str, case_input: dict[str, Any]) -> str:
    payload = json.dumps(
        {"stage": stage_id, "input": case_input},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _find_captured(exc: BaseException) -> Captured | None:
    if isinstance(exc, Captured):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            if found := _find_captured(inner):
                return found
    return None


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


async def export_case(
    stage: Stage, case: Case, *, context_window: int | None
) -> dict[str, Any]:
    llm = CaptureLlm(context_window=context_window)
    record: dict[str, Any] = {
        "stage": case.stage,
        "case": case.id,
        "input_sha": input_sha(case.stage, case.input),
    }
    try:
        await stage.fn(
            case.input,
            StageContext(llm=llm, model=EXPORT_MODEL, temperature=0.0),
        )
    except BaseException as exc:  # noqa: BLE001 - Captured is the expected way out
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        if _find_captured(exc) is None and not llm.requests:
            return {
                **record,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
    if not llm.requests:
        # The stage's code decided without asking the model: nothing to measure.
        return {**record, "status": "code"}
    return {**record, "status": "ok", "request": _jsonable(llm.requests[0])}


def _git_revision(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def export(
    stages: dict[str, Stage],
    cases: list[Case],
    data_dir: Path,
    *,
    gmart_root: Path,
    context_window: int | None,
) -> dict[str, Any]:
    async def run_all() -> list[dict[str, Any]]:
        return [
            await export_case(stages[case.stage], case, context_window=context_window)
            for case in cases
        ]

    records = asyncio.run(run_all())
    data_dir.mkdir(parents=True, exist_ok=True)
    frozen = [record for record in records if record["status"] == "ok"]
    with (data_dir / "requests.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for record in frozen:
            fh.write(
                json.dumps(
                    {
                        key: record[key]
                        for key in ("stage", "case", "input_sha", "request")
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    used = sorted({case.stage for case in cases})
    manifest = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "gmart": _git_revision(gmart_root),
        "context_window": context_window,
        "requests": len(frozen),
        "stages": {
            stage_id: {
                "kind": stages[stage_id].kind,
                "description": stages[stage_id].description,
                "scope": stages[stage_id].scope,
            }
            for stage_id in used
        },
        "decided_by_code": [
            f"{r['stage']}/{r['case']}" for r in records if r["status"] == "code"
        ],
        "errors": {
            f"{r['stage']}/{r['case']}": r["error"]
            for r in records
            if r["status"] == "error"
        },
    }
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
