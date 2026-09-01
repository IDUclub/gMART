#!/usr/bin/env python3
"""Run the publication-grade restriction experiment matrix.

Every cell has its own output directory and manifest.  The runner performs an
Ollama/model preflight and aborts after consecutive backend failures, so a dead
server cannot silently turn thousands of rows into a model result.

The default full matrix is deliberately finite:

* Gemma 4 12B and GPT-OSS 20B: two paired repeats of base/no-catalog;
* the same repeats with the historical optional schema, base arm only;
* Gemma 3 12B: one historical bridge repeat of those three cells;
* one fixed, cluster-balanced five-prompts-per-scenario synthetic robustness
  slice for all three models.

Run ``--phase smoke`` first, then ``--phase full`` with the same ``--run-root``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inproc_runner as runner  # noqa: E402

# Windows may start this orchestration shell as cp1251 while Loguru and model
# outputs contain symbols outside that code page.  A reporting character must
# never be able to kill the experiment controller.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

EXPERT = "benchmarks/data/gold/exp_data_restrictions.csv"
SYNTHETIC = "benchmarks/data/gold/expanded_catalog.csv"
MODERN_MODELS = ["gemma4:12b", "gpt-oss:20b"]
BRIDGE_MODEL = "gemma3:12b"


@dataclass(frozen=True)
class Cell:
    phase: str
    model: str
    arm: str
    schema_arm: str
    repeat: int
    dataset: str
    sample_per_scenario: int = 0
    limit: int = 0
    save_layers: bool = False

    @property
    def slug(self) -> str:
        dataset = "expert" if self.dataset == EXPERT else "synthetic"
        model = runner._safe_name(self.model)
        return (
            f"{self.phase}/{dataset}/schema-{self.schema_arm}/repeat-{self.repeat:02d}/"
            f"{model}/{self.arm}"
        )


def smoke_cells(models: list[str]) -> list[Cell]:
    cells: list[Cell] = []
    for model in models:
        cells.extend(
            [
                Cell(
                    "smoke",
                    model,
                    runner.ARM_BASE,
                    runner.SCHEMA_REQUIRED,
                    1,
                    EXPERT,
                    limit=3,
                ),
                Cell(
                    "smoke",
                    model,
                    runner.ARM_NO_CATALOG,
                    runner.SCHEMA_REQUIRED,
                    1,
                    EXPERT,
                    limit=3,
                ),
                Cell(
                    "smoke",
                    model,
                    runner.ARM_BASE,
                    runner.SCHEMA_OPTIONAL,
                    1,
                    EXPERT,
                    limit=3,
                ),
            ]
        )
    return cells


def full_cells(repeats: int, robustness_per_scenario: int) -> list[Cell]:
    cells: list[Cell] = []
    for model in MODERN_MODELS:
        for repeat in range(1, repeats + 1):
            for arm in (runner.ARM_BASE, runner.ARM_NO_CATALOG):
                cells.append(
                    Cell(
                        "primary",
                        model,
                        arm,
                        runner.SCHEMA_REQUIRED,
                        repeat,
                        EXPERT,
                        save_layers=(repeat == 1 and arm == runner.ARM_BASE),
                    )
                )
            cells.append(
                Cell(
                    "schema_ablation",
                    model,
                    runner.ARM_BASE,
                    runner.SCHEMA_OPTIONAL,
                    repeat,
                    EXPERT,
                )
            )
    for schema, arm in (
        (runner.SCHEMA_REQUIRED, runner.ARM_BASE),
        (runner.SCHEMA_REQUIRED, runner.ARM_NO_CATALOG),
        (runner.SCHEMA_OPTIONAL, runner.ARM_BASE),
    ):
        cells.append(
            Cell(
                "historical_bridge",
                BRIDGE_MODEL,
                arm,
                schema,
                1,
                EXPERT,
                save_layers=(
                    schema == runner.SCHEMA_REQUIRED and arm == runner.ARM_BASE
                ),
            )
        )
    for model in [*MODERN_MODELS, BRIDGE_MODEL]:
        cells.append(
            Cell(
                "robustness",
                model,
                runner.ARM_BASE,
                runner.SCHEMA_REQUIRED,
                1,
                SYNTHETIC,
                sample_per_scenario=robustness_per_scenario,
            )
        )
    return cells


def expected_rows(cell: Cell) -> int:
    return len(
        runner.load_dataset(
            Path(cell.dataset),
            cell.limit or None,
            cell.sample_per_scenario,
            20260831,
        )
    )


def result_path(root: Path, cell: Cell) -> Path:
    return (
        root
        / cell.slug
        / runner._safe_name(cell.model)
        / f"{cell.arm}--{runner.LOCAL}"
        / "results.jsonl"
    )


def cell_complete(root: Path, cell: Cell) -> bool:
    path = result_path(root, cell)
    if not path.exists():
        return False
    latest: dict[int, dict] = {}
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[int(row["idx"])] = row
    if len(latest) != expected_rows(cell):
        return False
    # Invalid/unexecutable plans are legitimate model outcomes. Infrastructure,
    # data and backend failures make a cell non-comparable and must be retried.
    invalidating = {
        runner.CLS_DATA_UNAVAILABLE,
        runner.CLS_URBAN_API,
        runner.CLS_TOOL_EXECUTION,
        runner.CLS_LLM_BACKEND,
        runner.CLS_TIMEOUT,
        runner.CLS_TOKEN_EXPIRED,
        runner.CLS_TRANSPORT,
        runner.CLS_OTHER,
    }
    return not any(row.get("error_class") in invalidating for row in latest.values())


def command_for(
    root: Path, run_id: str, cell: Cell, args: argparse.Namespace
) -> list[str]:
    out = root / cell.slug
    command = [
        sys.executable,
        "benchmarks/harness/inproc_runner.py",
        "--dataset",
        cell.dataset,
        "--out-dir",
        str(out),
        "--models",
        cell.model,
        "--arms",
        cell.arm,
        "--transports",
        runner.LOCAL,
        "--urban-data",
        "replay",
        "--llm-host",
        args.llm_host,
        "--urban-api-url",
        args.urban_api_url,
        "--schema-arm",
        cell.schema_arm,
        "--repeat-id",
        str(cell.repeat),
        "--run-id",
        run_id,
        "--concurrency",
        "1",
        "--timeout",
        str(args.timeout),
        "--fail-fast-after",
        str(args.fail_fast_after),
        "--progress-every",
        "10",
        "--log-level",
        "WARNING",
    ]
    if cell.limit:
        command.extend(["--limit", str(cell.limit)])
    if cell.sample_per_scenario:
        command.extend(
            [
                "--sample-per-scenario",
                str(cell.sample_per_scenario),
                "--sample-seed",
                "20260831",
            ]
        )
    if cell.save_layers:
        command.append("--save-layers")
    return command


def write_plan(
    root: Path, run_id: str, cells: list[Cell], args: argparse.Namespace
) -> None:
    path = root / "matrix_plan.json"
    previous: dict = {}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
    known = {item["slug"]: item for item in previous.get("cells", [])}
    for cell in cells:
        item = asdict(cell)
        item["slug"] = cell.slug
        item["expected_rows"] = expected_rows(cell)
        known[cell.slug] = item
    payload = {
        "run_id": run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "repeats": args.repeats,
        "robustness_per_scenario": args.robustness_per_scenario,
        "cells": list(known.values()),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cell(root: Path, run_id: str, cell: Cell, args: argparse.Namespace) -> None:
    if cell_complete(root, cell):
        print(f"SKIP complete: {cell.slug}", flush=True)
        return
    command = command_for(root, run_id, cell, args)
    log_path = root / cell.slug / "cell.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "LLM_BACKEND": "ollama",
            "OLLAMA_REQUEST_NUM_CTX": "16384",
            "OLLAMA_THINK_LEVELS": "gpt-oss:20b=low",
            "OLLAMA_CONTEXT_LENGTH": "16384",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": "q8_0",
        }
    )
    if cell.model.startswith("gpt-oss"):
        env["OPENAI_THINK_OFF_EFFORT"] = "low"
    print(f"RUN {cell.slug} ({expected_rows(cell)} rows)", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise SystemExit(f"cell failed with exit code {code}: {cell.slug}")
    if not cell_complete(root, cell):
        raise SystemExit(f"cell ended without all successful rows: {cell.slug}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "full"], required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--robustness-per-scenario", type=int, default=5)
    parser.add_argument("--llm-host", default="http://localhost:11434")
    parser.add_argument(
        "--urban-api-url", default="https://urban-api.testing.idulab.ru/api"
    )
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--fail-fast-after", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    models = [*MODERN_MODELS, BRIDGE_MODEL]
    cells = (
        smoke_cells(models)
        if args.phase == "smoke"
        else full_cells(args.repeats, args.robustness_per_scenario)
    )
    write_plan(root, args.run_id, cells, args)
    for position, cell in enumerate(cells, 1):
        print(f"\nCELL {position}/{len(cells)}", flush=True)
        run_cell(root, args.run_id, cell, args)
    print(f"matrix phase complete: {args.phase} -> {root}")


if __name__ == "__main__":
    main()
