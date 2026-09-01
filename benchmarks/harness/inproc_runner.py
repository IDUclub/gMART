#!/usr/bin/env python3
"""Run the restrictions pipeline in this process — no HTTP interface at all.

The previous harness drove the experiment through the deployed application:
``GET /restrictions/generate_restrictions/stream``, SSE, Redis-backed pipeline
state, ChatStorage, a token proxy. None of that is the object of study, and all
of it produced failures that were then scored against the model. Measured on the
last run, technical error rates ran 31–66 % per model, and a good part of them
were message-size limits, stream timeouts and transport hiccups.

This runner keeps the part the paper is about — entity extraction, plan building,
buffer and restriction construction — and drops the rest:

===========================  ==========================================
dropped                      how
===========================  ==========================================
FastAPI + SSE                the pipeline is an async generator; it is
                             iterated directly
Redis pipeline state         ``NullPipelineStateStore``
ChatStorage                  ``persist_history=False`` + DISABLE_CHAT_HISTORY
``/auth/token`` proxy        the token is passed in
MCP over HTTP                ``LocalIduMcpClient`` (``--transport local``)
===========================  ==========================================

What still speaks HTTP is what genuinely is a separate service: Urban API and the
model server. The Urban API traffic can itself be served from disk — see
``--urban-data`` and ``prefetch_scenarios.py`` — which makes a run reproducible
and survives a dropped VPN.

Two things the SSE harness could not do, and this one does:

* **The plan is recorded.** The pipeline never emits the ``RestrictionPlan`` as
  an event (see ``RestrictionsResponse``: there is no such event type), so
  ``run_benchmark.py``'s ``_find_plan`` scrape of the event stream had nothing to
  find. Here the plan object is captured where it is built, so intent, source and
  target entities and the buffer parameter can actually be scored.
* **Failures are classified where they happen** — a missing offline layer, an
  Urban API status, a geometry tool error and a model failure are different
  exceptions at different stages, not one string parsed after the fact.

Usage::

    python benchmarks/harness/inproc_runner.py \\
        --dataset benchmarks/data/gold/exp_data.csv \\
        --models gemma3:12b gpt-oss:20b \\
        --out-dir benchmarks/data/results_inproc \\
        --urban-data replay --arms base no_catalog
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

# The pipeline must never touch ChatStorage on a benchmark run. persist_history
# is passed False as well; this makes the read path impossible too, and has to be
# set before the services are imported since the flag is read at call time.
os.environ.setdefault("DISABLE_CHAT_HISTORY", "1")
os.environ.setdefault("DISABLE_PIPELINE_STATE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import Field  # noqa: E402

from src.agents.api_clients.chat_storage_client.chat_storage_client import (  # noqa: E402
    ChatStorageApiClient,
)
from src.agents.api_clients.urban_api_client.urban_api_client import (  # noqa: E402
    UrbanApiClient,
)
from src.agents.common.api_handlers.json_api_handler import (  # noqa: E402
    JsonApiHandler,
)
from src.agents.common.exceptions.token_exceptions import (  # noqa: E402
    TokenExpiredError,
)
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient  # noqa: E402
from src.agents.mcp_clients.local_idu_mcp_client import (  # noqa: E402
    LocalIduMcpClient,
)
from src.agents.mcp_clients.mcp_http import build_mcp_client  # noqa: E402
from src.agents.model_clients.llm_base import LlmResponseError  # noqa: E402
from src.agents.services.pipeline_state import (  # noqa: E402
    NullPipelineStateStore,
)
from src.agents.services.restriction_parser_service import (  # noqa: E402
    RestrictionParserService,
)
from src.agents.services.service_entities.restriction_plan import (  # noqa: E402
    RestrictionPlan,
    RestrictionRule,
)
from src.idu_mcp.common.api_handlers.urban_data_store import (  # noqa: E402
    MODES,
    REPLAY,
    UrbanDataStore,
    UrbanDataUnavailable,
)

COL_Q = "Промт (вопрос)"
COL_SID = "scenario_id"

LOCAL = "local"
MCP_HTTP = "mcp-http"

ARM_BASE = "base"
ARM_NO_CATALOG = "no_catalog"

SCHEMA_REQUIRED = "required"
SCHEMA_OPTIONAL = "optional"


class OptionalTargetNamesRule(RestrictionRule):
    """Historical schema used only by the controlled benchmark ablation."""

    target_names: list[str] = Field(default_factory=list)


class OptionalTargetNamesPlan(RestrictionPlan):
    restriction_rules: list[OptionalTargetNamesRule] = Field(default_factory=list)


PLAN_MODELS: dict[str, type[RestrictionPlan]] = {
    SCHEMA_REQUIRED: RestrictionPlan,
    SCHEMA_OPTIONAL: OptionalTargetNamesPlan,
}

# --------------------------------------------------------------------------- #
# Taxonomy
# --------------------------------------------------------------------------- #
# Failure classes, decided from the exception at the point it is raised rather
# than by matching substrings in a message afterwards. `data_unavailable` is
# deliberately its own class and is excluded from every model-facing denominator:
# it means the offline store had no answer, which says nothing about the model.
CLS_DATA_UNAVAILABLE = "data_unavailable"
CLS_URBAN_API = "urban_api"
CLS_TOOL_EXECUTION = "tool_execution"
CLS_INVALID_PLAN = "invalid_plan"
# A plan that satisfies the schema but names nothing the executor can build
# from -- no source layer resolves, or no restriction relation survives.
# It is the model's failure, not the infrastructure's, and separating the two
# is the whole point of the taxonomy, so it does not share a class with a
# tool that genuinely broke.
CLS_UNEXECUTABLE_PLAN = "unexecutable_plan"
CLS_LLM_BACKEND = "llm_backend"
CLS_TIMEOUT = "timeout"
CLS_TOKEN_EXPIRED = "token_expired"
CLS_TRANSPORT = "transport"
CLS_OTHER = "other"

# Mutually exclusive end states; they sum to 100 % of the rows of an arm.
STATE_FULL_SUCCESS = "full_success"
STATE_PARTIAL_SPATIAL = "partial_spatial"
STATE_CLARIFICATION = "clarification"
STATE_PLANNING_FAILURE = "planning_failure"
STATE_TOOL_INFRA_FAILURE = "tool_infra_failure"
STATE_TIMEOUT = "timeout"
STATE_EMPTY = "empty"
STATE_DATA_UNAVAILABLE = "data_unavailable"

# The stage a row was in when it failed, taken from the last status event the
# pipeline emitted. "start" covers everything before the first one.
STAGE_START = "start"


def classify(exc: BaseException) -> str:
    """The failure class of an exception, by type and origin."""

    if isinstance(exc, UrbanDataUnavailable):
        return CLS_DATA_UNAVAILABLE
    if isinstance(exc, TokenExpiredError):
        return CLS_TOKEN_EXPIRED
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return CLS_TIMEOUT
    if isinstance(exc, LlmResponseError):
        return CLS_LLM_BACKEND
    name = type(exc).__name__
    if name == "ValidationError":
        return CLS_INVALID_PLAN
    # The plan builder catches the ValidationError, retries, and finally raises a
    # plain ValueError. Without this the model's own schema failure is filed as
    # `other` and lands in the infrastructure column — the exact confusion the
    # taxonomy exists to remove. Observed on gpt-oss:20b, whose empty completion
    # fails validation on every retry.
    if isinstance(exc, ValueError) and "restriction plan" in str(exc).lower():
        return CLS_INVALID_PLAN
    # Same origin, later stage: the executor raises a plain ValueError when the
    # plan parsed but resolved to nothing runnable.
    if isinstance(exc, ValueError) and (
        "no source layers found" in str(exc).lower()
        or "no valid restriction relations" in str(exc).lower()
    ):
        return CLS_UNEXECUTABLE_PLAN
    if name == "ToolError":
        # Both the Urban API handler and the geometry tools raise ToolError; the
        # message names which, and it is generated by us, not by a model.
        text = str(exc).lower()
        if "urban api" in text:
            return CLS_URBAN_API
        return CLS_TOOL_EXECUTION
    if name in {
        "ClientConnectorError",
        "ClientConnectionError",
        "ServerDisconnectedError",
        "ConnectError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectionError",
        "McpError",
    }:
        return CLS_TRANSPORT
    if isinstance(exc, json.JSONDecodeError):
        return CLS_INVALID_PLAN
    return CLS_OTHER


def end_state(record: "RunRecord") -> str:
    """One mutually-exclusive outcome per row."""

    if record.error_class == CLS_DATA_UNAVAILABLE:
        return STATE_DATA_UNAVAILABLE
    if record.error_class == CLS_TIMEOUT:
        return STATE_TIMEOUT
    if record.error_class in {CLS_INVALID_PLAN, CLS_UNEXECUTABLE_PLAN, CLS_LLM_BACKEND}:
        return STATE_PLANNING_FAILURE
    if record.error_class in {
        CLS_URBAN_API,
        CLS_TOOL_EXECUTION,
        CLS_TRANSPORT,
        CLS_TOKEN_EXPIRED,
    }:
        return STATE_TOOL_INFRA_FAILURE
    if record.error_class == CLS_OTHER:
        return STATE_TOOL_INFRA_FAILURE
    if record.clarification:
        return STATE_CLARIFICATION
    if not record.layer_counts:
        return STATE_EMPTY
    # A restrictions task is complete only with both layers; a buffers_only task
    # has no `objects` layer by design, which is exactly what the old universal
    # completion proxy got wrong.
    mode = (record.restriction_plan or {}).get("mode")
    if mode == "buffers_only":
        complete = bool(record.layer_counts)
    else:
        produced = set(record.layer_counts)
        complete = {"objects", "generators"} <= produced or {
            "Объекты в зоне ограничений",
            "Источники ограничений",
        } <= produced
    if complete and record.llm_response.strip():
        return STATE_FULL_SUCCESS
    return STATE_PARTIAL_SPATIAL


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    idx: int
    model: str
    transport: str
    arm: str
    prompt: str
    scenario_id: int
    schema_arm: str = SCHEMA_REQUIRED
    repeat_id: str = "1"
    run_id: str = ""
    restriction_plan: dict | None = None
    model_plan: dict | None = None
    layer_counts: dict[str, int] = field(default_factory=dict)
    layer_files: dict[str, str] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)
    llm_response: str = ""
    clarification: str | None = None
    error: str | None = None
    error_class: str | None = None
    error_stage: str | None = None
    missing_data: dict | None = None
    end_state: str = ""
    duration_sec: float = 0.0

    def as_json(self) -> dict:
        data = self.__dict__.copy()
        return data


# The plan is not part of the event stream — there is no ``plan`` event type in
# ``RestrictionsResponse`` — so scraping the stream for it (as the SSE harness
# did) finds nothing. Wrapping the builder is the only way to get the object
# without changing the service's public contract, and it captures exactly what
# the pipeline went on to execute.
#
# The slot is a ContextVar, not an attribute on the builder, because rows share
# one service: asyncio copies the context when it creates a task, so each row's
# worker gets its own slot and concurrent rows cannot pick up one another's plan.
# The wrapper itself is installed once per builder — installing it per row would
# stack the wrappers and route every plan through every one of them.
#
# The slot holds a *list*, and the capture appends to it rather than re-setting
# the variable. Context copies are shallow, so the inner task the pipeline runs
# in shares the list object with the row that created it; a plain ``set()`` there
# would land in the copy and be invisible to the caller — which is exactly what
# happens across ``asyncio.wait_for``, since it wraps the coroutine in a task.
#
# Two plans are captured, not one, and the difference between them is the whole
# point. ``build_plan`` returns the plan *after* canonicalisation and mode
# validation, and that step is lossy in a way that destroys evidence: a plan
# whose ``restriction_rules`` lack ``target_names`` has every rule dropped, which
# flips the mode to ``needs_clarification``, which then blanks ``target_entities``
# on the way out. Scoring entity correctness on that record reports "the model
# named no target entity" about a model that named it correctly at the top level.
# ``_request_plan`` returns what the model actually emitted, so entity and
# parameter accuracy are scored there, and the canonicalised plan says what the
# pipeline agreed to execute.
_PLAN_SLOT: ContextVar[list | None] = ContextVar("plan_slot", default=None)
_MODEL_PLAN_SLOT: ContextVar[list | None] = ContextVar("model_plan_slot", default=None)
_CAPTURE_FLAG = "_inproc_plan_capture_installed"


def install_plan_capture(service: RestrictionParserService) -> None:
    """Record both the model's plan and the pipeline's, into this row's slots."""

    builder = service.plan_builder
    if getattr(builder, _CAPTURE_FLAG, False):
        return
    original = builder.build_plan

    async def capturing(*args, **kwargs):
        plan = await original(*args, **kwargs)
        slot = _PLAN_SLOT.get()
        if slot is not None:
            slot.append(plan)
        return plan

    builder.build_plan = capturing  # type: ignore[method-assign]

    # ``_request_plan`` is the builder's own private step, so it is wrapped only
    # when it is there: a stand-in builder in a test legitimately has no such
    # method, and the pipeline plan alone is still worth capturing.
    original_request = getattr(builder, "_request_plan", None)
    if original_request is not None:

        async def capturing_request(*args, **kwargs):
            plan = await original_request(*args, **kwargs)
            slot = _MODEL_PLAN_SLOT.get()
            if slot is not None:
                slot.append(plan)
            return plan

        builder._request_plan = capturing_request  # type: ignore[method-assign]
    setattr(builder, _CAPTURE_FLAG, True)


def open_plan_slot() -> tuple[list, list]:
    """Start fresh slots for the current row: (pipeline plan, model plan)."""

    slot: list = []
    model_slot: list = []
    _PLAN_SLOT.set(slot)
    _MODEL_PLAN_SLOT.set(model_slot)
    return slot, model_slot


def captured_plan(slot: list) -> dict | None:
    """The plan this row built, as plain JSON, or None if it never got that far."""

    if not slot:
        return None
    try:
        return slot[-1].model_dump(mode="json")
    except AttributeError:
        return None


# --------------------------------------------------------------------------- #
# Wiring — the dependency graph, by hand
# --------------------------------------------------------------------------- #
def build_service(
    llm_host: str,
    urban_api_url: str,
    chat_storage_url: str,
    no_catalog: bool,
    schema_arm: str = SCHEMA_REQUIRED,
) -> RestrictionParserService:
    """A RestrictionParserService with nothing behind it but the model server.

    Deliberately does not import ``src.agents.dependencies.dependencies``: that
    module builds the whole application graph at import time and is wired for
    FastAPI's ``Depends``. The ChatStorage client is constructed because the
    constructor asks for one, and is never called — ``persist_history=False`` and
    ``DISABLE_CHAT_HISTORY`` both close that path.
    """

    service = RestrictionParserService(
        llm_host,
        ChatStorageApiClient(JsonApiHandler(chat_storage_url)),
        UrbanApiClient(JsonApiHandler(urban_api_url)),
        NullPipelineStateStore(),
        ablation_no_catalog=no_catalog,
    )
    service.plan_builder.plan_model = PLAN_MODELS[schema_arm]
    return service


def build_client(
    transport: str, token: str, urban_api_url: str, idu_mcp_url: str
) -> LocalIduMcpClient | IduMcpClient:
    if transport == LOCAL:
        return LocalIduMcpClient(token, urban_api_url=urban_api_url)
    return IduMcpClient(build_mcp_client(idu_mcp_url, auth=token), mcp_url=idu_mcp_url)


# --------------------------------------------------------------------------- #
# One query
# --------------------------------------------------------------------------- #
async def run_one(
    service: RestrictionParserService,
    client: LocalIduMcpClient | IduMcpClient,
    *,
    idx: int,
    model: str,
    prompt: str,
    scenario_id: int,
    transport: str,
    arm: str,
    temperature: float,
    timeout: float,
    layers_dir: Path | None,
    schema_arm: str = SCHEMA_REQUIRED,
    repeat_id: str = "1",
    run_id: str = "",
) -> RunRecord:
    record = RunRecord(
        idx=idx,
        model=model,
        transport=transport,
        arm=arm,
        prompt=prompt,
        scenario_id=scenario_id,
        schema_arm=schema_arm,
        repeat_id=repeat_id,
        run_id=run_id,
    )
    install_plan_capture(service)
    # This row's own slot, even though the service (and its builder) is shared.
    plan_slot, model_plan_slot = open_plan_slot()
    stage = STAGE_START
    started = time.time()
    stage_started = started
    text_parts: list[str] = []

    async def drive() -> None:
        nonlocal stage, stage_started
        async for event in service.run_restriction_execution_pipline(
            mcp_client=client,
            temperature=temperature,
            model=model,
            user_query=prompt,
            scenario_id=scenario_id,
            persist_history=False,
        ):
            etype = event.get("type")
            content = event.get("content", {}) or {}
            if etype == "status":
                now = time.time()
                record.stages.append(
                    {"stage": stage, "seconds": round(now - stage_started, 2)}
                )
                stage = content.get("status", stage)
                stage_started = now
            elif etype == "feature_collection":
                name = str(content.get("name"))
                collection = content.get("feature_collection") or {}
                features = collection.get("features", [])
                record.layer_counts[name] = len(features)
                if layers_dir is not None:
                    path = _write_layer(layers_dir, idx, name, collection)
                    record.layer_files[name] = str(path)
                # The collection is hundreds of megabytes on the large
                # scenarios; nothing downstream needs it in memory once it is
                # counted and on disk.
                del collection, features
            elif etype == "chunk":
                text = content.get("text")
                if text:
                    text_parts.append(str(text))
            elif etype == "tool_call":
                record.tool_calls.append(
                    {
                        "stage": content.get("execution_mode"),
                        "source": content.get("mcp_source"),
                        "calls": _tool_call_names(content.get("tool_calls", [])),
                    }
                )

    try:
        await asyncio.wait_for(drive(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        record.error = f"Timeout: no completion within {timeout}s"
        record.error_class = CLS_TIMEOUT
        record.error_stage = stage
    except UrbanDataUnavailable as exc:
        record.error = str(exc)
        record.error_class = CLS_DATA_UNAVAILABLE
        record.error_stage = stage
        record.missing_data = {"endpoint": exc.endpoint, "params": exc.params}
    except Exception as exc:  # noqa: BLE001 — every failure is a datum here
        record.error = f"{type(exc).__name__}: {exc}"
        record.error_class = classify(exc)
        record.error_stage = stage
        if record.error_class == CLS_OTHER:
            # An unclassified failure is a gap in the taxonomy, so keep enough to
            # close it rather than only the message.
            record.error = record.error + "\n" + traceback.format_exc(limit=8)
    finally:
        record.stages.append(
            {"stage": stage, "seconds": round(time.time() - stage_started, 2)}
        )

    record.restriction_plan = captured_plan(plan_slot)
    record.model_plan = captured_plan(model_plan_slot)
    record.duration_sec = round(time.time() - started, 2)
    joined = "".join(text_parts).strip()
    plan_mode = (record.restriction_plan or {}).get("mode")
    if plan_mode == "needs_clarification":
        record.clarification = joined or (record.restriction_plan or {}).get(
            "clarification_question"
        )
    record.llm_response = joined
    record.end_state = end_state(record)
    return record


def _tool_call_names(tool_calls: list) -> list[str]:
    """Just the names — the arguments carry whole layers."""

    names = []
    for call in tool_calls:
        if isinstance(call, dict):
            function = call.get("function") or {}
            names.append(str(function.get("name") or call.get("name") or "?"))
        else:
            names.append(str(getattr(call, "name", "?")))
    return names


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60]


def _write_layer(layers_dir: Path, idx: int, name: str, collection: dict) -> Path:
    row_dir = layers_dir / f"{idx:05d}"
    row_dir.mkdir(parents=True, exist_ok=True)
    path = row_dir / f"{_safe_name(name)}.geojson"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(collection, handle, ensure_ascii=False)
    return path


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def read_table(path: Path) -> pd.DataFrame:
    """Read the dataset whatever delimiter it uses.

    The gold export is semicolon-separated (``gold_parser.load_gold`` hardcodes
    ``sep=";"``) while the paraphrase expansion is written with commas. Sniffing
    covers both; a hardcoded separator silently yields a single fused column and
    fails later as a missing-column error.
    """

    return pd.read_csv(path, sep=None, engine="python")


def load_dataset(
    path: Path,
    limit: int | None,
    sample_per_scenario: int = 0,
    sample_seed: int = 20260831,
) -> pd.DataFrame:
    frame = read_table(path)
    missing = [column for column in (COL_Q, COL_SID) if column not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing required column(s) {missing}; "
            f"found {list(frame.columns)}"
        )
    frame = frame[frame[COL_SID].notna() & frame[COL_Q].notna()].reset_index(drop=True)
    if sample_per_scenario:
        # Cluster-balanced robustness slice: every scenario contributes the same
        # maximum number of prompts, and the fixed seed makes the selected rows
        # reproducible.  Sampling happens before --limit by design.
        frame = (
            frame.groupby(COL_SID, group_keys=False)
            .sample(n=sample_per_scenario, random_state=sample_seed, replace=False)
            .sort_index()
            .reset_index(drop=True)
        )
    return frame.head(limit) if limit else frame


def done_indices(path: Path) -> set[int]:
    """Row indices already written, so a run resumes instead of restarting.

    Rows that failed are *not* counted as done: a re-run after fixing the cause
    (a refilled offline store, a restarted model server) must redo them, which is
    what the SSE harness got wrong and had to grow a repair script for.
    """

    if not path.exists():
        return set()
    finished: set[int] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") is None:
                finished.add(int(row["idx"]))
    return finished


async def run_arm(
    frame: pd.DataFrame,
    *,
    model: str,
    arm: str,
    transport: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    layers_dir = out_dir / "layers" if args.save_layers else None
    already = done_indices(results_path)
    todo = [i for i in range(len(frame)) if i not in already]
    print(
        f"  {model} / {arm} / {transport}: {len(todo)} rows to run "
        f"({len(already)} already done) -> {results_path}"
    )
    if not todo:
        return

    service = build_service(
        args.llm_host,
        args.urban_api_url,
        args.chat_storage_url,
        arm == ARM_NO_CATALOG,
        args.schema_arm,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters: dict[str, int] = {}
    consecutive_backend_failures = 0
    abort = asyncio.Event()
    fatal_reason: str | None = None

    async def worker(idx: int) -> None:
        nonlocal consecutive_backend_failures, fatal_reason
        row = frame.iloc[idx]
        # One client per row: it holds the per-call log, and the HTTP variant
        # holds a session.
        client = build_client(
            transport, args.token, args.urban_api_url, args.idu_mcp_url
        )
        async with semaphore:
            if abort.is_set():
                return
            record = await run_one(
                service,
                client,
                idx=idx,
                model=model,
                prompt=str(row[COL_Q]),
                scenario_id=int(row[COL_SID]),
                transport=transport,
                arm=arm,
                temperature=args.temperature,
                timeout=args.timeout,
                layers_dir=layers_dir,
                schema_arm=args.schema_arm,
                repeat_id=args.repeat_id,
                run_id=args.run_id,
            )
        async with write_lock:
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
            counters[record.end_state] = counters.get(record.end_state, 0) + 1
            if record.error_class in {CLS_LLM_BACKEND, CLS_TRANSPORT}:
                consecutive_backend_failures += 1
            else:
                consecutive_backend_failures = 0
            if consecutive_backend_failures >= args.fail_fast_after:
                abort.set()
            done = sum(counters.values())
            if done == 1:
                try:
                    runtime = verify_runtime_context(args, model)
                    update_runtime_manifest(args, model, runtime)
                    actual_context = int(runtime["context_length"])
                    if actual_context != args.expected_context:
                        fatal_reason = (
                            f"runtime context mismatch for {model}: expected "
                            f"{args.expected_context}, /api/ps reports {actual_context}"
                        )
                        abort.set()
                except Exception as exc:  # noqa: BLE001 - failed verification is fatal
                    fatal_reason = (
                        f"could not verify runtime context for {model}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    abort.set()
            if done % args.progress_every == 0 or done == len(todo):
                print(f"    {done}/{len(todo)}  {counters}", flush=True)

    await asyncio.gather(*(worker(idx) for idx in todo))
    if abort.is_set():
        if fatal_reason:
            raise RuntimeError(fatal_reason)
        raise RuntimeError(
            f"fail-fast: {consecutive_backend_failures} consecutive backend/transport "
            f"failures in {model}/{arm}/{transport}; refusing to contaminate the run"
        )
    print(f"    finished {model}/{arm}/{transport}: {counters}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _command_output(*command: str) -> str | None:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, encoding="utf-8"
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _ollama_json(host: str, endpoint: str, payload: dict | None = None) -> dict:
    base = host.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        f"{base}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    with request.urlopen(req, timeout=15) as response:  # noqa: S310 - configured host
        return json.load(response)


def preflight(args: argparse.Namespace) -> dict:
    """Fail before row 1 when Ollama or a requested model is unavailable."""

    try:
        version = _ollama_json(args.llm_host, "/api/version").get("version")
        tags = _ollama_json(args.llm_host, "/api/tags").get("models", [])
    except Exception as exc:  # noqa: BLE001 - converted to a concise hard failure
        raise SystemExit(
            f"Ollama preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    by_name = {item.get("name"): item for item in tags}
    missing = [model for model in args.models if model not in by_name]
    if missing:
        raise SystemExit(f"Ollama preflight: missing model(s): {', '.join(missing)}")
    models: dict[str, dict] = {}
    for model in args.models:
        show = _ollama_json(args.llm_host, "/api/show", {"model": model})
        tag = by_name[model]
        models[model] = {
            "digest": tag.get("digest"),
            "size": tag.get("size"),
            "modified_at": tag.get("modified_at"),
            "details": show.get("details"),
            "capabilities": show.get("capabilities"),
            "parameters": show.get("parameters"),
        }
    return {"version": version, "models": models}


def verify_runtime_context(args: argparse.Namespace, model: str) -> dict:
    """Read the allocation Ollama actually made, not merely requested settings."""

    running = _ollama_json(args.llm_host, "/api/ps").get("models", [])
    match = next(
        (
            item
            for item in running
            if item.get("name") == model or item.get("model") == model
        ),
        None,
    )
    if match is None:
        raise RuntimeError(f"{model} is not present in /api/ps after a completed row")
    return {
        "name": match.get("name"),
        "digest": match.get("digest"),
        "context_length": match.get("context_length"),
        "size": match.get("size"),
        "size_vram": match.get("size_vram"),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def update_runtime_manifest(
    args: argparse.Namespace, model: str, runtime: dict
) -> None:
    path = Path(args.out_dir) / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("runtime_verification", {})[model] = runtime
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_manifest(args: argparse.Namespace, ollama: dict) -> Path:
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset)
    repo_root = Path(__file__).resolve().parents[2]
    tracked_sources = [
        repo_root / "benchmarks/harness/inproc_runner.py",
        repo_root / "benchmarks/harness/run_validated_matrix.py",
        repo_root / "benchmarks/eval/validated_statistics.py",
        repo_root / "src/agents/model_clients/ollama_adapter.py",
        repo_root / "src/agents/services/restriction_catalog.py",
        repo_root / "src/agents/services/restriction_parser_service.py",
        repo_root / "src/agents/services/service_entities/restriction_plan.py",
        repo_root / "src/agents/mcp_clients/local_idu_mcp_client.py",
        repo_root / "src/idu_mcp/tools_services/geometry_tools.py",
    ]
    status = _command_output("git", "status", "--porcelain") or ""
    diff = _command_output("git", "diff", "--binary", "HEAD") or ""
    safe_args = vars(args).copy()
    safe_args["token"] = "<redacted>" if args.token else ""
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "repeat_id": args.repeat_id,
        "schema_arm": args.schema_arm,
        "command": safe_args,
        "dataset": {
            "path": str(dataset),
            "sha256": _sha256_file(dataset),
            "rows": len(
                load_dataset(
                    dataset,
                    args.limit,
                    args.sample_per_scenario,
                    args.sample_seed,
                )
            ),
        },
        "schema": {
            "sha256": _sha256_bytes(
                json.dumps(
                    PLAN_MODELS[args.schema_arm].model_json_schema(), sort_keys=True
                ).encode("utf-8")
            ),
            "model": PLAN_MODELS[args.schema_arm].__name__,
        },
        "git": {
            "commit": _command_output("git", "rev-parse", "HEAD"),
            "status": status.splitlines(),
            "working_diff_sha256": _sha256_bytes(diff.encode("utf-8")),
        },
        "source_sha256": {
            str(path.relative_to(repo_root)): _sha256_file(path)
            for path in tracked_sources
        },
        "ollama": ollama,
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "gpu": _command_output(
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ),
            "ollama_context_length": os.getenv("OLLAMA_CONTEXT_LENGTH"),
            "ollama_max_loaded_models": os.getenv("OLLAMA_MAX_LOADED_MODELS"),
            "ollama_flash_attention": os.getenv("OLLAMA_FLASH_ATTENTION"),
            "ollama_kv_cache_type": os.getenv("OLLAMA_KV_CACHE_TYPE"),
            "ollama_request_num_ctx": os.getenv("OLLAMA_REQUEST_NUM_CTX"),
            "ollama_think_levels": os.getenv("OLLAMA_THINK_LEVELS"),
            "llm_backend": os.getenv("LLM_BACKEND", "openai"),
        },
    }
    path = root / "run_manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


async def main_async(args: argparse.Namespace) -> None:
    ollama = preflight(args)
    manifest = write_manifest(args, ollama)
    print(f"manifest: {manifest}")
    frame = load_dataset(
        Path(args.dataset),
        args.limit,
        args.sample_per_scenario,
        args.sample_seed,
    )
    print(f"dataset: {len(frame)} rows from {args.dataset}")
    store = UrbanDataStore()
    print(f"urban data store: {store.stats()}")
    if store.mode == REPLAY and store.stats()["entries"] == 0:
        raise SystemExit(
            "--urban-data replay but the store is empty; run "
            "benchmarks/harness/prefetch_scenarios.py first (with the VPN up)"
        )
    for model in args.models:
        for arm in args.arms:
            for transport in args.transports:
                out_dir = Path(args.out_dir) / _safe_name(model) / f"{arm}--{transport}"
                await run_arm(
                    frame,
                    model=model,
                    arm=arm,
                    transport=transport,
                    args=args,
                    out_dir=out_dir,
                )
    print(f"urban data store: {store.stats()}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="benchmarks/data/gold/exp_data.csv")
    parser.add_argument("--out-dir", default="benchmarks/data/results_inproc")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[ARM_BASE],
        choices=[ARM_BASE, ARM_NO_CATALOG],
        help="ablation arms; no_catalog builds the plan without domain-catalog "
        "grounding",
    )
    parser.add_argument(
        "--transports",
        nargs="+",
        default=[LOCAL],
        choices=[LOCAL, MCP_HTTP],
        help="local calls the IDU tools in-process; mcp-http goes through the "
        "MCP server, for the transport-cost comparison",
    )
    parser.add_argument(
        "--urban-data",
        default=os.getenv("URBAN_DATA_MODE", "live"),
        choices=list(MODES),
        help="live: always the network; record: fetch and store; replay: serve "
        "from the store and never touch the network",
    )
    parser.add_argument(
        "--urban-data-dir", default=os.getenv("URBAN_DATA_DIR", "runtime/urban_data")
    )
    parser.add_argument(
        "--llm-host",
        default=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
        help="model server; with LLM_BACKEND=openai (the default) /v1 is "
        "appended, which is what Ollama's OpenAI-compatible API expects",
    )
    parser.add_argument(
        "--urban-api-url", default=os.getenv("URBAN_API_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--idu-mcp-url",
        default=os.getenv("IDU_MCP_SERVER", "http://localhost:8001/mcp"),
    )
    parser.add_argument(
        "--chat-storage-url",
        default=os.getenv("CHAT_STORAGE", "http://unused.invalid"),
        help="never called (persist_history=False); the constructor asks for it",
    )
    parser.add_argument("--token", default=os.getenv("URBAN_API_JWT", ""))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample-per-scenario",
        type=int,
        default=0,
        help="deterministic cluster-balanced sample size per scenario (0 = all)",
    )
    parser.add_argument("--sample-seed", type=int, default=20260831)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="console verbosity; structured plans remain in results.jsonl",
    )
    parser.add_argument(
        "--schema-arm",
        choices=[SCHEMA_REQUIRED, SCHEMA_OPTIONAL],
        default=SCHEMA_REQUIRED,
        help="required is the production contract; optional reproduces the historical "
        "target_names schema for a controlled ablation",
    )
    parser.add_argument("--repeat-id", default="1")
    parser.add_argument("--run-id", default="manual")
    parser.add_argument(
        "--fail-fast-after",
        type=int,
        default=3,
        help="abort an arm after this many consecutive backend/transport failures",
    )
    parser.add_argument(
        "--expected-context",
        type=int,
        default=16384,
        help="hard runtime gate checked against Ollama /api/ps after row 1",
    )
    parser.add_argument(
        "--save-layers",
        action="store_true",
        help="write every produced layer as GeoJSON for geometry_eval.py",
    )
    args = parser.parse_args(argv)
    args.limit = args.limit or None
    return args


def main() -> None:
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    # The store is read from the environment wherever a JsonApiHandler is built,
    # including inside idu_mcp, so the mode is set here rather than threaded
    # through every construction site.
    os.environ["URBAN_DATA_MODE"] = args.urban_data
    os.environ["URBAN_DATA_DIR"] = args.urban_data_dir
    if not args.token and args.urban_data != REPLAY:
        raise SystemExit(
            "--token is required unless --urban-data replay (offline runs never "
            "call Urban API)"
        )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
