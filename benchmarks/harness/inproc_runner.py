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
import json
import os
import sys
import time
import traceback
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The pipeline must never touch ChatStorage on a benchmark run. persist_history
# is passed False as well; this makes the read path impossible too, and has to be
# set before the services are imported since the flag is read at call time.
os.environ.setdefault("DISABLE_CHAT_HISTORY", "1")
os.environ.setdefault("DISABLE_PIPELINE_STATE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

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
    if record.error_class in {CLS_INVALID_PLAN, CLS_LLM_BACKEND}:
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
    restriction_plan: dict | None = None
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
_PLAN_SLOT: ContextVar[list | None] = ContextVar("plan_slot", default=None)
_CAPTURE_FLAG = "_inproc_plan_capture_installed"


def install_plan_capture(service: RestrictionParserService) -> None:
    """Make ``plan_builder`` record the plan it returns into this row's slot."""

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
    setattr(builder, _CAPTURE_FLAG, True)


def open_plan_slot() -> list:
    """Start a fresh slot for the current row and return it."""

    slot: list = []
    _PLAN_SLOT.set(slot)
    return slot


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
) -> RestrictionParserService:
    """A RestrictionParserService with nothing behind it but the model server.

    Deliberately does not import ``src.agents.dependencies.dependencies``: that
    module builds the whole application graph at import time and is wired for
    FastAPI's ``Depends``. The ChatStorage client is constructed because the
    constructor asks for one, and is never called — ``persist_history=False`` and
    ``DISABLE_CHAT_HISTORY`` both close that path.
    """

    return RestrictionParserService(
        llm_host,
        ChatStorageApiClient(JsonApiHandler(chat_storage_url)),
        UrbanApiClient(JsonApiHandler(urban_api_url)),
        NullPipelineStateStore(),
        ablation_no_catalog=no_catalog,
    )


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
) -> RunRecord:
    record = RunRecord(
        idx=idx,
        model=model,
        transport=transport,
        arm=arm,
        prompt=prompt,
        scenario_id=scenario_id,
    )
    install_plan_capture(service)
    # This row's own slot, even though the service (and its builder) is shared.
    plan_slot = open_plan_slot()
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
def load_dataset(path: Path, limit: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in (COL_Q, COL_SID) if column not in frame.columns]
    if missing:
        raise SystemExit(
            f"{path} is missing required column(s) {missing}; "
            f"found {list(frame.columns)}"
        )
    frame = frame[frame[COL_SID].notna() & frame[COL_Q].notna()].reset_index(drop=True)
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
        args.llm_host, args.urban_api_url, args.chat_storage_url, arm == ARM_NO_CATALOG
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    counters: dict[str, int] = {}

    async def worker(idx: int) -> None:
        row = frame.iloc[idx]
        # One client per row: it holds the per-call log, and the HTTP variant
        # holds a session.
        client = build_client(
            transport, args.token, args.urban_api_url, args.idu_mcp_url
        )
        async with semaphore:
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
            )
        async with write_lock:
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
            counters[record.end_state] = counters.get(record.end_state, 0) + 1
            done = sum(counters.values())
            if done % args.progress_every == 0 or done == len(todo):
                print(f"    {done}/{len(todo)}  {counters}", flush=True)

    await asyncio.gather(*(worker(idx) for idx in todo))
    print(f"    finished {model}/{arm}/{transport}: {counters}")


async def main_async(args: argparse.Namespace) -> None:
    frame = load_dataset(Path(args.dataset), args.limit)
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
    parser.add_argument("--progress-every", type=int, default=10)
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
