"""Latency benchmark of the deployed document-QA agent (GET /documents/qa/stream).

Every case is sent to a running gMART over HTTP. The client timestamps each SSE
event, which gives the stage breakdown for any deployed version, including one
without server metrics. A version that logs ``DVD run metrics`` (Stage 0) also
gets the exact server-side record attached: after the run the script downloads
``/system/logs`` once and matches the lines by ``request_id``.

Single-turn cases run anonymously over the shared index. Multi-turn cases (class
G) need a chat, so they run only with a user token in the environment variable
named by ``--token-env``; without it they are skipped, never faked.

    python evaluation/documents_latency/run_http.py --base-url http://host:31004 \
        --label baseline --out benchmarks/data/documents_latency/baseline.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
METRICS_LINE = "DVD run metrics "
# Client-side stage of each status the agent announces; the stage lasts until
# the next event. ``self_review`` covers both review-context preparation and
# the critic call, which only the server record separates.
STATUS_STAGE = {
    "context_check": "context_check",
    "retrieval_planning": "planner",
    "searching": "retrieval",
    "context_processing": "context_prepare",
    "answer_drafting": "answer_generation",
    "self_review": "critic",
    "finalizing": "finalize",
}


def load_cases(path: Path, ids, classes):
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return [
        case
        for case in cases
        if (not ids or case["id"] in ids) and (not classes or case["class"] in classes)
    ]


async def stream_turn(client, base_url, query, *, model, chat_id, token, timeout):
    params = {"request": query}
    if model:
        params["model"] = model
    if chat_id:
        params["chat_id"] = chat_id
    headers = {"Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    started = time.perf_counter()
    events = []
    async with client.stream(
        "GET",
        base_url.rstrip("/") + "/documents/qa/stream",
        params=params,
        headers=headers,
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:].strip())
            event["_t_ms"] = round((time.perf_counter() - started) * 1000, 1)
            events.append(event)
            content = event.get("content") or {}
            if event.get("type") == "chunk" and content.get("done"):
                break
            if event.get("type") == "error":
                break
    return events, round((time.perf_counter() - started) * 1000, 1)


def summarize(events, total_ms):
    stages: dict[str, float] = {}
    current, since = None, 0.0
    ttft = None
    answer = []
    attempts = 1
    tool_calls = 0
    request_id = chat_id = error = None
    for event in events:
        kind, content = event.get("type"), event.get("content") or {}
        if current is not None:
            stages[current] = round(
                stages.get(current, 0.0) + event["_t_ms"] - since, 1
            )
            current = None
        if kind == "pipeline_started":
            request_id = content.get("request_id")
        elif kind == "status":
            current, since = STATUS_STAGE.get(content.get("status")), event["_t_ms"]
            if match := re.search(r"попытка (\d+)", content.get("text") or ""):
                attempts = max(attempts, int(match[1]))
        elif kind == "tool_call":
            tool_calls += len(content.get("tool_calls") or [])
        elif kind == "chunk" and content.get("text"):
            ttft = event["_t_ms"] if ttft is None else ttft
            answer.append(content["text"])
        elif kind == "service_event":
            chat_id = (content.get("event") or {}).get("chat_id") or chat_id
        elif kind == "error":
            error = content.get("message") or "error"
    return {
        "request_id": request_id,
        "chat_id": chat_id,
        "total_ms": total_ms,
        "ttft_ms": ttft,
        "client_stages_ms": stages,
        "attempts": attempts,
        "tool_calls": tool_calls,
        "answer": "".join(answer),
        "error": error,
    }


async def run_case(client, args, case, run, token):
    turns = case.get("turns") or [case["query"]]
    if len(turns) > 1 and not token:
        return {
            "id": case["id"],
            "class": case["class"],
            "run": run,
            "skipped": "no_token",
        }
    chat_id, turn_results = None, []
    for query in turns:
        try:
            events, total = await stream_turn(
                client,
                args.base_url,
                query,
                model=args.model,
                chat_id=chat_id,
                token=token,
                timeout=args.timeout,
            )
            result = summarize(events, total)
        except Exception as exc:  # one failed case must not stop the benchmark
            result = {"error": f"{type(exc).__name__}: {exc}", "total_ms": None}
        chat_id = result.get("chat_id") or chat_id
        turn_results.append({"query": query, **result})
    # The measured turn is the last one; earlier turns only build the dialogue.
    return {
        "id": case["id"],
        "class": case["class"],
        "run": run,
        "label": args.label,
        **turn_results[-1],
        "setup_turns": turn_results[:-1],
    }


def attach_server_metrics(base_url, records):
    try:
        text = httpx.get(base_url.rstrip("/") + "/system/logs", timeout=120).text
    except httpx.HTTPError as exc:
        print(f"server logs unavailable: {exc}")
        return
    by_request = {}
    for line in text.splitlines():
        if METRICS_LINE in line:
            try:
                record = json.loads(line.split(METRICS_LINE, 1)[1])
            except ValueError:
                continue
            by_request[record.get("request_id")] = record
    for record in records:
        if record.get("request_id") in by_request:
            record["server"] = by_request[record["request_id"]]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--label", required=True, help="e.g. baseline, adaptive-critic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--classes", nargs="*")
    parser.add_argument("--repeat", type=int, default=1)
    # Sequential by default: concurrent runs share one LLM server and inflate latency.
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--model")
    parser.add_argument("--token-env", default="GMART_TOKEN")
    parser.add_argument("--no-server-metrics", action="store_true")
    args = parser.parse_args()

    token = os.getenv(args.token_env) or None
    cases = load_cases(args.cases, args.ids, args.classes)
    semaphore = asyncio.Semaphore(args.concurrency)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(verify=False) as client:

        async def one(case, run):
            async with semaphore:
                record = await run_case(client, args, case, run, token)
                status = record.get("skipped") or record.get("error") or "ok"
                print(
                    f"{case['id']} run={run} total_ms={record.get('total_ms')} {status}"
                )
                return record

        records = await asyncio.gather(
            *(one(case, run) for run in range(args.repeat) for case in cases)
        )
    if not args.no_server_metrics:
        attach_server_metrics(args.base_url, records)
    with args.out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{len(records)} records -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
