"""Immutable real-LLM episodes, outcome checks and mandatory independent judge.

Each invocation creates a new directory; failed episodes are never replaced.
--episodes is diagnostic only unless all fifteen episodes have finished.
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import dotenv_values

from .acceptance import verify_result
from .judge import evaluate
from .preflight import run as preflight
from .scenarios import episodes
from .transport import headers_for, parse_events, save, stored_context

ROOT = Path(__file__).resolve().parents[3]


def application_digest(root):
    root = Path(root)
    files = [
        *root.glob("src/agents/**/*.py"),
        *root.glob("src/common/**/*.py"),
        root / "src/__init__.py",
        root / "src/__version__.py",
    ]
    records = {
        p.relative_to(root)
        .as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n"))
        .hexdigest()
        for p in files
    }
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def fingerprint(config):
    expected_application = application_digest(ROOT)
    actual_application = subprocess.check_output(
        [
            "docker",
            "exec",
            "gmart-sdk-local-agents-1",
            "python",
            "-c",
            "import hashlib,json\nfrom pathlib import Path\n"
            + inspect.getsource(application_digest)
            + '\nprint(application_digest("/app"))',
        ],
        text=True,
        timeout=30,
    ).strip()
    if actual_application != expected_application:
        raise ValueError(
            "Running agents image does not match this checkout; rebuild before acceptance"
        )
    digest = hashlib.sha256()
    for folder in (
        "src",
        "tests/integration/industrial",
        "tests/integration/local_stack",
    ):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml"}:
                digest.update(path.relative_to(ROOT).as_posix().encode())
                digest.update(path.read_bytes())
    containers = subprocess.check_output(
        [
            "docker",
            "ps",
            "--filter",
            "label=com.docker.compose.project=gmart-sdk-local",
            "--format",
            "{{.ID}}",
        ],
        text=True,
        timeout=20,
    ).split()
    if not containers:
        raise ValueError("Local stack is not running")
    inspected = json.loads(
        subprocess.check_output(
            ["docker", "inspect", *containers], text=True, timeout=20
        )
    )
    deployment = {
        c["Name"]: {
            "image": c["Image"],
            "env_sha256": hashlib.sha256(
                json.dumps(sorted(c["Config"].get("Env") or [])).encode()
            ).hexdigest(),
        }
        for c in inspected
    }
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=15
        ).strip(),
        "source_sha256": digest.hexdigest(),
        "application_sha256": actual_application,
        "deployment": deployment,
        "model": config["LLM_MODEL"],
        "base_url": config["LLM_BASE_URL"],
    }


def series_verdict(report):
    expected = {(e["id"], e["formulation"]) for e in episodes()}
    actual = [(e["id"], e["formulation"]) for e in report.get("episodes", [])]
    requests = [
        t.get("request_id")
        for e in report.get("episodes", [])
        for t in e.get("turns", [])
    ]
    return (
        bool(report.get("preflight_passed"))
        and len(actual) == 15
        and set(actual) == expected
        and (
            len(requests) == 30
            and len(requests) == len(set(requests))
            and None not in requests
            and report.get("fingerprint") == report.get("final_fingerprint")
            and bool(report.get("fingerprint"))
            and all(e.get("passed") is True for e in report["episodes"])
        )
    )


async def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = dotenv_values(args.env_file)
    initial = fingerprint(config)
    report = {
        "fingerprint": initial,
        "episodes": [],
        "passed": False,
        "mode": "acceptance" if args.episodes == 15 else "diagnostic",
    }
    save(output / "report.json", report)
    report["preflight_passed"] = await preflight(config, output / "preflight")
    save(output / "report.json", report)
    if not report["preflight_passed"]:
        return 1
    async with httpx.AsyncClient(timeout=660, trust_env=False) as http:
        for episode in episodes()[: args.episodes]:
            if fingerprint(config) != initial:
                raise RuntimeError("Build changed during immutable series")
            directory = output / f"{episode['id']}-{episode['formulation']}"
            directory.mkdir()
            row = {
                "id": episode["id"],
                "formulation": episode["formulation"],
                "turns": [],
                "passed": False,
            }
            report["episodes"].append(row)
            save(output / "report.json", report)
            chat_id, context = None, {}
            for i, query in enumerate(episode["queries"]):
                if fingerprint(config) != initial:
                    raise RuntimeError("Build changed during immutable series")
                turn_dir = directory / f"turn-{i+1}"
                turn_dir.mkdir()
                turn = {"request_id": str(uuid4()), "query": query, "passed": False}
                row["turns"].append(turn)
                save(output / "report.json", report)
                print(
                    f"START {episode['id']} {episode['formulation']}/3 turn {i+1}",
                    flush=True,
                )
                params = {
                    "request": query,
                    "scenario_id": episode["scenario_id"],
                    "request_id": turn["request_id"],
                    "model": config["LLM_MODEL"],
                    "temperature": 0,
                }
                if chat_id:
                    params["chat_id"] = chat_id
                started = time.monotonic()
                try:
                    events = parse_events(
                        await http.get(
                            "http://localhost:18000/orchestrator/route/stream",
                            params=params,
                            headers=await headers_for(http, config),
                        )
                    )
                    save(turn_dir / "events.json", events)
                    final = next(
                        e["content"]
                        for e in reversed(events)
                        if e["type"] == "orchestrator_final"
                    )
                    turn["final"] = final
                    chat_id = chat_id or next(
                        (
                            e["content"]["event"]["chat_id"]
                            for e in events
                            if e["type"] == "service_event"
                            and e["content"].get("event", {}).get("storage_event_type")
                            == "chat_created"
                        ),
                        None,
                    )
                    headers = await headers_for(http, config)
                    context = await stored_context(
                        http, headers, events, final, turn_dir, chat_id=chat_id
                    )
                    save(turn_dir / "context.json", context)
                    verdict = verify_result(final, context, episode["contracts"][i])
                    save(turn_dir / "checks.json", verdict)
                    turn["checks"] = verdict
                    replay = parse_events(
                        await http.get(
                            "http://localhost:18000/orchestrator/route/stream",
                            params=params,
                            headers=headers,
                        )
                    )
                    save(turn_dir / "replay.json", replay)
                    turn["replay_passed"] = replay == events
                    turn["passed"] = (
                        verdict["passed"]
                        and turn["replay_passed"]
                        and not any(e["type"] == "error" for e in events)
                    )
                except Exception as exc:
                    turn["error"] = type(exc).__name__
                turn["seconds"] = round(time.monotonic() - started, 2)
                save(output / "report.json", report)
                print(
                    f"END turn {i+1}: {'PASS' if turn['passed'] else 'FAIL'}, {turn['seconds']} s",
                    flush=True,
                )
                if not chat_id:
                    break
            try:
                async with asyncio.timeout(120):
                    judge = await evaluate(http, config, episode, row["turns"], context)
            except Exception as exc:
                judge = {"verdict": "needs_review", "reason": type(exc).__name__}
            save(directory / "judge.json", judge)
            row["judge_verdict"] = judge["verdict"]
            row["passed"] = (
                len(row["turns"]) == len(episode["queries"])
                and all(t["passed"] for t in row["turns"])
                and judge["verdict"] == "pass"
            )
            save(output / "report.json", report)
            print(
                f"EPISODE {episode['id']} {episode['formulation']}: {'PASS' if row['passed'] else 'FAIL'}; judge={judge['verdict']}",
                flush=True,
            )
        response = await http.get("http://localhost:18090/audit")
        response.raise_for_status()
        save(output / "source-audit.json", response.json())
    report["final_fingerprint"] = fingerprint(config)
    report["passed"] = series_verdict(report)
    save(output / "report.json", report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, choices=range(1, 16), default=15)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))
