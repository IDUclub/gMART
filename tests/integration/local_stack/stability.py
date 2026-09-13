"""Repeatable live stability audit of the isolated stack, without automatic retries.

Run from the workspace with --env-file and --output. Raw traces stay in output;
summary.json records separate acceptance, persistence and replay verdicts.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx
from document_cycle import verify_analysis
from dotenv import dotenv_values
from scenario_cycle import verify_scenario_analysis

ORIGIN = "http://localhost:18000"
ANALYSIS = [
    "В сценарии 772 выполни три отдельные подзадачи: 1. Получи услуги типа «Школа»: количество, таблицу и слой. 2. Получи услуги типа «Детский сад»: количество, таблицу и слой. 3. Рассчитай обеспеченность школами. Затем сопоставь количества школ и детских садов. Если нормативы не заданы, сохрани результаты подсчёта, таблицы и слои и прямо объясни, каких данных не хватает для расчёта обеспеченности.",
    "Сравни в сценарии 772 услуги типов «Школа» и «Детский сад»: нужны полные таблицы, слои и разница количества. Также рассчитай обеспеченность школами. Сначала проверь доступные данные и нормативы инструментами. При невозможности расчёта верни остальные результаты и конкретно объясни, что необходимо добавить.",
    "Для сценария 772 подготовь аналитический ответ: сколько услуг типа «Детский сад» и сколько типа «Школа», как различаются их количества и какова обеспеченность школами? Приложи полные таблицы и географические слои обеих выборок. Если расчёт недоступен, проверь причину через сервис и сохрани все полученные результаты.",
]
DATA = "В сценарии 772 посчитай услуги типов «Школа» и «Детский сад», сравни их количество в таблице и верни полные таблицы и слои этих услуг. Не оценивай обеспеченность."
DOCUMENTS = "Сравни исходный текст пункта 1.1 документа LOCAL SDK TEST из DVD с ограничением, извлечённым в NormGraph из этого документа. Проверь совпадение числового требования и объектов, объясни результат и приложи ссылки на исходный пункт и запись ограничения. Используй оба источника. Это синтетические тестовые данные, не действующий норматив."


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_events(response):
    response.raise_for_status()
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def headers_for(http, config):
    response = await http.post(
        f"{config['SERVICE_AUTH_SERVER_URL']}/realms/{config['SERVICE_AUTH_REALM']}/protocol/openid-connect/token",
        data={
            "grant_type": "client_credentials",
            "client_id": config["SERVICE_AUTH_CLIENT_ID"],
            "client_secret": config["SERVICE_AUTH_CLIENT_SECRET"],
        },
    )
    response.raise_for_status()
    token = response.json()["access_token"]
    claims = json.loads(base64.urlsafe_b64decode(token.split(".")[1] + "==="))
    return {"Authorization": f"Bearer {token}", "X-User-Id": claims["sub"]}


async def persisted(http, headers, output, events, final):
    chat_id = next(
        e["content"]["event"]["chat_id"]
        for e in events
        if e["type"] == "service_event"
        and e["content"].get("event", {}).get("storage_event_type") == "chat_created"
    )
    response = await http.get(
        f"http://localhost:18010/api/v1/chat_history/{chat_id}", headers=headers
    )
    response.raise_for_status()
    history = response.json()
    save(output / "history.json", history)
    saved = next(
        p["payload"]["content"]
        for m in reversed(history["messages"])
        for p in m["parts"]
        if p["kind"] == "data" and p["payload"].get("event_type") == "analysis_context"
    )
    assert {a["id"] for a in saved["artifacts"]} == {
        a["id"] for a in final["artifacts"]
    }, "Final/history artifact IDs differ"
    for event in events:
        if event["type"] != "step_event":
            continue
        item = event["content"]["event"]
        if item["type"] not in {"table", "feature_collection"}:
            continue
        content = dict(item["content"])
        aid = content.pop("artifact_id")
        assert (
            next(a["content"] for a in saved["artifacts"] if a["id"] == aid) == content
        ), "Artifact payload changed in storage"
    return saved


def verify_data(saved, final):
    assert final["status"] == "completed", "Data-only analysis must complete"
    assert final.get("goal"), "Goal missing"
    assert all(
        r["status"] == "satisfied" for r in final["goal"]["requirements"]
    ), "Unsatisfied data requirement"
    counts = {}
    for subject in ("Школа", "Детский сад"):
        tables = [
            a["content"]
            for a in saved["artifacts"]
            if a["confirmed"]
            and a["kind"] == "table"
            and a["content"].get("title") == subject
        ]
        layers = [
            a["content"]["feature_collection"]
            for a in saved["artifacts"]
            if a["confirmed"]
            and a["kind"] == "feature_collection"
            and a["content"].get("name") == subject
        ]
        assert len(tables) == len(layers) == 1, "Missing or duplicate typed artifacts"
        table = tables[0]
        assert table["complete"] and table["total_rows"] == len(table["rows"]) == len(
            layers[0]["features"]
        ), "Incomplete selection"
        counts[subject] = table["total_rows"]
    comparisons = [
        a["content"]
        for a in saved["artifacts"]
        if a["kind"] == "table" and a["content"].get("name") == "goal_entity_counts"
    ]
    assert len(comparisons) == 1, "Comparison artifact missing"
    rows = comparisons[0]["rows"]
    assert {r["subject"].casefold(): r["count"] for r in rows} == {
        s.casefold(): c for s, c in counts.items()
    }
    assert all(
        r["difference_from_first"] == r["count"] - rows[0]["count"] for r in rows
    )


def fingerprint():
    root = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=15
    ).strip()
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    image = subprocess.check_output(
        ["docker", "inspect", "gmart-sdk-local-agents-1", "--format", "{{.Image}}"],
        text=True,
        timeout=15,
    ).strip()
    return {
        "commit": commit,
        "source_sha256": digest.hexdigest(),
        "agents_image": image,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=20, choices=range(1, 21))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = dotenv_values(args.env_file)
    initial = fingerprint()
    report = {
        "fingerprint": initial,
        "model": config["LLM_MODEL"],
        "base_url": config["LLM_BASE_URL"],
        "temperature": 0,
        "cases": [],
    }
    # Sequential requests avoid contaminating reliability with load testing.
    cases = (
        [("analysis", ANALYSIS[i % 3]) for i in range(10)]
        + [("data", DATA)] * 4
        + [("documents", DOCUMENTS)] * 4
        + [
            (
                "continuation",
                "Продолжи сохранённый анализ: повторно проверь расчёт обеспеченности школами.",
            )
        ]
        * 2
    )
    anchor = None
    async with httpx.AsyncClient(timeout=660, trust_env=False) as http:
        for index, (kind, query) in enumerate(cases[: args.runs], 1):
            assert fingerprint() == initial, "Build changed during the audit"
            output = args.output / f"{index:02d}-{kind}"
            output.mkdir(exist_ok=True)
            request_id = str(uuid4())
            row = {
                "index": index,
                "kind": kind,
                "request_id": request_id,
                "checks": {},
                "query": query,
            }
            report["cases"].append(row)
            save(args.output / "summary.json", report)
            print(f"START {index}/{args.runs} {kind} {request_id}", flush=True)
            started = time.monotonic()
            params = {
                "request": query,
                "model": config["LLM_MODEL"],
                "temperature": 0,
                "request_id": request_id,
            }
            if kind != "documents":
                params["scenario_id"] = 772
            if kind == "continuation":
                if not anchor:
                    row["checks"][
                        "prerequisite"
                    ] = "FAIL: no accepted analysis to continue"
                    save(args.output / "summary.json", report)
                    continue
                params["continue_from"] = anchor
            try:
                headers = await headers_for(http, config)
                events = parse_events(
                    await http.get(
                        ORIGIN + "/orchestrator/route/stream",
                        params=params,
                        headers=headers,
                    )
                )
                save(output / "events.json", events)
                row["elapsed_seconds"] = round(time.monotonic() - started, 2)
                final = next(
                    e["content"]
                    for e in reversed(events)
                    if e["type"] == "orchestrator_final"
                )
                row.update(
                    {
                        key: final.get(key)
                        for key in ("status", "budget", "continue_from")
                    }
                )
                row["checks"]["terminal"] = (
                    "PASS"
                    if not any(e["type"] == "error" for e in events)
                    else "FAIL: SSE error"
                )
                # Long analyses can outlive the client's JWT. Verification uses
                # a fresh token; the analysis itself is never retried.
                headers = await headers_for(http, config)
                saved = None
                try:
                    saved = await persisted(http, headers, output, events, final)
                    row["checks"]["persistence"] = "PASS"
                except Exception as exc:
                    row["checks"][
                        "persistence"
                    ] = f"FAIL: {type(exc).__name__}: {str(exc)[:250]}"
                try:
                    if kind == "analysis":
                        await verify_scenario_analysis(
                            http, headers, output, events, final
                        )
                        anchor = anchor or final["continue_from"]
                    elif kind == "data":
                        verify_data(saved, final)
                    elif kind == "documents":
                        await verify_analysis(http, headers, output, events, final)
                    else:
                        assert (
                            final["status"] == "blocked"
                        ), "Unavailable norm must remain blocked"
                        assert not any(
                            s["agent"] == "scenario_data" for s in final["steps"]
                        ), "Continuation fetched confirmed data again"
                        assert any(
                            s["agent"] == "provision" for s in final["steps"]
                        ), "Continuation did not retry calculation"
                        assert final.get("goal") and final.get(
                            "missing"
                        ), "Lost goal/blocker"
                        assert (
                            saved
                            and len(
                                [
                                    a
                                    for a in saved["artifacts"]
                                    if a["kind"] == "feature_collection"
                                ]
                            )
                            == 2
                        ), "Lost saved layers"
                    row["checks"]["acceptance"] = "PASS"
                except Exception as exc:
                    # Full data stays in the trace, never dump a potentially sensitive final.
                    row["checks"][
                        "acceptance"
                    ] = f"FAIL: {type(exc).__name__}: {str(exc)[:250]}"
                try:
                    replay = parse_events(
                        await http.get(
                            ORIGIN + "/orchestrator/route/stream",
                            params=params,
                            headers=headers,
                        )
                    )
                    save(output / "replay.json", replay)
                    assert replay == events, "Terminal replay differs"
                    row["checks"]["replay"] = "PASS"
                except Exception as exc:
                    row["checks"][
                        "replay"
                    ] = f"FAIL: {type(exc).__name__}: {str(exc)[:250]}"
            except Exception as exc:
                row["elapsed_seconds"] = round(time.monotonic() - started, 2)
                row["checks"]["request"] = f"FAIL: {type(exc).__name__}"
            row["passed"] = bool(row["checks"]) and all(
                v == "PASS" for v in row["checks"].values()
            )
            save(args.output / "summary.json", report)
            print(
                json.dumps(
                    {
                        k: row.get(k)
                        for k in (
                            "index",
                            "kind",
                            "status",
                            "elapsed_seconds",
                            "checks",
                            "passed",
                        )
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    report["passed"] = sum(bool(r.get("passed")) for r in report["cases"])
    report["total"] = len(report["cases"])
    report["final_fingerprint"] = fingerprint()
    save(args.output / "summary.json", report)
    print(f"FINISHED {report['passed']}/{report['total']} accepted", flush=True)
    return int(report["passed"] != report["total"])


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
