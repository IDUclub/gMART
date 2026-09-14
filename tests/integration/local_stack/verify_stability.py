"""Recheck saved live runs with fresh authentication; never rerun inference."""

import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
from dotenv import dotenv_values
from scenario_contract import verify_provision_scope
from series_contract import verify_series_manifest
from source_contract import verify_source_records
from stability import (
    headers_for,
    persisted,
    save,
    verify_analysis,
    verify_data,
    verify_scenario_analysis,
)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    config = dotenv_values(args.env_file)
    audit = {"fingerprint": original["fingerprint"], "cases": []}
    try:
        verify_series_manifest(original)
        audit["complete_immutable_series"] = True
    except AssertionError as exc:
        audit["complete_immutable_series"] = False
        audit["series_error"] = str(exc)
    async with httpx.AsyncClient(timeout=60, trust_env=False) as http:
        for row in original["cases"]:
            output = args.output / f"{row['index']:02d}-{row['kind']}"
            if not (output / "events.json").exists():
                audit["cases"].append(
                    {
                        "index": row["index"],
                        "kind": row["kind"],
                        "checks": {"terminal": False},
                        "passed": False,
                    }
                )
                continue
            events = json.loads((output / "events.json").read_text(encoding="utf-8"))
            result = {
                "index": row["index"],
                "kind": row["kind"],
                "request_id": row["request_id"],
                "checks": {},
            }
            audit["cases"].append(result)
            final = next(
                (
                    e["content"]
                    for e in reversed(events)
                    if e["type"] == "orchestrator_final"
                ),
                None,
            )
            if final is None:
                result["checks"]["terminal"] = False
                result["passed"] = False
                continue
            result["status"] = final["status"]
            result["elapsed_seconds"] = row["elapsed_seconds"]
            result["budget"] = final.get("budget")
            result["missing"] = [m["missing"] for m in final.get("missing", [])]
            result["pending_goals"] = [
                r["id"]
                for r in (final.get("goal") or {}).get("requirements", [])
                if r["status"] == "pending"
            ]
            result["checks"]["terminal"] = not any(e["type"] == "error" for e in events)
            replay_path = output / "replay.json"
            result["checks"]["replay"] = (
                replay_path.exists()
                and json.loads(replay_path.read_text(encoding="utf-8")) == events
            )
            headers = await headers_for(http, config)
            saved = None
            try:
                saved = await persisted(http, headers, output, events, final)
                result["checks"]["persistence"] = True
            except Exception as exc:
                result["checks"]["persistence"] = False
                result["persistence_error"] = type(exc).__name__
                # Keep auditing stored payloads if only the emitted artifact ID
                # is absent; that is a delivery defect, not lost database data.
                history_path = output / "history.json"
                if history_path.exists():
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    saved = next(
                        (
                            p["payload"]["content"]
                            for m in reversed(history["messages"])
                            for p in m["parts"]
                            if p["kind"] == "data"
                            and p["payload"].get("event_type") == "analysis_context"
                        ),
                        None,
                    )
            result["checks"]["stable_emitted_artifact_ids"] = all(
                "artifact_id" in e["content"]["event"]["content"]
                for e in events
                if e["type"] == "step_event"
                and e["content"]["event"]["type"] in {"table", "feature_collection"}
            )
            result["storage_artifacts_retained"] = bool(saved) and {
                a["id"] for a in saved["artifacts"]
            } == {a["id"] for a in final["artifacts"]}
            try:
                if row["kind"] == "analysis":
                    await verify_scenario_analysis(http, headers, output, events, final)
                elif row["kind"] == "data":
                    verify_data(saved, final)
                elif row["kind"] == "documents":
                    await verify_analysis(http, headers, output, events, final)
                else:
                    assert final["status"] == "blocked"
                    assert any(s["agent"] == "provision" for s in final["steps"])
                    assert not any(
                        s["agent"] == "scenario_data" for s in final["steps"]
                    )
                    assert final.get("goal") and final.get("missing")
                    assert any(
                        e["type"] == "step_event"
                        and "missing_service_normative"
                        in json.dumps(e["content"]["event"])
                        for e in events
                    ), "Continuation must recheck the calculation service"
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
                    )
                result["checks"]["functional_acceptance"] = True
            except Exception as exc:
                result["checks"]["functional_acceptance"] = False
                result["acceptance_error"] = type(exc).__name__
            if row["kind"] in {"analysis", "continuation"}:
                try:
                    verify_provision_scope(events)
                    result["checks"]["calculation_scope"] = True
                except AssertionError as exc:
                    result["checks"]["calculation_scope"] = False
                    result["calculation_scope_error"] = str(exc)
            # A useful partial answer is distinct from a clean controller finish.
            expected_blocker = "Применимый норматив обеспеченности"
            result["checks"]["controller_clean"] = (
                bool(final.get("goal"))
                and not result["pending_goals"]
                and (
                    not result["missing"]
                    if row["kind"] in {"data", "documents"}
                    else result["missing"] == [expected_blocker]
                )
            )
            if row["kind"] == "documents":
                # Internal artifact IDs and orphan [1] markers do not fulfill
                # this prompt's explicit request for links to both sources.
                links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", final.get("answer", ""))
                result["checks"]["document_source_links"] = len(set(links)) >= 2
                source_systems = set()
                source_records = {}
                for link in set(links):
                    # Never send the service credential to model-supplied origins.
                    url = httpx.URL(
                        link
                        if link.startswith("http")
                        else "http://localhost:18000" + link
                    )
                    if str(
                        url.copy_with(path="/", query=None)
                    ) != "http://localhost:18000/" or not url.path.startswith(
                        "/orchestrator/runs/"
                    ):
                        continue
                    response = await http.get(url, headers=headers)
                    if response.status_code != 200:
                        continue
                    artifact = response.json()
                    if artifact.get("kind") != "source_evidence" or not artifact.get(
                        "confirmed"
                    ):
                        continue
                    payload = artifact["content"]
                    source_systems.add(payload["system"])
                    source_records[payload["system"]] = payload["sources"]
                result["checks"]["resolvable_source_records"] = source_systems == {
                    "documents",
                    "norms",
                }
                save(output / "source-records.json", source_records)
                try:
                    verify_source_records(source_records)
                    result["checks"]["matching_fixture_sources"] = True
                except (AssertionError, TypeError, KeyError, AttributeError) as exc:
                    result["checks"]["matching_fixture_sources"] = False
                    result["source_error"] = str(exc)
            if saved and row["kind"] != "documents":
                try:
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
                        assert tables and layers
                        assert {r["service_id"] for r in tables[-1]["rows"]} == {
                            f["properties"]["service_id"]
                            for f in layers[-1]["features"]
                        }
                    result["checks"]["matching_service_ids"] = True
                except (AssertionError, KeyError):
                    result["checks"]["matching_service_ids"] = False
            result["passed"] = all(result["checks"].values())
            save(args.output / "verified-summary.json", audit)
    save(args.output / "verified-summary.json", audit)
    print(
        json.dumps(
            [
                {"index": r["index"], "kind": r["kind"], "checks": r["checks"]}
                for r in audit["cases"]
            ],
            ensure_ascii=False,
        )
    )
    return int(
        not audit["complete_immutable_series"]
        or len(audit["cases"]) != len(original["cases"])
        or not all(r.get("passed") for r in audit["cases"])
    )


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
