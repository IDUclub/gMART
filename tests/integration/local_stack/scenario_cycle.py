"""Acceptance checks for service counts, layers and an unavailable provision norm."""

import json


async def verify_scenario_analysis(http, headers, output, events, final):
    if final.get("goal"):
        requirements = final["goal"]["requirements"]
        for subject in ("школа", "детский сад"):
            assert any(
                r["agent"] == "scenario_data"
                and r["subject"].casefold() == subject
                and r["status"] == "satisfied"
                for r in requirements
            ), "A requested data goal is unfulfilled"
        assert any(
            r["agent"] == "provision" and r["status"] == "blocked" for r in requirements
        ), "The unavailable normative must be attached to the calculation goal"
    assert any(
        s["agent"] == "provision" for s in final["steps"]
    ), "Provision calculation was never attempted"
    assert (
        final["status"] == "blocked"
    ), "Missing normative must not become completed analysis"
    assert final["missing"], "The blocker must explain what is missing"
    assert "норматив" in final["answer"].lower(), final["answer"]
    assert any(
        e["type"] == "step_event"
        and "missing_service_normative"
        in json.dumps(e["content"]["event"], ensure_ascii=False)
        for e in events
    ), "The missing normative must come from a service result"
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
    saved = next(
        p["payload"]["content"]
        for m in reversed(history["messages"])
        for p in m["parts"]
        if p["kind"] == "data" and p["payload"].get("event_type") == "analysis_context"
    )
    assert {a["id"] for a in saved["artifacts"]} == {
        a["id"] for a in final["artifacts"]
    }
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
        )
    counts = {}
    for name, filename in [("Школа", "schools"), ("Детский сад", "kindergartens")]:
        tables = [
            a["content"]
            for a in saved["artifacts"]
            if a["confirmed"]
            and a["kind"] == "table"
            and a["content"].get("title") == name
        ]
        layers = [
            a["content"]["feature_collection"]
            for a in saved["artifacts"]
            if a["confirmed"]
            and a["kind"] == "feature_collection"
            and a["content"].get("name") == name
        ]
        assert (
            len(tables) == len(layers) == 1
        ), "Each service type must be fetched once and retain both artifacts"
        table, layer = tables[0], layers[0]
        assert table["complete"] and table["total_rows"] == len(table["rows"])
        assert len(layer["features"]) == table["total_rows"]
        counts[name] = table["total_rows"]
        for suffix, payload in [("table.json", table), ("geojson", layer)]:
            (output / f"{filename}.{suffix}").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    (output / "analysis-chat.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if final.get("goal"):
        comparisons = [
            a["content"]
            for a in saved["artifacts"]
            if a["confirmed"]
            and a["kind"] == "table"
            and a["content"].get("name") == "goal_entity_counts"
        ]
        assert (
            len(comparisons) == 1
        ), "A verified count-comparison artifact must be returned"
        rows = comparisons[0]["rows"]
        assert {r["subject"].casefold(): r["count"] for r in rows} == {
            name.casefold(): count for name, count in counts.items()
        }
        assert all(
            r["difference_from_first"] == r["count"] - rows[0]["count"] for r in rows
        )
        (output / "comparison.table.json").write_text(
            json.dumps(comparisons[0], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output / "counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PASS counts, full layers/tables, persisted history and explicit missing-normative blocker: "
        + json.dumps(counts, ensure_ascii=False),
        flush=True,
    )
