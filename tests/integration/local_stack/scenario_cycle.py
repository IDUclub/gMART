"""Acceptance checks for service counts, layers and an unavailable provision norm."""

import json


async def verify_scenario_analysis(http, headers, output, events, final):
    assert any(
        s["agent"] == "provision" for s in final["steps"]
    ), "Provision calculation was never attempted"
    assert (
        final["status"] == "blocked"
    ), "Missing normative must not become completed analysis"
    assert final["missing"], "The blocker must explain what is missing"
    assert "норматив" in final["answer"].lower(), final["answer"]
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
    (output / "counts.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PASS counts, full layers/tables, persisted history and explicit missing-normative blocker: "
        + json.dumps(counts, ensure_ascii=False),
        flush=True,
    )
