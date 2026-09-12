"""Synthetic document ingestion and extraction against the local HTTP stack."""

import asyncio
import json

NAME = "LOCAL SDK TEST"
CORPUS = "local-sdk-synthetic"


async def verify_analysis(http, headers, output, events, final):
    assert final.get("status") == "completed", final
    assert "50" in final.get("answer", ""), final
    assert {s["agent"] for s in final["steps"] if s["status"] == "completed"} >= {
        "documents",
        "norms",
    }
    assert len(final["evidence_ids"]) >= 2, final
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
    assert all(a.get("content") for a in saved["artifacts"])
    assert set(final["evidence_ids"]) <= {
        a["id"] for a in saved["artifacts"] if a["confirmed"]
    }
    (output / "documents-chat.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PASS analytical DVD + NormGraph comparison, evidence references and persisted full artifacts",
        flush=True,
    )


async def document_cycle(http, headers, output):
    async def request(method, origin, path, **kwargs):
        response = await http.request(method, origin + path, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

    dvd = "http://localhost:18100"
    norm = "http://localhost:18020"
    listing = await request("GET", dvd, "/library/documents")
    documents = [d for d in listing["documents"] if d["name"] == NAME]
    if not documents:
        uploaded = await request(
            "POST",
            dvd,
            "/documents/direct",
            json=[
                {
                    "name": NAME,
                    "version": "2026",
                    "corpus": CORPUS,
                    "lang": "ru",
                    "title": "Синтетический документ для интеграционного теста",
                    "metadata": {"synthetic": True},
                    "fragments": [
                        {
                            "numbering": "1",
                            "text": "Синтетические тестовые требования. Этот документ не является действующим нормативным актом.",
                        },
                        {
                            "numbering": "1.1",
                            "text": "Расстояние от здания школы до открытой автомобильной стоянки должно быть не менее 50 метров.",
                        },
                    ],
                }
            ],
        )
        assert uploaded[0]["status"] == "queued", uploaded
        job_id = uploaded[0]["job_id"]
        print("Synthetic document queued", flush=True)
        async with asyncio.timeout(300):
            while True:
                job = await request("GET", dvd, f"/documents/{job_id}")
                assert job["status"] != "error", job
                if job["status"] == "done":
                    break
                await asyncio.sleep(2)
        listing = await request("GET", dvd, "/library/documents")
        documents = [d for d in listing["documents"] if d["name"] == NAME]
    assert len(documents) == 1 and documents[0]["corpus"] == CORPUS, documents
    doc_id = documents[0]["doc_id"]
    detail = await request("GET", dvd, f"/library/documents/{doc_id}")
    assert len(detail["fragments"]) == 2, detail
    hits = await request(
        "POST",
        dvd,
        "/search/texts",
        json={"query": "расстояние от школы до стоянки", "doc_id": doc_id, "limit": 5},
    )
    assert hits["hits"], hits
    print(
        "PASS DVD: queued ingestion, embeddings, library and vector search", flush=True
    )

    synced = await request("POST", norm, f"/sync/documents/{doc_id}")
    assert not synced["skipped"] and synced["restrictions"] >= 1, synced
    print("PASS NormGraph: real vLLM extraction and graph writes", flush=True)
    replay = await request("POST", norm, f"/sync/documents/{doc_id}")
    assert replay["extraction_skipped"] is True, replay
    assert replay["restrictions"] == synced["restrictions"], replay
    found = await request(
        "POST",
        norm,
        "/restrictions/search",
        json={"query": "расстояние от школы до стоянки", "doc_id": doc_id, "limit": 5},
    )
    assert found["hits"], found
    assert any(
        hit.get("value", {}).get("number") == 50
        for hit in found["hits"]
        if hit.get("value")
    ), found
    result = {
        "doc_id": doc_id,
        "dvd_hits": hits,
        "sync": synced,
        "repeat_sync": replay,
        "normgraph_hits": found,
    }
    (output / "document-cycle.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "PASS NormGraph: grounded numeric restriction and idempotent repeated sync",
        flush=True,
    )
