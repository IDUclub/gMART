"""The benchmark exporter freezes the first model request of each stage."""

from __future__ import annotations

import json

from benchmarks.harness.cases import Case
from benchmarks.harness.export import export, export_case, input_sha
from benchmarks.harness.stage import Stage, load_stages


def _stage(fn, kind="auto") -> Stage:
    return Stage(id="test.stage", kind=kind, description="test", fn=fn)


def _case(stage_id="test.stage", case_input=None) -> Case:
    return Case(id="case", stage=stage_id, input=case_input or {"q": "школы"})


async def test_first_request_is_frozen_and_the_stage_stops_there():
    after_call = []

    async def fn(inp, ctx):
        await ctx.llm.chat(
            ctx.model,
            [{"role": "user", "content": inp["q"]}],
            think=False,
            format={"type": "object"},
            options={"temperature": 0, "num_predict": 100},
            request_kind="test.kind",
            reasoning_effort="medium",
        )
        after_call.append(True)

    record = await export_case(_stage(fn), _case(), context_window=None)

    assert record["status"] == "ok"
    assert after_call == []
    assert record["request"] == {
        "api": "chat",
        "request_kind": "test.kind",
        "messages": [{"role": "user", "content": "школы"}],
        "format": {"type": "object"},
        "options": {"temperature": 0, "num_predict": 100},
        "think": False,
        "stream": False,
        "kwargs": {"reasoning_effort": "medium"},
    }


async def test_a_broad_except_around_the_call_does_not_hide_the_capture():
    async def fn(inp, ctx):
        try:
            await ctx.llm.chat(ctx.model, [{"role": "user", "content": "a"}])
        except Exception:  # the services fall back like this
            return "fallback"

    record = await export_case(_stage(fn), _case(), context_window=None)

    assert record["status"] == "ok"


async def test_a_stage_decided_by_code_is_not_a_request():
    async def fn(inp, ctx):
        return "decided without the model"

    record = await export_case(_stage(fn), _case(), context_window=None)

    assert record["status"] == "code"
    assert "request" not in record


async def test_a_stage_error_is_reported():
    async def fn(inp, ctx):
        raise KeyError("user_query")

    record = await export_case(_stage(fn), _case(), context_window=None)

    assert record["status"] == "error"
    assert "KeyError" in record["error"]


async def test_generate_is_frozen_as_a_single_user_message():
    async def fn(inp, ctx):
        await ctx.llm.generate(ctx.model, "Придумай название", request_kind="title")

    record = await export_case(_stage(fn, "manual"), _case(), context_window=None)

    assert record["request"]["api"] == "generate"
    assert record["request"]["messages"] == [
        {"role": "user", "content": "Придумай название"}
    ]


def test_input_sha_ignores_key_order_and_tracks_the_stage():
    assert input_sha("a", {"x": 1, "y": [1, 2]}) == input_sha(
        "a", {"y": [1, 2], "x": 1}
    )
    assert input_sha("a", {"x": 1}) != input_sha("b", {"x": 1})
    assert input_sha("a", {"x": 1}) != input_sha("a", {"x": 2})


async def test_real_stages_reach_their_model_call():
    stages = load_stages()
    cases = [
        _case(
            "orchestrator.plan",
            {"user_query": "Какая обеспеченность школами?", "scenario_id": 1},
        ),
        _case(
            "provision.plan",
            {
                "user_query": "Какая обеспеченность школами?",
                "services_catalog": ["Школа"],
            },
        ),
        _case("chat_title", {"user_query": "Школы в зоне шума"}),
        _case("compliance.scope", {"user_query": "Проверь нормы по школам"}),
    ]
    for case in cases:
        record = await export_case(stages[case.stage], case, context_window=32768)
        assert record["status"] == "ok", (case.stage, record)
        text = json.dumps(record["request"]["messages"], ensure_ascii=False)
        assert "Школ" in text or "школ" in text, case.stage


def test_export_writes_requests_and_manifest(tmp_path):
    async def asks(inp, ctx):
        await ctx.llm.chat(ctx.model, [{"role": "user", "content": inp["q"]}])

    async def decides(inp, ctx):
        return None

    stages = {
        "s.asks": Stage(id="s.asks", kind="auto", description="d", fn=asks),
        "s.code": Stage(id="s.code", kind="auto", description="d", fn=decides),
    }
    cases = [_case("s.asks"), _case("s.code")]

    manifest = export(stages, cases, tmp_path, gmart_root=tmp_path, context_window=4096)

    rows = [
        json.loads(line)
        for line in (tmp_path / "requests.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["stage"], row["case"]) for row in rows] == [("s.asks", "case")]
    assert rows[0]["input_sha"] == input_sha("s.asks", {"q": "школы"})
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert manifest["requests"] == 1
    assert manifest["decided_by_code"] == ["s.code/case"]
    assert manifest["context_window"] == 4096
    assert set(manifest["stages"]) == {"s.asks", "s.code"}
