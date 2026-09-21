import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.agents.a2a.pzz_agent import PzzA2AAgent
from src.agents.dto.pzz_request_dto import PzzInputs
from src.agents.services.pzz.pzz_a2a_service import PzzA2AService
from src.agents.services.pzz.pzz_columns import detect_columns
from src.agents.services.pzz.pzz_service import PzzService
from src.common.service_auth import internal_user_context_jwt

TOKEN = internal_user_context_jwt("pzz-user")


def collection(properties):
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": None, "properties": properties}],
    }


async def answer_stream():
    yield SimpleNamespace(message=SimpleNamespace(content="Проверено по отчёту."))


@pytest.fixture
def pzz_service(monkeypatch, state_store):
    llm = Mock()
    llm.chat = AsyncMock(side_effect=lambda **kwargs: answer_stream())
    monkeypatch.setattr(
        "src.agents.model_clients.base_client.build_llm_adapter", lambda *a, **k: llm
    )
    service = PzzService("http://localhost:11434", Mock(), Mock(), state_store)
    service.llm_client = llm
    service.POLL_INTERVAL = 0
    return service


class Mcp:
    api_client = None

    def __init__(self, submit=None, statuses=None):
        self.calls = []
        self.submit = submit or {"external_id": "pzz-task", "status": "queued"}
        self.statuses = iter(statuses or ["running", "finished"])

    async def call(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "get_task_result":
            return collection(
                {
                    "verdict": "allowed",
                    "cad_num": "65:01:123:4",
                    "ВРИ_ЕГРН": "ИЖС",
                    "Подобранный_ВРИ": "Жилая застройка",
                    "Код_подобранного_ВРИ": "2.1",
                    "Вердикт_ПЗЗ": "Разрешен",
                    "Причина": "ВРИ разрешён в зоне",
                    "Топ1_возможный_ВРИ": "2.1 — ИЖС",
                    "zone_code": "Ж-1",
                    "zone_name": "Жилая зона",
                    "created_at": "old",
                    "debug": {"prompt": "internal"},
                    "Топ5_возможных_ВРИ": "long candidate list",
                }
            )
        if name.startswith("submit_") or name == "classify_scenario":
            return self.submit
        if name in {"get_task_status", "get_scenario_classification_status"}:
            return {"status": next(self.statuses)}
        return {
            "summary": {"total": 1, "in_correct_zone": 1},
            "chat_message": "Один объект соответствует зоне.",
        }


async def run(service, mcp, **kwargs):
    return [
        event
        async for event in service.run_pzz_pipeline(
            mcp,
            TOKEN,
            "test-model",
            0,
            "Проверь ПЗЗ",
            persist_history=False,
            **kwargs,
        )
    ]


async def test_cadastral_auto_flow_and_reconnect_do_not_resubmit(pzz_service):
    mcp = Mcp()
    inputs = PzzInputs(
        mode="pzz_check",
        cadastral_geojson=collection({"vri": "ИЖС"}),
        pzz_zones_geojson=collection({"zone_code": "Ж-1", "zone_name": "Жилая зона"}),
    )
    events = await run(pzz_service, mcp, inputs=inputs)
    names = [name for name, _ in mcp.calls]
    assert names == [
        "submit_pzz_check_task",
        "get_task_status",
        "get_task_status",
        "get_task_report",
        "get_task_result",
    ]
    assert mcp.calls[0][1]["cadastral_vri_col"] == "vri"
    assert any(e["type"] == "object_zone_fit" for e in events)
    layer = next(
        e["content"]["feature_collection"]
        for e in events
        if e["type"] == "feature_collection"
    )
    properties = layer["features"][0]["properties"]
    assert properties == {
        "verdict": "allowed",
        "cad_num": "65:01:123:4",
        "ВРИ_ЕГРН": "ИЖС",
        "Подобранный_ВРИ": "Жилая застройка",
        "Код_подобранного_ВРИ": "2.1",
        "Вердикт_ПЗЗ": "Разрешен",
        "Причина": "ВРИ разрешён в зоне",
        "Топ1_возможный_ВРИ": "2.1 — ИЖС",
        "zone_code": "Ж-1",
        "zone_name": "Жилая зона",
    }
    assert events[-1]["content"]["done"] is True
    request_id = events[0]["content"]["request_id"]
    replay = await run(pzz_service, mcp, request_id=request_id)
    assert replay == events
    assert len(mcp.calls) == 5


async def test_classify_only_uses_classify_summary_not_zone_report(pzz_service):
    api = SimpleNamespace(
        classify_summary=AsyncMock(return_value={"summary": {"total": 1}})
    )
    mcp = Mcp(statuses=["finished"])
    events = await run(
        pzz_service,
        mcp,
        inputs=PzzInputs(
            mode="classify_only", cadastral_geojson=collection({"vri": "ИЖС"})
        ),
        pzz_api_client=api,
    )
    assert [name for name, _ in mcp.calls] == [
        "submit_classify_only_task",
        "get_task_status",
        "get_task_result",
    ]
    api.classify_summary.assert_awaited_once_with("pzz-task")
    assert any(e["type"] == "classify_summary" for e in events)
    layer = next(
        e["content"]["feature_collection"]
        for e in events
        if e["type"] == "feature_collection"
    )
    properties = layer["features"][0]["properties"]
    assert properties["Топ1_возможный_ВРИ"] == "2.1 — ИЖС"
    assert "debug" not in properties and "Топ5_возможных_ВРИ" not in properties


async def test_missing_columns_never_start_a_task(pzz_service):
    mcp = Mcp()
    events = await run(
        pzz_service,
        mcp,
        inputs=PzzInputs(
            mode="pzz_check",
            cadastral_geojson=collection({}),
            pzz_zones_geojson=collection({}),
        ),
    )
    assert events[-1]["type"] == "clarification"
    assert not mcp.calls


async def test_column_model_cannot_invent_columns():
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value={"message": {"content": '{"cadastral_vri_col":"fake"}'}}
        )
    )
    result = await detect_columns(
        llm, "model", collection({"description": "ИЖС"}), ["cadastral_vri_col"], {}
    )
    assert result == {"cadastral_vri_col": None}


async def test_unknown_column_uses_llm_but_numeric_companion_does_not_match_alias():
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value={"message": {"content": '{"pzz_zone_code_col":"actual"}'}}
        )
    )
    result = await detect_columns(
        llm,
        "model",
        collection({"Код_Индекс_зоны": 1, "actual": "Ж-1"}),
        ["pzz_zone_code_col"],
        {},
    )
    assert result["pzz_zone_code_col"] == "actual"
    assert llm.chat.await_count == 1


@pytest.mark.parametrize("action", ["confirm", "suggest_upload", "detection_failed"])
async def test_building_review_requires_input_without_polling(pzz_service, action):
    mcp = Mcp(
        submit={
            "action": action,
            "chat_message": "Нужно уточнение",
            "suggestions": [{"code": "Ж"}],
        }
    )
    events = await run(
        pzz_service,
        mcp,
        inputs=PzzInputs(
            mode="building_pzz_check",
            buildings_upload_id="buildings",
            pzz_zones_upload_id="zones",
        ),
    )
    assert events[-1]["type"] == "clarification"
    assert events[-1]["content"]["action"] == action
    assert len(mcp.calls) == 1
    assert "confirmed_zone_map" not in mcp.calls[0][1]


async def test_building_approved_mapping_and_nested_task(pzz_service):
    mcp = Mcp(
        submit={"action": "created", "task": {"external_id": "task-2"}},
        statuses=["finished"],
    )
    await run(
        pzz_service,
        mcp,
        inputs=PzzInputs(
            mode="building_pzz_check",
            buildings_upload_id="b",
            pzz_zones_upload_id="z",
            confirmed_zone_map={"Ж": "Ж-1"},
        ),
    )
    assert mcp.calls[0][1]["confirmed_zone_map"] == {"Ж": "Ж-1"}
    assert mcp.calls[1][1]["external_id"] == "task-2"


async def test_scenario_flow_uses_explicit_context(pzz_service):
    mcp = Mcp(statuses=["finished"])
    await run(
        pzz_service,
        mcp,
        scenario_id=42,
        inputs=PzzInputs(mode="scenario", year=2026, source="PZZ"),
    )
    assert mcp.calls[0][0] == "classify_scenario"
    assert mcp.calls[0][1]["scenario_id"] == 42
    assert mcp.calls[-1][0] == "get_scenario_classification_report"


async def test_failed_task_does_not_draft_answer(pzz_service):
    mcp = Mcp(statuses=["failed"])
    events = await run(
        pzz_service,
        mcp,
        scenario_id=42,
        inputs=PzzInputs(mode="scenario", year=2026, source="PZZ"),
    )
    assert events[-1]["type"] == "error"
    pzz_service.llm_client.chat.assert_not_awaited()


async def test_timeout_reconnect_continues_existing_task(pzz_service):
    pzz_service.MAX_WAIT_SECONDS = 0
    mcp = Mcp(statuses=["running", "finished"])
    events = await run(
        pzz_service,
        mcp,
        scenario_id=42,
        inputs=PzzInputs(mode="scenario", year=2026, source="PZZ"),
    )
    assert events[-1]["type"] == "error"
    events = await run(pzz_service, mcp, request_id=events[0]["content"]["request_id"])
    assert events[-1]["content"]["done"] is True
    assert [name for name, _ in mcp.calls].count("classify_scenario") == 1


async def test_other_user_cannot_replay(pzz_service):
    events = await run(pzz_service, Mcp(), inputs=PzzInputs(mode="pzz_check"))
    with pytest.raises(ValueError, match="caller"):
        _ = [
            e
            async for e in pzz_service.run_pzz_pipeline(
                Mcp(),
                internal_user_context_jwt("other"),
                "m",
                0,
                "Q",
                request_id=events[0]["content"]["request_id"],
            )
        ]


class FakePipeline:
    def __init__(self, events):
        self.events = events
        self.calls = []

    async def run_pzz_pipeline(self, **kwargs):
        self.calls.append(kwargs)
        for event in self.events:
            yield event


async def test_a2a_card_stream_artifacts_and_no_history():
    card = PzzA2AAgent().get_agent_card("https://gmart.example")
    assert card["url"] == "https://gmart.example/pzz/a2a"
    assert card["capabilities"]["streaming"] is True
    pipeline = FakePipeline(
        [
            {"type": "object_zone_fit", "content": {"summary": {"total": 1}}},
            {"type": "chunk", "content": {"text": "Ответ", "done": True}},
        ]
    )
    service = PzzA2AService(pipeline)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/stream",
        "params": {
            "message": {
                "parts": [
                    {"kind": "text", "text": "Проверь"},
                    {
                        "kind": "data",
                        "data": {
                            "inputs": {
                                "mode": "scenario",
                                "year": 2026,
                                "source": "PZZ",
                            },
                            "scenario_id": 42,
                        },
                    },
                ]
            }
        },
    }
    events = [e async for e in service.stream_json_rpc(payload, Mcp(), TOKEN)]
    assert events[0]["result"]["kind"] == "task"
    assert events[-1]["result"]["status"]["state"] == "completed"
    assert events[-1]["result"]["final"] is True
    assert pipeline.calls[0]["persist_history"] is False
    assert pipeline.calls[0]["inputs"].year == 2026
    assert any(
        e["result"].get("artifact", {}).get("parts", [{}])[0].get("kind") == "data"
        for e in events
    )


async def test_a2a_clarification_does_not_complete_task():
    service = PzzA2AService(
        FakePipeline(
            [{"type": "clarification", "content": {"question": "Укажите год"}}]
        )
    )
    response = await service.handle_json_rpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {"message": {"parts": [{"text": "Проверь"}]}},
        },
        Mcp(),
        TOKEN,
    )
    assert response["result"]["status"]["state"] == "input-required"


async def test_concurrent_reconnect_cannot_submit_twice(pzz_service):
    request_id = "same-request"
    async with pzz_service.state_store.execution_lock(request_id):
        with pytest.raises(ValueError, match="already running"):
            await run(
                pzz_service,
                Mcp(),
                request_id=request_id,
                inputs=PzzInputs(mode="pzz_check"),
            )


async def test_uncertain_submission_is_not_retried(pzz_service):
    mcp = Mcp()
    mcp.call = AsyncMock(side_effect=TimeoutError("connection lost after submission"))
    events = await run(
        pzz_service,
        mcp,
        scenario_id=42,
        inputs=PzzInputs(mode="scenario", year=2026, source="PZZ"),
    )
    assert events[-1]["type"] == "error"
    events = await run(pzz_service, mcp, request_id=events[0]["content"]["request_id"])
    assert events[-1]["type"] == "clarification"
    assert mcp.call.await_count == 1


async def test_clarification_is_saved_for_follow_up_context(pzz_service):
    pzz_service.create_chat = AsyncMock(return_value=("chat", "Проверка"))
    pzz_service.add_single_message = AsyncMock()
    events = [
        event
        async for event in pzz_service.run_pzz_pipeline(
            Mcp(), TOKEN, "model", 0, "Проверь", inputs=PzzInputs(mode="pzz_check")
        )
    ]
    assert events[-1]["type"] == "clarification"
    assert (
        pzz_service.add_single_message.await_args.args[3]
        == events[-1]["content"]["question"]
    )


async def test_missing_zones_do_not_silently_downgrade_to_classification(pzz_service):
    mcp = Mcp()
    events = await run(
        pzz_service, mcp, inputs=PzzInputs(cadastral_geojson=collection({"vri": "ИЖС"}))
    )
    assert events[-1]["type"] == "clarification"
    assert not mcp.calls


async def test_custom_classifier_uses_rest_submission_and_mcp_results(pzz_service):
    api = SimpleNamespace(
        submit_file_task=AsyncMock(return_value={"external_id": "custom-task"}),
        classify_summary=AsyncMock(return_value={"summary": {"total": 1}}),
    )
    mcp = Mcp(statuses=["finished"])
    events = await run(
        pzz_service,
        mcp,
        inputs=PzzInputs(
            mode="classify_only",
            cadastral_geojson=collection({"vri": "ИЖС"}),
            classifier_upload_id="custom",
        ),
        pzz_api_client=api,
    )
    assert api.submit_file_task.await_args.args[3] == "custom"
    assert [name for name, _ in mcp.calls] == ["get_task_status", "get_task_result"]
    assert events[-1]["content"]["done"] is True


async def test_demo_column_names_resolve_without_llm(pzz_service):
    result = await detect_columns(
        pzz_service.llm_client,
        "test-model",
        collection(
            {
                "Кадастровый_номер": "65:00:0000000:5110",
                "Вид_использования_по_документу": "Ведение садоводства",
                "Индекс_зоны": "Ж-1",
                "Наименование_зоны": "Зона индивидуальных жилых домов",
            }
        ),
        ["cadastral_vri_col", "pzz_zone_code_col", "pzz_zone_name_col"],
        {},
    )
    assert result == {
        "cadastral_vri_col": "Вид_использования_по_документу",
        "pzz_zone_code_col": "Индекс_зоны",
        "pzz_zone_name_col": "Наименование_зоны",
    }
    pzz_service.llm_client.chat.assert_not_awaited()


async def test_file_intent_with_scenario_asks_for_files_not_year(pzz_service):
    pzz_service.llm_client.chat = AsyncMock(
        return_value={
            "message": {
                "content": json.dumps(
                    {"mode": "pzz_check", "year": None, "source": None}
                )
            }
        }
    )
    mcp = Mcp()
    events = await run(pzz_service, mcp, scenario_id=843)
    question = next(
        e["content"]["question"] for e in events if e["type"] == "clarification"
    )
    assert "кадастровый слой" in question
    assert not mcp.calls


def test_answer_context_uses_disjoint_verdicts_without_mutating_report():
    report = {
        "summary": {
            "total": 208,
            "unclear": 14,
            "not_in_zone": 13,
            "by_verdict": {
                "Разрешен": 194,
                "Нет пересечения с ПЗЗ": 13,
                "Требуется ручная проверка": 1,
            },
        },
        "chat_message": "требуют ручной проверки: 14",
        "zones": ["x" * 60001],
    }
    evidence = json.loads(PzzService._answer_context(report, has_layer=True))
    assert evidence["summary"] == {
        "total": 208,
        "by_verdict": report["summary"]["by_verdict"],
    }
    assert not evidence.get("chat_message")
    assert evidence["detail_omitted"] is True
    assert evidence["result_layer"]["format"] == "GeoJSON FeatureCollection"
    assert report["summary"]["unclear"] == 14
    assert report["chat_message"] == "требуют ручной проверки: 14"
