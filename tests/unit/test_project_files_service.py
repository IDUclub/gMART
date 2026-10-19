"""Dialogue of the project creation from files: questions, answers, retry."""

import json

import fakeredis.aioredis
import pytest

from src.agents.services.compilance.compliance_entity_mapping import EntityMapper
from src.agents.services.compilance.compliance_uploads import ComplianceUploadService
from src.agents.services.pipeline_state import PipelineStateStore
from src.agents.services.project_files.project_creation import ProjectCreator
from src.agents.services.project_files.project_files_service import (
    ProjectFilesService,
)
from src.agents.services.project_files.project_types import PROJECT_CLASSIFY_PROMPT
from src.common.service_auth import internal_user_context_jwt
from tests.unit.test_project_creation import BORDER, FakeUrban, square
from tests.unit.test_project_types import PHYSICAL, SERVICES, ZONES

ALICE = "e30.eyJzdWIiOiJhbGljZSJ9.sig"  # {"sub": "alice"}
CHAT = "chat-1"


def layer_bytes(*features: dict) -> bytes:
    return json.dumps(
        {"type": "FeatureCollection", "features": list(features)}
    ).encode()


def feature(geometry: dict, **properties) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


DESIGN = layer_bytes(
    feature(
        square(37.001, 55.001),
        id="h1",
        type="buildings",
        building="residential",
        levels="12",
    ),
    feature(square(37.003, 55.003), id="p1", type="playground"),
)
BOUNDARY = layer_bytes(feature(BORDER, type="border"))


class Urban(FakeUrban):
    def __init__(self, regions=None, around=None, **kwargs):
        super().__init__(**kwargs)
        self.regions = (
            [{"territory_id": 3, "name": "Москва"}] if regions is None else regions
        )
        self.around = around or []

    async def get_physical_object_types(self, token):
        return PHYSICAL

    async def get_service_types(self, token):
        return SERVICES

    async def get_functional_zone_types(self, token):
        return ZONES

    async def get_root_territories(self, token):
        return [{"territory_id": 12639, "name": "Россия"}]

    async def get_intersecting_territories(self, token, parent_id, geometry):
        return self.regions

    async def get_child_territories(self, token, parent_id):
        return [
            {"territory_id": 3, "name": "Москва"},
            {"territory_id": 4, "name": "Тверь"},
        ]

    async def get_physical_objects_around(self, token, geometry):
        return self.around


class Llm:
    """Classifies tag groups as told; extracts no intent."""

    def __init__(self, classified: dict | None = None):
        self.classified = classified or {}

    async def chat(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "сопоставляешь объекты" in system:
            groups = json.loads(kwargs["messages"][-1]["content"])["groups"]
            items = []
            for group in groups:
                entity, confidence = self.classified.get(
                    group["tags"].get("type"), (None, 0.0)
                )
                items.append(
                    {"id": group["id"], "entity": entity, "confidence": confidence}
                )
            return {"message": {"content": json.dumps({"items": items})}}
        return {"message": {"content": "{}"}}


@pytest.fixture
def env():
    store = PipelineStateStore(fakeredis.aioredis.FakeRedis(decode_responses=True))
    uploads = ComplianceUploadService(store)

    def build(urban=None, llm=None):
        service = ProjectFilesService(
            "http://ollama", None, urban or Urban(), store, uploads
        )
        service.llm_client = llm or Llm()
        service.entity_mapper = EntityMapper(
            service.llm_client, PROJECT_CLASSIFY_PROMPT
        )
        service.creator = ProjectCreator(service.urban)
        return service

    return store, uploads, build


async def upload(uploads, *files) -> str:
    return (await uploads.create("alice", list(files)))["upload_id"]


async def run(service, query, upload_ids=None, key=CHAT, inputs=None):
    return [
        event
        async for event in service.run_project_pipeline(
            token=ALICE,
            model="m",
            temperature=0,
            user_query=query,
            persist_history=False,
            conversation_key=key,
            upload_ids=upload_ids,
            inputs=inputs,
        )
    ]


def of_type(events, kind):
    return [event["content"] for event in events if event["type"] == kind]


async def test_without_files_asks_to_upload(env):
    _, _, build = env

    events = await run(build(), "Создай проект из файлов")

    assert "загрузите файлы проекта" in of_type(events, "chunk")[0]["text"]


async def test_chat_files_are_offered_and_used_after_yes(env):
    store, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    await store.add_chat_uploads(CHAT, [upload_id])
    urban = Urban()
    service = build(urban)

    first = await run(service, "Создай проект")
    question = of_type(first, "clarification")[0]
    assert question["kind"] == "reuse_files"
    assert "«design.geojson»" in question["question"]
    assert not urban.calls

    second = await run(service, "да")

    summary = of_type(second, "project_summary")[0]
    assert summary["project_id"] == 10 and summary["scenario_id"] == 12
    assert summary["profile"] == "Многоэтажная жилая зона"
    assert summary["objects"] == {
        "total": 2,
        "saved": 2,
        "by_type": {
            "Жилой дом": {"total": 1, "saved": 1, "services": 0},
            "Площадка": {"total": 1, "saved": 1, "services": 1},
        },
        "services": {"Детская площадка": 1},
    }
    assert "Проект «Проект из файлов: design» создан" in (
        of_type(second, "chunk")[0]["text"]
    )
    assert await store.get_project_dialogue(CHAT) is None


async def test_new_files_with_earlier_ones_ask_whether_to_add_them(env):
    store, uploads, build = env
    earlier = await upload(uploads, ("old.geojson", DESIGN))
    await store.add_chat_uploads(CHAT, [earlier])
    fresh = await upload(uploads, ("new.geojson", DESIGN), ("border.geojson", BOUNDARY))
    service = build()

    first = await run(service, "Создай проект", upload_ids=[fresh])
    assert "Добавить их в проект?" in of_type(first, "clarification")[0]["question"]

    second = await run(service, "нет")

    assert of_type(second, "project_summary")[0]["objects"]["total"] == 2
    assert await store.get_chat_uploads(CHAT) == [earlier, fresh]


async def test_doubtful_types_are_confirmed_by_the_user(env):
    _, uploads, build = env
    design = layer_bytes(
        feature(square(37.001, 55.001), id="r1", type="calm_rest"),
        feature(
            square(37.002, 55.002),
            id="h1",
            type="buildings",
            building="residential",
            levels="12",
        ),
    )
    upload_id = await upload(
        uploads, ("design.geojson", design), ("border.geojson", BOUNDARY)
    )
    urban = Urban()
    service = build(urban, Llm({"calm_rest": ("Площадка", 0.6)}))

    first = await run(service, "Создай проект", upload_ids=[upload_id])
    question = of_type(first, "clarification")[0]
    assert question["kind"] == "entity_mapping"
    assert question["options"][0]["value"] == "Площадка"
    assert not urban.calls

    second = await run(service, "1 — Зеленая зона")

    assert ("object", "r1", 3) in urban.calls
    record = await uploads.get(upload_id, "alice")
    assert record["project_confirmed_mapping"] == {
        "type=calm_rest": {"entity": "Зеленая зона", "entity_type": "physical_object"}
    }
    # A compliance check of the same files is not affected by this answer.
    assert record["confirmed_mapping"] == {}
    assert of_type(second, "project_summary")


async def test_region_and_existing_objects_are_asked(env):
    _, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    urban = Urban(
        regions=[
            {"territory_id": 3, "name": "Москва"},
            {"territory_id": 5, "name": "Московская область"},
        ],
        around=[
            {
                "physical_object_id": 9,
                "physical_object_type": {"name": "Жилой дом"},
                "geometry": square(37.005, 55.005),
            },
            {
                "physical_object_id": 10,
                "physical_object_type": {"name": "Жилой дом"},
                "geometry": square(38.0, 56.0),
            },
        ],
    )
    service = build(urban)

    first = await run(service, "Создай проект", upload_ids=[upload_id])
    assert of_type(first, "clarification")[0]["kind"] == "region"
    assert await service.answers_dialogue("m", CHAT, "2")
    assert not await service.answers_dialogue("m", CHAT, "какая погода?")

    second = await run(service, "2")
    question = of_type(second, "clarification")[0]
    assert question["kind"] == "existing_objects"
    assert "Жилой дом — 1." in question["question"]

    third = await run(service, "2")

    summary = of_type(third, "project_summary")[0]
    assert summary["region"] == {"territory_id": 5, "name": "Московская область"}
    assert summary["existing_policy"] == "delete_all"
    assert summary["deleted_existing"] == 2


async def test_failed_objects_are_retried_after_yes(env):
    store, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    urban = Urban(fail_keys={"h1"})
    service = build(urban)

    first = await run(service, "Создай проект", upload_ids=[upload_id])
    assert of_type(first, "project_summary")[0]["failed_total"] == 1
    assert "422" in of_type(first, "chunk")[0]["text"]
    assert of_type(first, "clarification")[0]["kind"] == "retry"

    urban.fail_keys.clear()
    urban.calls.clear()
    second = await run(service, "да")

    assert [call[0] for call in urban.calls] == ["object", "building"]
    assert of_type(second, "project_summary")[0]["failed_total"] == 0
    assert await store.get_project_dialogue(CHAT) is None


async def test_retry_declined_keeps_the_project(env):
    _, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    service = build(Urban(fail_keys={"h1"}))
    await run(service, "Создай проект", upload_ids=[upload_id])

    events = await run(service, "нет")

    assert "Проект 10 и сценарий 12 остаются" in of_type(events, "chunk")[0]["text"]


async def test_question_without_a_chat_is_not_kept(env):
    store, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    service = build(Urban(regions=[]))

    events = await run(service, "Создай проект", upload_ids=[upload_id], key=None)

    question = of_type(events, "clarification")[0]
    assert question["kind"] == "region"
    assert "Ответ не сохранится" in question["question"]


async def test_explicit_inputs_skip_the_questions(env):
    _, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    urban = Urban(regions=[], around=[{"physical_object_id": 1, "geometry": BORDER}])
    service = build(urban)

    events = await run(
        service,
        "Создай проект",
        upload_ids=[upload_id],
        inputs={
            "project_name": "Город-сад",
            "territory_id": 3,
            "functional_zone_type_id": 1,
            "existing_objects": "keep",
        },
    )

    summary = of_type(events, "project_summary")[0]
    assert summary["project_name"] == "Город-сад"
    assert summary["profile"] == "Жилая зона"
    assert not of_type(events, "clarification")


async def test_context_layers_are_not_created(env):
    _, uploads, build = env
    surroundings = layer_bytes(
        feature(square(37.008, 55.008), type="buildings", building="residential")
    )
    upload_id = await upload(
        uploads,
        ("design.geojson", DESIGN),
        ("border.geojson", BOUNDARY),
        ("osm_surroundings.geojson", surroundings),
    )
    service = build()

    events = await run(service, "Создай проект", upload_ids=[upload_id])

    assert of_type(events, "project_summary")[0]["objects"]["total"] == 2


async def test_project_is_created_with_the_users_own_client(env):
    store, uploads, _ = env
    service_client, user_client = Urban(), Urban()

    service = ProjectFilesService(
        "http://ollama", None, service_client, store, uploads, user_client
    )

    assert service.urban is user_client and service.creator.urban_api is user_client


async def test_synapse_call_without_a_user_token_is_refused(env):
    _, uploads, build = env
    upload_id = await upload(
        uploads, ("design.geojson", DESIGN), ("border.geojson", BOUNDARY)
    )
    urban = Urban()
    service = build(urban)

    events = [
        event
        async for event in service.run_project_pipeline(
            token=internal_user_context_jwt("alice"),
            model="m",
            temperature=0,
            user_query="Создай проект",
            persist_history=False,
            conversation_key=CHAT,
            upload_ids=[upload_id],
        )
    ]

    assert "через Synapse нельзя" in of_type(events, "chunk")[0]["text"]
    assert not urban.calls
