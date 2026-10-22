"""Mapping of uploaded objects to norm entities: table, LLM, the user's answer."""

import json

import pytest

from src.agents.services.compilance.compliance_entity_mapping import (
    FGISTP_CLASSIFIER,
    EntityMapper,
    EntityMapping,
    EntityTarget,
    MappingReply,
    apply_reply,
    build_entity_layers,
    feature_tags,
    normalized_properties,
    render_mapping_question,
    signature_of,
    table_proposal,
    tag_groups,
)
from src.agents.services.compilance.compliance_uploads import (
    UploadedLayer,
    UserFiles,
    detect_role,
)
from src.agents.services.compilance.compliance_user_files import (
    user_files_report,
    user_files_summary,
    user_files_text,
)

HOUSE = EntityTarget("Жилой дом", "physical_object")
PLAYGROUND = EntityTarget("Детская площадка", "service")
SPORT = EntityTarget("Спортивная площадка", "service")


def feature(properties, ring=None, feature_id=None):
    ring = ring or [[37.48, 55.67], [37.49, 55.67], [37.49, 55.68], [37.48, 55.67]]
    item = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": properties,
    }
    if feature_id is not None:
        item["id"] = feature_id
    return item


def layer(name, role, features):
    return UploadedLayer(
        name=name,
        role=role,
        feature_collection={"type": "FeatureCollection", "features": features},
    )


def files(confirmed=None):
    return UserFiles(
        upload_ids=("u1",),
        layers=[
            layer(
                "map_objects",
                "project",
                [
                    feature(
                        {"type": "buildings", "building": "residential", "levels": "9"},
                        feature_id="58975350",
                    ),
                    feature({"type": "calm_rest", "amenity": "playground"}),
                    feature({"type": "public_space"}),
                ],
            ),
            layer(
                "osm_surroundings",
                "context",
                [
                    feature(
                        {"type": "buildings", "building": "school", "amenity": "school"}
                    )
                ],
            ),
            layer("border", "border", [feature({"type": "border"})]),
        ],
        confirmed_mapping=confirmed or {},
    )


@pytest.mark.parametrize(
    ("properties", "expected"),
    [
        ({"type": "buildings", "building": "residential"}, (HOUSE, 1.0)),
        ({"type": "buildings", "building": "apartments", "shop": "x"}, (HOUSE, 1.0)),
        (
            {"type": "buildings", "building": "school", "amenity": "school"},
            (EntityTarget("Школа", "service"), 1.0),
        ),
        ({"type": "playground", "amenity": "playground"}, (PLAYGROUND, 1.0)),
        ({"type": "sport", "amenity": "playground", "leisure": "pitch"}, (SPORT, 1.0)),
        (
            {"type": "highway", "highway": "tertiary"},
            (EntityTarget("Местная дорога", "physical_object"), 1.0),
        ),
        (
            {"type": "block", "blockType": "residential"},
            (EntityTarget("residential", "functional_zone"), 0.9),
        ),
        ({"type": "block", "blockType": "empty"}, (None, 1.0)),
        ({"type": "border"}, (None, 1.0)),
        ({"type": "entry_point"}, (None, 1.0)),
        ({"type": "calm_rest", "amenity": "playground"}, None),
        ({"type": "buildings", "building": "yes"}, None),
    ],
)
def test_table_rules(properties, expected):
    assert table_proposal(feature_tags(properties)) == expected


def test_signature_uses_only_classifying_tags():
    tags = feature_tags({"type": "Buildings", "building": "residential", "levels": 9})
    assert signature_of(tags) == "type=buildings|building=residential"


@pytest.mark.parametrize("key", ["CLASSID", "CLASS_ID", "classid", "ClAsS_iD"])
@pytest.mark.parametrize(
    "value", [701010101, 701010101.0, " 701010101 ", "701010101.0"]
)
def test_classifier_tags_normalize_dbf_identifiers(key, value):
    tags = feature_tags({key: value})
    assert tags["classid"] == "701010101"
    assert tags["fgistp_class"] == FGISTP_CLASSIFIER["701010101"]["name"]
    assert len(FGISTP_CLASSIFIER) == 463


@pytest.mark.parametrize(
    ("code", "zone", "confidence"),
    [
        ("701010100", "residential", 0.95),
        ("701010101", "residential", 0.95),
        ("701010200", "business", 0.9),
        ("701010301", "business", 0.9),
        ("701010303", "business", 0.7),
        ("701010401", "industrial", 0.9),
        ("701010402", "industrial", 0.9),
        ("701010403", "industrial", 0.9),
        ("701010404", "transport", 0.7),
        ("701010405", "transport", 0.7),
        ("701010601", "recreation", 0.9),
        ("701010701", "special", 0.85),
        ("701010800", "special", 0.85),
    ],
)
def test_classifier_zone_rules_precede_osm(code, zone, confidence):
    assert table_proposal(feature_tags({"CLASSID": code, "amenity": "school"})) == (
        EntityTarget(zone, "functional_zone"),
        confidence,
    )


@pytest.mark.parametrize("code", ["701010501", "701010900", "701011000"])
def test_classifier_zones_without_entity(code):
    assert table_proposal(feature_tags({"CLASSID": code})) == (None, 1.0)


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("602010101", "Детский сад"),
        ("602010102", "Школа"),
        ("602010401", "Больница"),
        ("602010402", "Поликлиника"),
        ("602010408", "Аптека"),
    ],
)
def test_classifier_unambiguous_services(code, name):
    assert table_proposal(feature_tags({"CLASSID": code})) == (
        EntityTarget(name, "service"),
        1.0,
    )


@pytest.mark.parametrize(
    "code",
    [
        "601010301",
        "603010801",
        "604010100",
        "606010100",
        "609010101",
        "702010100",
        "704010100",
        "706020100",
    ],
)
async def test_unsupported_classifier_groups_are_settled_and_reported(code):
    user_files = UserFiles(
        upload_ids=("u",),
        layers=[layer("classes", "context", [feature({"CLASSID": code})])],
    )
    llm = FakeLlm([])
    mapping = await EntityMapper(llm).propose("m", user_files, [])
    proposal = mapping.proposals[0]
    assert proposal.target is None and proposal.confidence == 1.0
    assert proposal.source == "table"
    assert proposal.reason.startswith("not_supported_entity:")
    if code.startswith("603"):
        assert proposal.reason == "not_supported_entity:zouit"
    assert not llm.requests
    restored = EntityMapping.from_dict(mapping.to_dict())
    assert restored.proposals[0].reason == proposal.reason
    summary = user_files_summary(
        files=user_files,
        mapping_result=mapping,
        layers=build_entity_layers(user_files, mapping.assignment()),
        not_checked=[],
        overlay=False,
    )
    assert summary["mapping"][0]["reason"] == proposal.reason
    report = "\n".join(user_files_report(summary))
    assert "распознано, нормами пока не используется" in report
    assert "Распознано, нормами пока не используется объектов: 1" in user_files_text(
        summary
    )


async def test_other_602_class_reaches_llm_with_name_and_no_zone_candidates():
    # The classifier knows this code, but the deterministic service table does not.
    code = "602010103"
    user_files = UserFiles(
        upload_ids=("u",),
        layers=[
            layer(
                "objects", "project", [feature({"CLASSID": code, "amenity": "school"})]
            )
        ],
    )
    llm = FakeLlm([{"items": [{"id": 1, "entity": "Школа", "confidence": 0.6}]}])
    await EntityMapper(llm).propose(
        "m",
        user_files,
        [
            EntityTarget("residential", "functional_zone"),
            EntityTarget("Школа", "service"),
        ],
    )
    payload = json.loads(llm.requests[0]["messages"][1]["content"])
    expected_name = FGISTP_CLASSIFIER[code]["name"]
    assert payload["groups"][0]["example"]["fgistp_class"] == expected_name
    assert all(
        item["entity_type"] != "functional_zone" for item in payload["candidates"]
    )
    assert (
        table_proposal(feature_tags({"CLASSID": "602999999", "amenity": "school"}))
        is None
    )


@pytest.mark.parametrize(
    "name",
    [
        "redline",
        "formedland",
        "constructionzoneborders",
        "elementplanningstructure",
        "publicterritoryborders",
        "indentline",
        "easement",
    ],
)
@pytest.mark.parametrize("suffix", ["", "_line", "_poly"])
async def test_gisogd_layer_names_are_settled_without_llm(name, suffix):
    user_files = UserFiles(
        upload_ids=("u",),
        layers=[
            layer(f"folder/{name}{suffix}", "project", [feature({"amenity": "school"})])
        ],
    )
    group = tag_groups(user_files)[0]
    assert group.tags["layer_kind"] == name
    mapping = await EntityMapper(FakeLlm([])).propose("m", user_files, [])
    assert mapping.proposals[0].reason == "not_supported_entity:gisogd"
    assert mapping.proposals[0].target is None


@pytest.mark.parametrize(
    "code",
    [
        "701010101",
        "702010100",
        "603010801",
        "604010100",
        "606010100",
        "609010101",
        "704010100",
        "706020100",
    ],
)
def test_classifier_context_roles(code):
    assert detect_role("classes", [feature({"CLASS_ID": float(code)})]) == "context"


def test_classifier_border_and_mixed_roles():
    assert detect_role("classes", [feature({"CLASSID": "601010301"})]) == "border"
    assert (
        detect_role("classes", [feature({"CLASSID": "601010301"}), feature({})])
        == "project"
    )
    assert (
        detect_role(
            "classes",
            [feature({"CLASSID": "701010101"}), feature({"CLASSID": "602010102"})],
        )
        == "project"
    )
    assert detect_role("classes", []) == "project"
    assert detect_role("boundary", [feature({"CLASSID": "701010101"})]) == "border"


@pytest.mark.parametrize("name", ["gr_dpt", "gr_dpt_line", "folder/GR_DPT_poly"])
def test_gisogd_boundary_role(name):
    assert detect_role(name, [feature({})]) == "border"


class FakeLlm:
    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    async def chat(self, **kwargs):
        self.requests.append(kwargs)
        return {
            "message": {"content": json.dumps(self.answers.pop(0), ensure_ascii=False)}
        }


async def test_doubtful_llm_proposal_for_a_needed_entity_waits_for_the_user():
    llm = FakeLlm(
        [
            {
                "items": [
                    {"id": 1, "entity": "Детская площадка", "confidence": 0.55},
                    {"id": 2, "entity": "Выдуманный тип", "confidence": 0.99},
                ]
            }
        ]
    )
    mapping = await EntityMapper(llm).propose("m", files(), [HOUSE, PLAYGROUND])
    by_signature = {item.signature: item for item in mapping.proposals}
    calm = by_signature["type=calm_rest|amenity=playground"]
    assert calm.target == PLAYGROUND and calm.source == "llm"
    # A name outside the candidates is no answer.
    assert by_signature["type=public_space"].target is None
    assert [item.signature for item in mapping.pending] == [calm.signature]
    assignment = mapping.assignment()
    assert assignment["type=buildings|building=residential"] == HOUSE
    assert calm.signature not in assignment
    payload = json.loads(llm.requests[0]["messages"][1]["content"])
    assert {"name": "Жилой дом", "entity_type": "physical_object"} in payload[
        "candidates"
    ]


async def test_doubt_about_an_entity_no_norm_needs_is_not_asked():
    llm = FakeLlm(
        [
            {
                "items": [
                    {"id": 1, "entity": "Детская площадка", "confidence": 0.5},
                    {"id": 2, "entity": None, "confidence": 0.9},
                ]
            }
        ]
    )
    mapping = await EntityMapper(llm).propose("m", files(), [HOUSE])
    assert mapping.pending == []
    assert "type=calm_rest|amenity=playground" not in mapping.assignment()


async def test_confirmed_mapping_replaces_table_and_llm():
    confirmed = {
        "type=calm_rest|amenity=playground": PLAYGROUND.to_dict(),
        "type=public_space": None,
    }
    llm = FakeLlm([])
    mapping = await EntityMapper(llm).propose("m", files(confirmed), [PLAYGROUND])
    assert llm.requests == []
    assert mapping.pending == []
    assert mapping.assignment()["type=calm_rest|amenity=playground"] == PLAYGROUND


@pytest.mark.parametrize(
    ("reply", "kind", "overrides"),
    [
        ("Да", "accept", {}),
        ("всё верно!", "accept", {}),
        ("нет", "reject", {}),
        ("2 — спортивная площадка; 1 нет", "edit", {"b": SPORT, "a": None}),
        ("1: Детская площадка", "edit", {"a": PLAYGROUND}),
    ],
)
async def test_reply_is_parsed_without_the_llm(reply, kind, overrides):
    pending = mapping_with_pending().pending
    llm = FakeLlm([])
    answer = await EntityMapper(llm).resolve_reply(
        "m", reply, pending, [HOUSE, PLAYGROUND, SPORT]
    )
    assert answer == MappingReply(kind=kind, overrides=overrides)
    assert llm.requests == []


async def test_unclear_reply_goes_to_the_llm():
    llm = FakeLlm([{"kind": "new_request", "overrides": []}])
    answer = await EntityMapper(llm).resolve_reply(
        "m", "а какие нормы есть для школ?", mapping_with_pending().pending, [HOUSE]
    )
    assert answer.kind == "new_request"


def mapping_with_pending():
    return EntityMapping.from_dict(
        {
            "proposals": [
                {
                    "signature": "a",
                    "label": "type=calm_rest",
                    "count": 2,
                    "entity": "Детская площадка",
                    "entity_type": "service",
                    "confidence": 0.6,
                    "source": "llm",
                },
                {
                    "signature": "b",
                    "label": "type=sport",
                    "count": 1,
                    "entity": "Детская площадка",
                    "entity_type": "service",
                    "confidence": 0.5,
                    "source": "llm",
                },
                {
                    "signature": "c",
                    "label": "type=buildings",
                    "count": 9,
                    "entity": "Жилой дом",
                    "entity_type": "physical_object",
                    "confidence": 1.0,
                    "source": "table",
                },
            ],
            "pending": ["a", "b"],
        }
    )


def test_apply_reply_confirms_every_pending_group():
    mapping = mapping_with_pending()
    assert apply_reply(mapping, MappingReply(kind="accept")) == {
        "a": PLAYGROUND.to_dict(),
        "b": PLAYGROUND.to_dict(),
    }
    assert apply_reply(mapping, MappingReply(kind="reject")) == {"a": None, "b": None}
    assert apply_reply(mapping, MappingReply(kind="edit", overrides={"b": SPORT})) == {
        "a": PLAYGROUND.to_dict(),
        "b": SPORT.to_dict(),
    }


def test_question_lists_groups_with_confidence():
    text = render_mapping_question(mapping_with_pending().pending)
    assert "1. type=calm_rest (2 об." in text
    assert "«Детская площадка» — уверенность 60%" in text


def test_generator_fields_become_urban_attributes():
    properties = normalized_properties(
        {
            "levels": "3",
            "levelHeights": "4.5,3.34,3.34",
            "livingSpace": "3496.8",
            "groundSpace": "955.0",
            "population": "69",
        },
        HOUSE,
    )
    assert properties["building"] == {
        "floors": 3,
        "height": 11.18,
        "living_area": 3496.8,
        "footprint_area": 955.0,
    }
    assert properties["floors"] == 3 and properties["population"] == 69.0


def test_floors_fall_back_to_the_number_of_level_heights():
    properties = normalized_properties({"levelHeights": "3,3,3,3"}, HOUSE)
    assert properties["building"]["floors"] == 4


def test_entity_layers_split_design_and_context():
    layers = build_entity_layers(
        files(),
        {
            "type=buildings|building=residential": HOUSE,
            "type=buildings|building=school|amenity=school": EntityTarget(
                "Школа", "service"
            ),
        },
    )
    (house,) = layers.project[HOUSE]
    assert house["id"] == "user/map_objects/58975350"
    assert house["properties"]["is_project"] is True
    assert house["properties"]["building"]["floors"] == 9
    assert house["properties"]["source_tags"] == "type=buildings, building=residential"
    assert house["properties"]["source_id"] == "58975350"
    assert list(layers.context) == [EntityTarget("Школа", "service")]
    assert layers.border is not None
    assert layers.counts()["Жилой дом"] == {"project": 1, "context": 0}


async def test_llm_never_maps_an_object_to_a_functional_zone():
    zone = EntityTarget("special", "functional_zone")
    llm = FakeLlm(
        [
            {
                "items": [
                    {"id": 1, "entity": "special", "confidence": 0.9},
                    {"id": 2, "entity": None, "confidence": 0.9},
                ]
            }
        ]
    )
    mapping = await EntityMapper(llm).propose("m", files(), [HOUSE, zone])
    payload = json.loads(llm.requests[0]["messages"][1]["content"])
    assert all(
        item["entity_type"] != "functional_zone" for item in payload["candidates"]
    )
    calm = next(
        item
        for item in mapping.proposals
        if item.signature == "type=calm_rest|amenity=playground"
    )
    assert calm.target is None and calm.source == "none"
    # Zones of the generator's blocks still come from the table.
    assert table_proposal({"type": "block", "blockType": "special"})[0] == EntityTarget(
        "special", "functional_zone"
    )


def context_files():
    return UserFiles(
        upload_ids=("u1",),
        layers=[
            layer(
                "map_objects",
                "project",
                [feature({"type": "buildings", "building": "residential"})],
            ),
            layer(
                "osm_surroundings",
                "context",
                [feature({"type": "buildings", "building": "yes"})],
            ),
        ],
    )


async def test_doubt_about_surroundings_is_asked_only_for_a_source_entity():
    answer = {"items": [{"id": 1, "entity": "Жилой дом", "confidence": 0.5}]}

    # Houses are only checked targets: the surroundings' houses serve no norm.
    as_target = await EntityMapper(FakeLlm([answer])).propose(
        "m", context_files(), [HOUSE], sources=[]
    )
    assert as_target.pending == []

    # A norm measuring from houses reads the surroundings: the doubt matters.
    as_source = await EntityMapper(FakeLlm([answer])).propose(
        "m", context_files(), [HOUSE], sources=[HOUSE]
    )
    assert [item.signature for item in as_source.pending] == [
        "type=buildings|building=yes"
    ]


async def test_cached_llm_answer_is_used_for_the_same_candidates_only():
    from src.agents.services.compilance.compliance_entity_mapping import (
        TABLE_TARGETS,
        classification_key,
    )

    first = await EntityMapper(
        FakeLlm([{"items": [{"id": 1, "entity": "Жилой дом", "confidence": 0.9}]}])
    ).propose("m", context_files(), [HOUSE])
    assert list(first.classified.values())[0]["entity"] == "Жилой дом"

    cached = context_files()
    cached.classified = first.classified
    again = await EntityMapper(FakeLlm([])).propose("m", cached, [HOUSE])
    yes = next(
        i for i in again.proposals if i.signature == "type=buildings|building=yes"
    )
    assert yes.target == HOUSE and yes.source == "llm" and again.classified == {}

    # Other norms bring other candidates: the cached answer does not apply.
    key = classification_key("type=buildings|building=yes", [HOUSE, *TABLE_TARGETS])
    assert key in first.classified
    other = await EntityMapper(
        FakeLlm([{"items": [{"id": 1, "entity": None, "confidence": 0.9}]}])
    ).propose("m", cached, [HOUSE, STATION])
    assert other.classified


STATION = EntityTarget("Железнодорожная станция", "physical_object")
