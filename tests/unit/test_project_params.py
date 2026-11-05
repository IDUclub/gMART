"""Names, region and profile of a project created from files."""

import pytest

from src.agents.services.compilance.compliance_uploads import UploadedLayer, UserFiles
from src.agents.services.project_files.project_params import (
    ProjectIntent,
    boundary_centre,
    boundary_decision,
    extract_intent,
    guess_profile,
    match_option,
    project_name,
    regions_for,
    scenario_name,
    yes_no,
)
from src.agents.services.project_files.project_types import CreationSpec
from tests.unit.test_project_types import types

HOUSE = CreationSpec("object", 4, "Жилой дом", building=True)
OFFICE = CreationSpec("object", 5, "Нежилое здание", building=True)
SQUARE = [
    [[37.0, 55.0], [37.001, 55.0], [37.001, 55.001], [37.0, 55.001], [37.0, 55.0]]
]


def files(*buildings: dict) -> UserFiles:
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": SQUARE},
            "properties": {"type": "buildings", **properties},
        }
        for properties in buildings
    ]
    return UserFiles(
        upload_ids=("u",),
        layers=[
            UploadedLayer(
                name="Проект",
                role="project",
                feature_collection={"type": "FeatureCollection", "features": features},
            )
        ],
    )


def specs_for(user_files: UserFiles) -> dict:
    return {
        "type=buildings|building=residential": HOUSE,
        "type=buildings|building=house": HOUSE,
        "type=buildings|building=commercial": OFFICE,
        "type=buildings|building=industrial": OFFICE,
    }


@pytest.mark.parametrize(
    ("levels", "zone"),
    [
        (["3", "4", "2"], "residential"),  # lowrise is absent from the dictionary
        (["9", "22", "16"], "residential_multistorey"),
    ],
)
def test_profile_from_the_median_floors(levels, zone):
    user_files = files(
        *({"building": "residential", "levels": item} for item in levels)
    )

    profile, reason = guess_profile(user_files, specs_for(user_files), types())

    assert profile.name == zone
    assert "медианная этажность" in reason


def test_profile_of_offices_and_of_no_buildings():
    offices = files({"building": "commercial"}, {"building": "commercial"})
    profile, reason = guess_profile(offices, specs_for(offices), types())
    assert profile is None  # «business» is not in this dictionary
    assert reason == "преобладают нежилые здания"

    empty = files()
    assert guess_profile(empty, {}, types()) == (None, "в файлах нет зданий")


def test_options_by_number_value_or_title():
    options = [
        {"value": 1, "label": "Москва"},
        {"value": 2, "label": "Московская область"},
    ]

    assert match_option("2", options)["value"] == 2
    assert match_option("москва", options)["value"] == 1
    assert match_option("Московская", options)["value"] == 2
    assert match_option("Мос", options) is None
    assert match_option("7", options) is None


def test_yes_no():
    assert yes_no("Да.") is True
    assert yes_no("повтори") is True
    assert yes_no("нет") is False
    assert yes_no("создай другой проект") is None


def test_default_names():
    user_files = files({"building": "residential"})

    assert project_name(user_files, None) == "Проект из файлов: Проект"
    assert project_name(user_files, "  Город-сад ") == "Город-сад"
    assert scenario_name(None) == "Проектное решение из файлов"


class Urban:
    async def get_root_territories(self, token):
        return [{"territory_id": 12639, "name": "Россия"}]

    async def get_intersecting_territories(self, token, parent_id, geometry):
        assert parent_id == 12639
        return [
            {"territory_id": 3, "name": "Москва"},
            {"territory_id": 3, "name": "Москва"},
        ]


async def test_regions_are_children_of_the_top_level_territories():
    regions = await regions_for(
        Urban(), "t", {"type": "Polygon", "coordinates": SQUARE}
    )

    assert regions == [{"territory_id": 3, "name": "Москва"}]


class Llm:
    def __init__(self, content: str):
        self.content = content

    async def chat(self, **kwargs):
        return {"message": {"content": self.content}}


async def test_intent_keeps_only_known_profiles():
    intent = await extract_intent(
        Llm('{"project_name": "Сад", "profile": "luxury", "existing_objects": "keep"}'),
        "m",
        "создай проект Сад",
        types().profiles(),
    )

    assert intent == ProjectIntent(project_name="Сад", existing_objects="keep")


async def test_intent_failure_falls_back_to_defaults():
    intent = await extract_intent(Llm("not json"), "m", "q", types().profiles())

    assert intent == ProjectIntent()


def test_centre_lies_inside_the_boundary():
    centre = boundary_centre({"type": "Polygon", "coordinates": SQUARE})

    assert centre["type"] == "Point"
    assert 37.0 < centre["coordinates"][0] < 37.001


def territory_layer(name="territory", role="project", count=1, **properties):
    return UploadedLayer(
        name=name,
        role=role,
        feature_collection={
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": SQUARE},
                    "properties": properties,
                }
                for _ in range(count)
            ],
        },
    )


def test_boundary_choice_precedes_border_and_uses_first_duplicate():
    first = territory_layer("same")
    second = territory_layer("same")
    border = territory_layer("border", role="border")
    user_files = UserFiles(("u",), [first, second, border])
    assert boundary_decision(user_files, "same").layer is first
    assert boundary_decision(user_files, "hull").hull
    assert boundary_decision(user_files, "missing").error
    assert boundary_decision(user_files).layer is None
    assert not boundary_decision(user_files).options
    assert first.role == "project"


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("tagged_objects", [False, True])
def test_ambiguous_boundaries_offer_hull_only_with_other_tagged_objects(
    count, tagged_objects
):
    layers = [territory_layer("one", count=count), territory_layer("two", count=count)]
    if tagged_objects:
        layers.append(territory_layer("houses", count=2, building="yes"))
    decision = boundary_decision(UserFiles(("u",), layers))
    assert [option["value"] for option in decision.options] == [
        "one",
        "two",
        *(["hull"] if tagged_objects else []),
        "none",
    ]
    assert "площадь ≈" in decision.options[0]["label"]


def test_single_untagged_boundary_precedes_multiple_polygons_and_tagged_objects():
    single = territory_layer()
    user_files = UserFiles(
        ("u",),
        [
            territory_layer("multiple", count=2),
            single,
            territory_layer("houses", count=2, building="yes"),
        ],
    )
    assert boundary_decision(user_files).layer is single
    assert all(layer.role == "project" for layer in user_files.layers)


def test_context_candidates_only_when_no_project_objects():
    context = territory_layer("osm_context", role="context")
    user_files = UserFiles(("u",), [context])
    assert [option["value"] for option in boundary_decision(user_files).options] == [
        "osm_context",
        "none",
    ]
    user_files.layers.append(territory_layer("houses", count=2, building="yes"))
    assert boundary_decision(user_files).hull
    assert boundary_decision(UserFiles(("u",), [])).error


def test_layer_name_tags_prevent_automatic_boundary():
    layer = territory_layer("formedland_poly")
    assert boundary_decision(UserFiles(("u",), [layer])).options


def test_boundary_area_failure_keeps_option(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("area unavailable")

    monkeypatch.setattr("src.agents.services.project_files.project_params.Geod", fail)
    decision = boundary_decision(UserFiles(("u",), [territory_layer(count=2)]))
    assert decision.options[0]["label"] == "«territory» — объектов: 2"
