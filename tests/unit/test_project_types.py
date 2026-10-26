"""Urban API types the objects of the user's files are created as."""

from src.agents.services.compilance.compliance_entity_mapping import (
    EntityMapping,
    EntityTarget,
    MappingProposal,
)
from src.agents.services.project_files.project_types import (
    UrbanTypes,
    creation_spec,
    creation_specs,
    render_type_question,
)

PHYSICAL = [
    {
        "physical_object_type_id": 4,
        "name": "Жилой дом",
        "physical_object_function": {"id": 1, "name": "Здание"},
    },
    {
        "physical_object_type_id": 5,
        "name": "Нежилое здание",
        "physical_object_function": {"id": 1, "name": "Здание"},
    },
    {
        "physical_object_type_id": 7,
        "name": "Площадка",
        "physical_object_function": {"id": 3, "name": "Площадка"},
    },
    {"physical_object_type_id": 3, "name": "Зеленая зона"},
    {"physical_object_type_id": 52, "name": "Местная дорога"},
]
SERVICES = [
    {"service_type_id": 21, "name": "Школа"},
    {"service_type_id": 70, "name": "Детская площадка"},
]
ZONES = [
    {
        "functional_zone_type_id": 1,
        "name": "residential",
        "zone_nickname": "Жилая зона",
    },
    {"functional_zone_type_id": 2, "name": "recreation", "zone_nickname": "Рекреация"},
    {"functional_zone_type_id": 8, "name": "basic", "zone_nickname": "basic"},
    {
        "functional_zone_type_id": 13,
        "name": "residential_multistorey",
        "zone_nickname": "Многоэтажная жилая зона",
    },
]


def types() -> UrbanTypes:
    return UrbanTypes.from_dictionaries(PHYSICAL, SERVICES, ZONES)


def proposal(target, source="table", confidence=1.0, signature="s") -> MappingProposal:
    return MappingProposal(
        signature=signature,
        label=signature,
        count=3,
        layers=["layer"],
        roles={"project": 3},
        target=target,
        confidence=confidence,
        source=source,
    )


def test_dictionaries_mark_buildings_and_skip_unofficial_profiles():
    urban = types()

    assert urban.physical_type("жилой  дом").building is True
    assert urban.physical_type("Площадка").building is False
    assert [zone.name for zone in urban.profiles()] == [
        "residential",
        "recreation",
        "residential_multistorey",
    ]


def test_targets_are_physical_types_profiles_and_known_table_services():
    targets = types().targets()

    assert EntityTarget("Жилой дом", "physical_object") in targets
    assert EntityTarget("residential", "functional_zone") in targets
    assert EntityTarget("basic", "functional_zone") not in targets
    assert EntityTarget("Школа", "service") in targets
    # The table knows «Поликлиника», this Urban API does not.
    assert EntityTarget("Поликлиника", "service") not in targets


def test_physical_zone_and_service_specs():
    urban = types()

    house, _ = creation_spec(
        proposal(EntityTarget("Жилой дом", "physical_object")), urban
    )
    assert (house.kind, house.type_id, house.building) == ("object", 4, True)
    zone, _ = creation_spec(
        proposal(EntityTarget("residential", "functional_zone")), urban
    )
    assert (zone.kind, zone.type_id) == ("zone", 1)
    playground, _ = creation_spec(
        proposal(EntityTarget("Детская площадка", "service")), urban
    )
    assert (playground.type_name, playground.service_type_id) == ("Площадка", 70)
    school, _ = creation_spec(proposal(EntityTarget("Школа", "service")), urban)
    assert (school.type_name, school.building, school.service_name) == (
        "Нежилое здание",
        True,
        "Школа",
    )


def test_service_is_never_taken_from_the_llm():
    spec, problem = creation_spec(
        proposal(EntityTarget("Школа", "service"), source="llm", confidence=0.95),
        types(),
    )

    assert spec is None and problem is None


def test_unknown_types_are_reported():
    spec, problem = creation_spec(
        proposal(EntityTarget("Причал", "physical_object")), types()
    )
    assert spec is None
    assert "Причал" in problem

    spec, problem = creation_spec(
        proposal(EntityTarget("Аптека", "service"), source="user"), types()
    )
    assert spec.type_name == "Нежилое здание" and spec.service_type_id is None
    assert "Аптека" in problem


def test_specs_skip_pending_and_doubtful_groups():
    confident = proposal(EntityTarget("Жилой дом", "physical_object"), signature="a")
    doubtful = proposal(
        EntityTarget("Площадка", "physical_object"),
        source="llm",
        confidence=0.6,
        signature="b",
    )
    mapping = EntityMapping(proposals=[confident, doubtful], pending=[doubtful])

    specs, problems = creation_specs(mapping, types())

    assert set(specs) == {"a"} and problems == {}


def test_question_lists_pending_groups_with_confidence():
    item = proposal(
        EntityTarget("Площадка", "physical_object"),
        source="llm",
        confidence=0.65,
        signature="type=calm_rest",
    )

    text = render_type_question([item])

    assert "1. type=calm_rest (3 об., layer) → «Площадка» — уверенность 65%" in text
    assert "не загружать" in text
