"""Parameters of a project created from files: names, region, profile, old objects.

Each is taken from the request first, then derived from the files; a question is
asked only when nothing settles it (the region of a boundary on two regions, the
profile of a design without housing).
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel
from shapely.geometry import shape

from src.agents.services.compilance.compliance_entity_mapping import (
    EntityTarget,
    feature_tags,
    normalized_properties,
    signature_of,
)
from src.agents.services.compilance.compliance_scope import normalized
from src.agents.services.compilance.compliance_uploads import UserFiles
from src.agents.services.normgraph.normgraph_reasoning import _request_json
from src.agents.services.project_files.project_types import (
    CreationSpec,
    UrbanType,
    UrbanTypes,
)

ExistingPolicy = Literal["replace_same", "delete_all", "keep"]
EXISTING_POLICIES: tuple[ExistingPolicy, ...] = ("replace_same", "delete_all", "keep")
EXISTING_TITLES: dict[str, str] = {
    "replace_same": "заменить объекты тех же типов, что в файлах",
    "delete_all": "удалить все существующие объекты внутри границы",
    "keep": "оставить существующие объекты, файлы добавить поверх",
}
_INDIVIDUAL = {"house", "detached", "semidetached_house", "bungalow"}
MAX_NAME = 200
_HOUSE = EntityTarget("Жилой дом", "physical_object")

_YES = re.compile(
    r"^(?:да|ок|ok|окей|хорошо|верно|подтверждаю|согласен|согласна|давай|"
    r"используй|использовать|повтори|повторить|продолжай|да, используй|да, повтори)$"
)
_NO = re.compile(r"^(?:нет|не надо|не нужно|отмена|отменить|не используй|стоп)$")


class ProjectIntent(BaseModel):
    project_name: str | None = None
    scenario_name: str | None = None
    profile: str | None = None
    existing_objects: ExistingPolicy | None = None


_INTENT_PROMPT = """Пользователь просит создать проект в Urban API из загруженных \
файлов. Извлеки из запроса только явно названные значения, иначе null:
- project_name — название проекта, если пользователь его назвал;
- scenario_name — название сценария, если названо;
- profile — профиль сценария: один из "profiles" (поле name), если пользователь его \
указал (например, «многоэтажная жилая застройка» → residential_multistorey);
- existing_objects — что сделать с существующими объектами внутри границы, если \
сказано: replace_same (заменить такие же), delete_all (снести всё), keep (оставить).
Не придумывай значений. Верни только валидный JSON без markdown:
{"project_name": null, "scenario_name": null, "profile": null, "existing_objects": null}
"""


async def extract_intent(
    llm_client,
    model: str,
    query: str,
    profiles: list[UrbanType],
    history: list[dict] | None = None,
) -> ProjectIntent:
    try:
        intent = await _request_json(
            llm_client,
            model,
            [
                {
                    "role": "system",
                    "content": _INTENT_PROMPT
                    + "\nprofiles: "
                    + "; ".join(f"{zone.name} — {zone.label}" for zone in profiles),
                },
                *(history or []),
                {"role": "user", "content": query},
            ],
            ProjectIntent,
            request_kind="project.inputs",
        )
    except Exception as exc:  # the defaults below still create the project
        logger.warning(f"Project intent extraction failed: {exc}")
        return ProjectIntent()
    if intent.profile and intent.profile not in {zone.name for zone in profiles}:
        intent.profile = None
    return intent


def yes_no(reply: str) -> bool | None:
    text = normalized(reply).strip(" .!")
    if _YES.match(text):
        return True
    if _NO.match(text):
        return False
    return None


def match_option(reply: str, options: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The option a reply names by its number, value or title."""
    text = normalized(reply).strip(' .!«»"')
    if text.isdigit() and 1 <= int(text) <= len(options):
        return options[int(text) - 1]
    for option in options:
        names = {normalized(str(option.get(key) or "")) for key in ("value", "label")}
        if text in names - {""}:
            return option
    matches = [
        option
        for option in options
        if text
        and len(text) >= 4
        and text in normalized(f"{option.get('label', '')} {option.get('value', '')}")
    ]
    return matches[0] if len(matches) == 1 else None


def project_name(files: UserFiles, requested: str | None) -> str:
    if requested and requested.strip():
        return requested.strip()[:MAX_NAME]
    names = [layer.name for layer in files.by_role("project")] or [
        layer.name for layer in files.layers
    ]
    return f"Проект из файлов: {', '.join(names)}"[:MAX_NAME]


def scenario_name(requested: str | None) -> str:
    if requested and requested.strip():
        return requested.strip()[:MAX_NAME]
    return "Проектное решение из файлов"


async def regions_for(urban_api, token: str, geometry: dict) -> list[dict[str, Any]]:
    """Regions (children of the top-level territories) the geometry lies in."""
    regions: list[dict[str, Any]] = []
    for root in await urban_api.get_root_territories(token):
        children = await urban_api.get_intersecting_territories(
            token, int(root["territory_id"]), geometry
        )
        for child in children or []:
            regions.append(
                {"territory_id": int(child["territory_id"]), "name": child.get("name")}
            )
    return list({item["territory_id"]: item for item in regions}.values())


async def all_regions(urban_api, token: str) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for root in await urban_api.get_root_territories(token):
        for child in await urban_api.get_child_territories(
            token, int(root["territory_id"])
        ):
            if child.get("name"):
                regions.append(
                    {
                        "territory_id": int(child["territory_id"]),
                        "name": child["name"],
                    }
                )
    return regions


def guess_profile(
    files: UserFiles, specs: dict[str, CreationSpec], types: UrbanTypes
) -> tuple[UrbanType | None, str]:
    """The scenario profile the design's buildings suggest, with the reason."""
    residential_type = types.physical_type("Жилой дом")
    floors: list[float] = []
    individual = 0
    residential = 0
    other: Counter[str] = Counter()
    for layer in files.by_role("project"):
        for feature in layer.features:
            properties = feature.get("properties") or {}
            tags = feature_tags(properties)
            spec = specs.get(signature_of(tags))
            if spec is None or spec.kind != "object" or not spec.building:
                continue
            if residential_type is not None and spec.type_id == residential_type.id:
                residential += 1
                individual += tags.get("building") in _INDIVIDUAL
                value = normalized_properties(properties, _HOUSE).get("floors")
                if value:
                    floors.append(float(value))
            else:
                other[tags.get("building") or ""] += 1
    if residential and residential >= sum(other.values()):
        if individual == residential:
            zone, reason = "residential_individual", "индивидуальные жилые дома"
        elif not floors:
            zone, reason = "residential", "жилые дома без этажности"
        else:
            median = statistics.median(floors)
            if median <= 4:
                zone = "residential_lowrise"
            elif median <= 8:
                zone = "residential_midrise"
            else:
                zone = "residential_multistorey"
            reason = f"жилые дома, медианная этажность {median:g}"
        return types.zone_type(zone) or types.zone_type("residential"), reason
    if other:
        industrial = other["industrial"] + other["warehouse"]
        if industrial * 2 > sum(other.values()):
            return types.zone_type("industrial"), "преобладают производственные здания"
        return types.zone_type("business"), "преобладают нежилые здания"
    return None, "в файлах нет зданий"


def boundary_centre(geometry: dict) -> dict:
    polygon = shape(geometry)
    point = polygon.centroid
    if not polygon.contains(point):
        point = polygon.representative_point()
    return {"type": "Point", "coordinates": [point.x, point.y]}
