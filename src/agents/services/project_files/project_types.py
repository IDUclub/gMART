"""Which Urban API type each object of the user's files is created as.

Objects are grouped by their classifying tags and mapped like in a compliance check
(``compliance_entity_mapping``): the table first, the LLM for the rest, and the user
confirms what the LLM is unsure of. Here every Urban API physical object type is a
candidate and the LLM may choose only among them: a service is created only where
the table names it (``amenity=school`` → «Школа») or the user does — never guessed.

A functional zone comes only from the generator's blocks (``blockType``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.agents.services.compilance.compliance_entity_mapping import (
    TABLE_TARGETS,
    EntityMapping,
    EntityTarget,
    MappingProposal,
)
from src.agents.services.compilance.compliance_scope import normalized

# Profiles the frontend does not offer for a scenario: «без профиля», «unknown» and
# the multifunctional zone.
EXCLUDED_PROFILE_IDS = frozenset({8, 14, 15})
BUILDING_FUNCTION = "здание"
# The physical object a table service stands on; any other service — a building.
_SERVICE_HOSTS = {
    "Детская площадка": "Площадка",
    "Спортивная площадка": "Площадка",
    "Площадка для выгула собак": "Площадка",
    "Парк": "Зеленая зона",
}
_DEFAULT_HOST = "Нежилое здание"

PROJECT_CLASSIFY_PROMPT = """Ты сопоставляешь объекты из файла проекта застройки с \
типами физических объектов Urban API, чтобы создать их в сценарии проекта.

Каждая группа "groups" — объекты с одинаковыми тегами (OSM-подобные теги \
генератора застройки: type, building, amenity, leisure, highway …), с числом \
объектов, именами слоёв и примером атрибутов. Для каждой группы выбери ОДИН тип из \
списка "candidates" (поле name) или null, если ни один тип не подходит. Не придумывай \
типов вне списка. confidence — насколько ты уверен (0..1): 0.9 и выше — только если \
теги однозначно означают этот тип; если тип лишь похож — не выше 0.6. reason — одна \
короткая фраза.

Верни только валидный JSON без markdown:
{"items": [{"id": 1, "entity": "Площадка", "confidence": 0.7, "reason": "..."}]}
"""


@dataclass(frozen=True)
class UrbanType:
    id: int
    name: str
    label: str = ""
    building: bool = False


@dataclass(frozen=True)
class UrbanTypes:
    """Urban API dictionaries: physical object, service and functional zone types."""

    physical: dict[str, UrbanType]
    services: dict[str, UrbanType]
    zones: dict[str, UrbanType]

    @classmethod
    def from_dictionaries(
        cls,
        physical: list[dict[str, Any]],
        services: list[dict[str, Any]],
        zones: list[dict[str, Any]],
    ) -> "UrbanTypes":
        def function_name(item: dict[str, Any]) -> str:
            function = item.get("physical_object_function") or {}
            return normalized(function.get("name") or "")

        return cls(
            physical={
                normalized(item["name"]): UrbanType(
                    id=int(item["physical_object_type_id"]),
                    name=item["name"],
                    building=function_name(item) == BUILDING_FUNCTION,
                )
                for item in physical or []
                if item.get("name") and item.get("physical_object_type_id") is not None
            },
            services={
                normalized(item["name"]): UrbanType(
                    id=int(item["service_type_id"]), name=item["name"]
                )
                for item in services or []
                if item.get("name") and item.get("service_type_id") is not None
            },
            zones={
                item["name"]: UrbanType(
                    id=int(item["functional_zone_type_id"]),
                    name=item["name"],
                    label=item.get("zone_nickname")
                    or item.get("description")
                    or item["name"],
                )
                for item in zones or []
                if item.get("name") and item.get("functional_zone_type_id") is not None
            },
        )

    def physical_type(self, name: str | None) -> UrbanType | None:
        return self.physical.get(normalized(name or ""))

    def service_type(self, name: str | None) -> UrbanType | None:
        return self.services.get(normalized(name or ""))

    def zone_type(self, name: str | None) -> UrbanType | None:
        return self.zones.get(name or "")

    def profiles(self) -> list[UrbanType]:
        """Functional zone types a scenario may take as its profile."""
        return sorted(
            (
                zone
                for zone in self.zones.values()
                if zone.id not in EXCLUDED_PROFILE_IDS
            ),
            key=lambda zone: zone.id,
        )

    def targets(self) -> list[EntityTarget]:
        """The mapping's candidates: physical object types, zones, table services."""
        targets = [
            EntityTarget(item.name, "physical_object")
            for item in self.physical.values()
        ]
        targets += [
            EntityTarget(zone.name, "functional_zone")
            for zone in self.zones.values()
            if zone.id not in EXCLUDED_PROFILE_IDS
        ]
        targets += [
            target
            for target in TABLE_TARGETS
            if target.entity_type == "service" and self.service_type(target.name)
        ]
        return list(dict.fromkeys(targets))


@dataclass(frozen=True)
class CreationSpec:
    """How the objects of one tag group are created in the scenario."""

    kind: Literal["zone", "object"]
    type_id: int
    type_name: str
    building: bool = False
    service_type_id: int | None = None
    service_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "building": self.building,
            "service_type_id": self.service_type_id,
            "service_name": self.service_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CreationSpec":
        return cls(
            kind=data["kind"],
            type_id=int(data["type_id"]),
            type_name=data["type_name"],
            building=bool(data.get("building")),
            service_type_id=data.get("service_type_id"),
            service_name=data.get("service_name"),
        )


def creation_spec(
    proposal: MappingProposal, types: UrbanTypes
) -> tuple[CreationSpec | None, str | None]:
    """The spec of a mapped group, or why it cannot be created."""
    target = proposal.target
    if target is None:
        return None, None
    if target.entity_type == "functional_zone":
        zone = types.zone_type(target.name)
        if zone is None:
            return None, f"нет типа функциональной зоны «{target.name}» в Urban API"
        return CreationSpec("zone", zone.id, zone.name), None
    if target.entity_type == "physical_object":
        physical = types.physical_type(target.name)
        if physical is None:
            return None, f"нет типа физического объекта «{target.name}» в Urban API"
        return (
            CreationSpec("object", physical.id, physical.name, physical.building),
            None,
        )
    # A service only from the table or the user: the LLM never names one here.
    if proposal.source not in {"table", "user"}:
        return None, None
    service = types.service_type(target.name)
    host = types.physical_type(_SERVICE_HOSTS.get(target.name, _DEFAULT_HOST))
    if host is None:
        return None, f"нет типа физического объекта для сервиса «{target.name}»"
    return (
        CreationSpec(
            "object",
            host.id,
            host.name,
            host.building,
            service_type_id=service.id if service else None,
            service_name=service.name if service else None,
        ),
        None if service else f"нет типа сервиса «{target.name}» в Urban API",
    )


def creation_specs(
    mapping: EntityMapping, types: UrbanTypes
) -> tuple[dict[str, CreationSpec], dict[str, str]]:
    """Specs of the groups to create, and the reasons some groups are not created."""
    assignment = mapping.assignment()
    specs: dict[str, CreationSpec] = {}
    problems: dict[str, str] = {}
    for proposal in mapping.proposals:
        if proposal.signature not in assignment:
            continue
        spec, problem = creation_spec(proposal, types)
        if spec is not None:
            specs[proposal.signature] = spec
        if problem:
            problems[proposal.signature] = problem
    return specs, problems


def render_type_question(pending: list[MappingProposal]) -> str:
    lines = [
        "Не удалось уверенно определить, какими объектами Urban API создать часть "
        "объектов ваших файлов. Проверьте сопоставление:",
        "",
    ]
    for number, item in enumerate(pending, start=1):
        target = item.target.name if item.target else "не загружать"
        where = ", ".join(item.layers[:3])
        lines.append(
            f"{number}. {item.label} ({item.count} об., {where}) → «{target}»"
            f" — уверенность {round(item.confidence * 100)}%"
        )
    lines += [
        "",
        "Ответьте «да», чтобы принять всё, «нет», чтобы не загружать эти объекты, "
        "или исправьте пункты: например, «2 — Зеленая зона; 3 — нет».",
    ]
    return "\n".join(lines)
