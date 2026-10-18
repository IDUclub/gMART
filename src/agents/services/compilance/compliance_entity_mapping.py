"""Which norm entity each object of the user's files is.

Norms name their objects with Urban API types — «Жилой дом», «Школа», «Детская
площадка». User files describe objects with tags instead (``type=buildings``,
``building=residential``, ``amenity=school``). Objects with the same classifying
tags form a group, and a group is mapped as a whole:

1. A table of tag rules maps the groups whose meaning is settled.
2. The LLM classifies the rest, choosing only among Urban API types — the entities
   the selected norms need and the table's own, never functional zones — and says
   how sure it is. Its answers are kept with the upload, so a rerun (after the
   user's answer) maps the same groups the same way.
3. A proposal below the confidence threshold is put to the user only when the
   group's objects would serve a norm: design objects of an entity some norm needs,
   surroundings of an entity some norm reads as a source or neighbour. The answer
   is kept with the upload.

The mapped objects are normalised into Urban-API-like layers: ``building.floors``,
``building.height`` and the other attributes norms read are filled from the
generator's fields (``levels``, ``levelHeights``, ``livingSpace`` …).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field
from shapely.geometry import shape
from shapely.ops import unary_union

from src.agents.services.compilance.compliance_scope import normalized
from src.agents.services.compilance.compliance_uploads import UploadedLayer, UserFiles
from src.agents.services.normgraph.normgraph_reasoning import _request_json

EntityType = Literal["service", "physical_object", "functional_zone"]
MappingSource = Literal["table", "llm", "user", "none"]

CONFIDENCE_THRESHOLD = 0.8
CLASSIFY_KEYS = (
    "type",
    "building",
    "amenity",
    "leisure",
    "landuse",
    "natural",
    "highway",
    "shop",
    "man_made",
    "healthcare",
    "blockType",
)
# Structural features of the generator: the design boundary and road entry points.
_IGNORED_TYPES = {"border", "entry_point"}
_LLM_BATCH = 12
_EXAMPLE_KEYS = 12

_RESIDENTIAL = {
    "residential",
    "apartments",
    "house",
    "detached",
    "semidetached_house",
    "terrace",
    "dormitory",
    "bungalow",
}
_NON_RESIDENTIAL = {
    "commercial",
    "office",
    "retail",
    "industrial",
    "warehouse",
    "garage",
    "garages",
    "service",
    "civic",
    "public",
    "hotel",
    "kiosk",
    "roof",
    "security_booth",
    "toilets",
    "stands",
    "ruins",
    "shed",
    "hangar",
    "church",
    "cathedral",
    "train_station",
    "transportation",
}
_LOCAL_ROADS = {
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
    "tertiary_link",
    "residential",
    "living_street",
    "unclassified",
    "service",
    "road",
}
_MAJOR_ROADS = {"motorway", "motorway_link", "trunk", "trunk_link"}
# Generator block types → Urban API functional zone types.
_BLOCK_ZONES: dict[str, tuple[str | None, float]] = {
    "residential": ("residential", 0.9),
    "special": ("special", 0.85),
    "square": ("recreation", 0.8),
    "surface_parking": ("transport", 0.8),
    "school": ("business", 0.6),
    "police": ("business", 0.6),
    "empty": (None, 1.0),
}


@dataclass(frozen=True)
class EntityTarget:
    name: str
    entity_type: EntityType

    def key(self) -> str:
        return f"{self.entity_type}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {"entity": self.name, "entity_type": self.entity_type}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EntityTarget | None":
        if not data or not data.get("entity"):
            return None
        return cls(name=str(data["entity"]), entity_type=data["entity_type"])


@dataclass
class TagGroup:
    """Objects of the user's files sharing classifying tags."""

    signature: str
    tags: dict[str, str]
    count: int = 0
    roles: dict[str, int] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    example: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.tags.items()) or (
            "без классифицирующих тегов"
        )


@dataclass
class MappingProposal:
    signature: str
    label: str
    count: int
    layers: list[str]
    roles: dict[str, int]
    target: EntityTarget | None
    confidence: float
    source: MappingSource
    reason: str | None = None

    @property
    def confident(self) -> bool:
        return self.source == "user" or self.confidence >= CONFIDENCE_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "label": self.label,
            "count": self.count,
            "layers": self.layers,
            "roles": self.roles,
            **(
                self.target.to_dict()
                if self.target
                else {"entity": None, "entity_type": None}
            ),
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MappingProposal":
        return cls(
            signature=data["signature"],
            label=data.get("label") or data["signature"],
            count=int(data.get("count") or 0),
            layers=list(data.get("layers") or []),
            roles=dict(data.get("roles") or {}),
            target=EntityTarget.from_dict(data),
            confidence=float(data.get("confidence") or 0),
            source=data.get("source") or "none",
            reason=data.get("reason"),
        )


@dataclass
class EntityMapping:
    """Every tag group's entity; ``pending`` waits for the user's confirmation."""

    proposals: list[MappingProposal]
    pending: list[MappingProposal] = field(default_factory=list)
    # LLM answers of this run, keyed by ``classification_key``, to keep with the upload.
    classified: dict[str, dict[str, Any]] = field(default_factory=dict)

    def assignment(self) -> dict[str, EntityTarget]:
        pending = {item.signature for item in self.pending}
        return {
            item.signature: item.target
            for item in self.proposals
            if item.target is not None
            and item.signature not in pending
            and item.confident
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [item.to_dict() for item in self.proposals],
            "pending": [item.signature for item in self.pending],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EntityMapping":
        proposals = [MappingProposal.from_dict(item) for item in data["proposals"]]
        pending = set(data.get("pending") or [])
        return cls(
            proposals=proposals,
            pending=[item for item in proposals if item.signature in pending],
        )


def feature_tags(properties: dict[str, Any]) -> dict[str, str]:
    tags = {}
    for key in CLASSIFY_KEYS:
        value = properties.get(key)
        if value is None or isinstance(value, (dict, list)):
            continue
        text = str(value).strip()
        if text:
            tags[key] = text.casefold()
    return tags


def signature_of(tags: dict[str, str]) -> str:
    return "|".join(f"{key}={value}" for key, value in tags.items())


def tag_groups(files: UserFiles) -> list[TagGroup]:
    """Groups of the project and context objects by their classifying tags."""
    groups: dict[str, TagGroup] = {}
    for layer in files.layers:
        if layer.role == "border":
            continue
        for feature in layer.features:
            properties = feature.get("properties") or {}
            tags = feature_tags(properties)
            signature = signature_of(tags)
            group = groups.get(signature)
            if group is None:
                group = groups[signature] = TagGroup(
                    signature=signature,
                    tags=tags,
                    example=_example(properties),
                )
            group.count += 1
            group.roles[layer.role] = group.roles.get(layer.role, 0) + 1
            if layer.name not in group.layers:
                group.layers.append(layer.name)
    return sorted(groups.values(), key=lambda item: (-item.count, item.signature))


def _example(properties: dict[str, Any]) -> dict[str, Any]:
    example = {}
    for key, value in properties.items():
        if isinstance(value, (dict, list)) or value in (None, ""):
            continue
        text = str(value)
        example[key] = text if len(text) <= 60 else text[:57] + "…"
        if len(example) >= _EXAMPLE_KEYS:
            break
    return example


def table_proposal(tags: dict[str, str]) -> tuple[EntityTarget | None, float] | None:
    """The table's entity for a tag group; ``None`` when the table has no rule.

    ``(None, 1.0)`` is a settled «no entity» (the boundary, empty blocks).
    """
    kind = tags.get("type", "")
    building = tags.get("building", "")
    amenity = tags.get("amenity", "")
    leisure = tags.get("leisure", "")

    def service(name: str, confidence: float = 1.0):
        return EntityTarget(name, "service"), confidence

    def physical(name: str, confidence: float = 1.0):
        return EntityTarget(name, "physical_object"), confidence

    if kind in _IGNORED_TYPES:
        return None, 1.0
    if kind == "block":
        zone, confidence = _BLOCK_ZONES.get(tags.get("blockType", ""), (None, 0.0))
        if zone is None:
            return (None, 1.0) if confidence else None
        return EntityTarget(zone, "functional_zone"), confidence
    if "school" in {amenity, building}:
        return service("Школа")
    if amenity in {"kindergarten", "childcare"} or building == "kindergarten":
        return service("Детский сад")
    if "university" in {amenity, building}:
        return service("Высшее учебное заведение")
    if "college" in {amenity, building}:
        return service("Среднее специальное учебное заведение")
    if amenity == "hospital" or building == "hospital":
        return service("Больница")
    if amenity in {"clinic", "doctors"} or tags.get("healthcare") == "clinic":
        return service("Поликлиника")
    if amenity == "pharmacy":
        return service("Аптека")
    if building == "sports_centre" or leisure == "sports_centre":
        return service("Спортивный зал")
    if kind == "sport" or leisure in {"pitch", "fitness_station", "track"}:
        return service("Спортивная площадка")
    if leisure == "dog_park":
        return service("Площадка для выгула собак")
    if kind == "playground" or (
        kind not in {"calm_rest", "sport"} and "playground" in {amenity, leisure}
    ):
        return service("Детская площадка")
    if leisure == "park":
        return service("Парк")
    if building == "parking" or amenity == "parking":
        return physical("Парковка")
    if kind == "buildings" or building:
        if building in _RESIDENTIAL:
            return physical("Жилой дом")
        if building in _NON_RESIDENTIAL:
            return physical("Нежилое здание")
    highway = tags.get("highway", "")
    if highway in _LOCAL_ROADS:
        return physical("Местная дорога")
    if highway in _MAJOR_ROADS:
        return physical("Региональная дорога", 0.6)
    if kind == "water" or tags.get("natural") == "water":
        return physical("Водный объект")
    if tags.get("natural") == "wood" or tags.get("landuse") == "forest":
        return physical("Лес")
    if kind == "greenery":
        return physical("Зеленая зона", 0.85)
    return None


TABLE_TARGETS: tuple[EntityTarget, ...] = (
    EntityTarget("Школа", "service"),
    EntityTarget("Детский сад", "service"),
    EntityTarget("Высшее учебное заведение", "service"),
    EntityTarget("Среднее специальное учебное заведение", "service"),
    EntityTarget("Больница", "service"),
    EntityTarget("Поликлиника", "service"),
    EntityTarget("Аптека", "service"),
    EntityTarget("Спортивный зал", "service"),
    EntityTarget("Спортивная площадка", "service"),
    EntityTarget("Площадка для выгула собак", "service"),
    EntityTarget("Детская площадка", "service"),
    EntityTarget("Парк", "service"),
    EntityTarget("Парковка", "physical_object"),
    EntityTarget("Жилой дом", "physical_object"),
    EntityTarget("Нежилое здание", "physical_object"),
    EntityTarget("Площадка", "physical_object"),
    EntityTarget("Местная дорога", "physical_object"),
    EntityTarget("Региональная дорога", "physical_object"),
    EntityTarget("Водный объект", "physical_object"),
    EntityTarget("Лес", "physical_object"),
    EntityTarget("Зеленая зона", "physical_object"),
)


class _ClassifiedGroup(BaseModel):
    id: int
    entity: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    reason: str | None = None


class _Classification(BaseModel):
    items: list[_ClassifiedGroup] = Field(default_factory=list)


class _OverrideItem(BaseModel):
    number: int
    entity: str | None = None


class _MappingReply(BaseModel):
    kind: Literal["accept", "reject", "edit", "new_request"] = "new_request"
    overrides: list[_OverrideItem] = Field(default_factory=list)


_CLASSIFY_PROMPT = """Ты сопоставляешь объекты из файла проекта застройки с типами \
объектов Urban API, которые упоминаются в градостроительных нормах.

Каждая группа "groups" — объекты с одинаковыми тегами (OSM-подобные теги \
генератора застройки: type, building, amenity, leisure, highway …), с числом \
объектов, именами слоёв и примером атрибутов. Для каждой группы выбери ОДИН тип из \
списка "candidates" (поле name) или null, если ни один тип не подходит. Не придумывай \
типов вне списка. confidence — насколько ты уверен (0..1): 0.9 и выше — только если \
теги однозначно означают этот тип; если тип лишь похож (например, «площадка для \
тихого отдыха» и «детская площадка») — не выше 0.6. reason — одна короткая фраза.

Верни только валидный JSON без markdown:
{"items": [{"id": 1, "entity": "Детская площадка", "confidence": 0.7, "reason": "..."}]}
"""

_REPLY_PROMPT = """Пользователю показали нумерованный список сопоставлений объектов \
его файлов с типами объектов норм ("items": номер → предложенный тип) и попросили \
подтвердить. Разбери его ответ "reply":
- kind = "accept" — согласен со всеми предложениями;
- kind = "reject" — не согласен ни с одним (такие объекты не проверять);
- kind = "edit" — меняет отдельные пункты: overrides — номер и новый тип из списка \
"candidates" (entity = null — не проверять этот пункт); неупомянутые пункты принимаются;
- kind = "new_request" — это не ответ на список, а новый вопрос или просьба.
Верни только валидный JSON без markdown:
{"kind": "edit", "overrides": [{"number": 2, "entity": "Детская площадка"}]}
"""

_ACCEPT = re.compile(
    r"^(?:да|ок|ok|окей|хорошо|верно|все верно|всё верно|подтверждаю|согласен|"
    r"согласна|принимаю|да, верно|да верно|продолжай|запускай|все так|всё так)$"
)
_REJECT = re.compile(
    r"^(?:нет|не верно|неверно|отклоняю|ни одного|не проверяй|не загружай|не создавай)$"
)
# «2 — школа; 3 нет, 4: детская площадка» → one item per part.
_EDIT_SPLIT = re.compile(r"[;\n]|,(?=\s*\d)")
_EDIT_ITEM = re.compile(r"^\s*(?P<number>\d+)\s*(?:[-–—:.)=→]|это)?\s*(?P<value>.*)$")
_SKIP_VALUES = {
    "нет",
    "пропустить",
    "пропусти",
    "не проверять",
    "не проверяй",
    "не загружать",
    "не создавать",
    "ничего",
    "никакой",
    "null",
    "-",
}


@dataclass(frozen=True)
class MappingReply:
    kind: Literal["accept", "reject", "edit", "new_request", "unresolved"]
    overrides: dict[str, EntityTarget | None] = field(default_factory=dict)


class EntityMapper:
    """Map tag groups to entities: table first, the LLM for the rest."""

    def __init__(self, llm_client, classify_prompt: str = _CLASSIFY_PROMPT) -> None:
        self.llm_client = llm_client
        self.classify_prompt = classify_prompt

    async def propose(
        self,
        model: str,
        files: UserFiles,
        needed: Iterable[EntityTarget],
        sources: Iterable[EntityTarget] | None = None,
        llm_types: Iterable[EntityType] = ("service", "physical_object"),
    ) -> EntityMapping:
        """Map every tag group of the files to an entity.

        ``needed`` are the entities of the selected norms (canonical names);
        ``sources`` the ones a norm reads as a source, neighbour or zone — the only
        entities the files' surroundings can serve. ``None`` takes every needed
        entity as one. ``llm_types`` are the entity types the LLM may choose.
        """
        needed = list(needed)
        needed_keys = {target.key() for target in needed}
        source_keys = (
            needed_keys if sources is None else {target.key() for target in sources}
        )
        candidates = _unique_targets([*needed, *TABLE_TARGETS])
        # Zones come only from the generator's blocks (the table): any other object
        # taken for a zone would turn a site into an area-wide restriction.
        allowed = set(llm_types) - {"functional_zone"}
        llm_candidates = [
            target for target in candidates if target.entity_type in allowed
        ]
        proposals: list[MappingProposal] = []
        unknown: list[TagGroup] = []
        for group in tag_groups(files):
            proposal = self._base(group)
            confirmed = files.confirmed_mapping.get(group.signature, ...)
            if confirmed is not ...:
                proposal.target = EntityTarget.from_dict(confirmed)
                proposal.confidence = 1.0
                proposal.source = "user"
                proposals.append(proposal)
                continue
            ruled = table_proposal(group.tags)
            if ruled is not None:
                proposal.target, proposal.confidence = ruled
                proposal.source = "table"
                proposals.append(proposal)
                continue
            cached = files.classified.get(
                classification_key(group.signature, llm_candidates)
            )
            if cached is not None:
                self._apply_llm(
                    proposal,
                    EntityTarget.from_dict(cached),
                    float(cached.get("confidence") or 0),
                    cached.get("reason"),
                )
            else:
                unknown.append(group)
            proposals.append(proposal)

        new_answers: dict[str, dict[str, Any]] = {}
        if unknown:
            classified = await self._classify(model, unknown, llm_candidates)
            for proposal in proposals:
                if proposal.signature not in classified:
                    continue
                target, confidence, reason = classified[proposal.signature]
                self._apply_llm(proposal, target, confidence, reason)
                new_answers[classification_key(proposal.signature, llm_candidates)] = {
                    **(
                        target.to_dict()
                        if target
                        else {"entity": None, "entity_type": None}
                    ),
                    "confidence": confidence,
                    "reason": reason,
                }

        # Only a doubt that changes some norm's data is worth the user's time.
        pending = [
            item
            for item in proposals
            if item.target is not None
            and not item.confident
            and self._serves_a_norm(item, needed_keys, source_keys)
        ]
        logger.info(
            "Compliance entity mapping: groups={} table={} llm={} pending={}",
            len(proposals),
            sum(item.source == "table" for item in proposals),
            sum(item.source == "llm" for item in proposals),
            len(pending),
        )
        return EntityMapping(
            proposals=proposals, pending=pending, classified=new_answers
        )

    @staticmethod
    def _apply_llm(
        proposal: MappingProposal,
        target: EntityTarget | None,
        confidence: float,
        reason: str | None,
    ) -> None:
        proposal.target = target
        proposal.confidence = confidence if target is not None else 0.0
        proposal.reason = reason
        proposal.source = "llm" if target is not None else "none"

    @staticmethod
    def _serves_a_norm(
        item: MappingProposal, needed_keys: set[str], source_keys: set[str]
    ) -> bool:
        """Design objects serve any needed entity; surroundings only sources."""
        key = item.target.key()
        return bool(
            (item.roles.get("project") and key in needed_keys)
            or (item.roles.get("context") and key in source_keys)
        )

    @staticmethod
    def _base(group: TagGroup) -> MappingProposal:
        return MappingProposal(
            signature=group.signature,
            label=group.label(),
            count=group.count,
            layers=list(group.layers),
            roles=dict(group.roles),
            target=None,
            confidence=0.0,
            source="none",
        )

    async def _classify(
        self,
        model: str,
        groups: list[TagGroup],
        candidates: list[EntityTarget],
    ) -> dict[str, tuple[EntityTarget | None, float, str | None]]:
        by_name = {normalized(target.name): target for target in candidates}
        result: dict[str, tuple[EntityTarget | None, float, str | None]] = {}
        for start in range(0, len(groups), _LLM_BATCH):
            batch = groups[start : start + _LLM_BATCH]
            payload = {
                "candidates": [
                    {"name": target.name, "entity_type": target.entity_type}
                    for target in candidates
                ],
                "groups": [
                    {
                        "id": index,
                        "tags": group.tags,
                        "count": group.count,
                        "layers": group.layers,
                        "example": group.example,
                    }
                    for index, group in enumerate(batch, start=1)
                ],
            }
            try:
                answer = await _request_json(
                    self.llm_client,
                    model,
                    [
                        {"role": "system", "content": self.classify_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    _Classification,
                )
            except Exception as exc:  # the groups stay unmapped, the check goes on
                logger.warning(f"Entity classification failed: {exc}")
                continue
            for item in answer.items:
                if not 1 <= item.id <= len(batch):
                    continue
                group = batch[item.id - 1]
                target = by_name.get(normalized(item.entity)) if item.entity else None
                # A name outside the candidates is no answer at all.
                confidence = item.confidence if target is not None else 0.0
                result[group.signature] = (target, confidence, item.reason)
        return result

    async def resolve_reply(
        self,
        model: str,
        reply: str,
        pending: list[MappingProposal],
        candidates: list[EntityTarget],
    ) -> MappingReply:
        """How the user's message answers the pending mapping question."""
        text = normalized(reply).strip(" .!")
        if not text:
            return MappingReply(kind="unresolved")
        if _ACCEPT.match(text):
            return MappingReply(kind="accept")
        if _REJECT.match(text):
            return MappingReply(kind="reject")
        by_name = {normalized(target.name): target for target in candidates}
        edits = _parse_edits(text, pending, by_name)
        if edits is not None:
            return MappingReply(kind="edit", overrides=edits)
        try:
            answer = await _request_json(
                self.llm_client,
                model,
                [
                    {"role": "system", "content": _REPLY_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "items": {
                                    str(number): (
                                        item.target.name if item.target else None
                                    )
                                    for number, item in enumerate(pending, start=1)
                                },
                                "candidates": [target.name for target in candidates],
                                "reply": reply,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                _MappingReply,
            )
        except Exception as exc:
            logger.warning(f"Mapping reply classification failed: {exc}")
            return MappingReply(kind="unresolved")
        if answer.kind != "edit":
            return MappingReply(kind=answer.kind)
        overrides: dict[str, EntityTarget | None] = {}
        for item in answer.overrides:
            if not 1 <= item.number <= len(pending):
                return MappingReply(kind="unresolved")
            if item.entity is None:
                overrides[pending[item.number - 1].signature] = None
                continue
            target = by_name.get(normalized(item.entity))
            if target is None:
                return MappingReply(kind="unresolved")
            overrides[pending[item.number - 1].signature] = target
        return MappingReply(kind="edit", overrides=overrides)


def _parse_edits(
    text: str,
    pending: list[MappingProposal],
    by_name: dict[str, EntityTarget],
) -> dict[str, EntityTarget | None] | None:
    """«2 — школа; 3 нет» without the LLM; ``None`` when the reply is not that."""
    if not re.match(r"^\d", text):
        return None
    overrides: dict[str, EntityTarget | None] = {}
    for part in _EDIT_SPLIT.split(text):
        if not part.strip():
            continue
        match = _EDIT_ITEM.match(part)
        if match is None:
            return None
        number = int(match.group("number"))
        if not 1 <= number <= len(pending):
            return None
        value = match.group("value").strip(' .,;:«»"')
        signature = pending[number - 1].signature
        if not value or value in _SKIP_VALUES:
            overrides[signature] = None
            continue
        target = by_name.get(value)
        if target is None:
            return None
        overrides[signature] = target
    return overrides or None


def _unique_targets(targets: Iterable[EntityTarget]) -> list[EntityTarget]:
    return list(dict.fromkeys(targets))


def classification_key(signature: str, candidates: Iterable[EntityTarget]) -> str:
    """A cached LLM answer holds only for the candidate list it was chosen from."""
    names = "\n".join(sorted({target.key() for target in candidates}))
    digest = hashlib.sha1(names.encode("utf-8")).hexdigest()[:12]
    return f"{signature}#{digest}"


def apply_reply(mapping: EntityMapping, reply: MappingReply) -> dict[str, Any]:
    """The confirmations the reply makes, keyed by tag signature."""
    confirmations: dict[str, Any] = {}
    for item in mapping.pending:
        if reply.kind == "reject":
            confirmations[item.signature] = None
        elif item.signature in reply.overrides:
            target = reply.overrides[item.signature]
            confirmations[item.signature] = target.to_dict() if target else None
        else:
            confirmations[item.signature] = (
                item.target.to_dict() if item.target else None
            )
    return confirmations


def render_mapping_question(pending: list[MappingProposal]) -> str:
    lines = [
        "Не удалось уверенно определить, какие объекты норм соответствуют части "
        "объектов ваших файлов. Проверьте сопоставление:",
        "",
    ]
    for number, item in enumerate(pending, start=1):
        target = item.target.name if item.target else "не проверять"
        where = ", ".join(item.layers[:3])
        lines.append(
            f"{number}. {item.label} ({item.count} об., {where}) → «{target}»"
            f" — уверенность {round(item.confidence * 100)}%"
        )
    lines += [
        "",
        "Ответьте «да», чтобы принять всё, «нет», чтобы не проверять эти объекты, "
        "или исправьте пункты: например, «2 — Спортивная площадка; 3 — нет».",
    ]
    return "\n".join(lines)


def mapping_options(pending: list[MappingProposal]) -> list[dict[str, Any]]:
    return [
        {
            "number": number,
            "label": item.label,
            "value": item.target.name if item.target else None,
            "entity_type": item.target.entity_type if item.target else None,
            "confidence": round(item.confidence, 3),
            "count": item.count,
            "layers": item.layers,
        }
        for number, item in enumerate(pending, start=1)
    ]


# --- normalised layers ------------------------------------------------------------


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _level_heights(value: Any) -> list[float]:
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = value.split(",")
    else:
        return []
    heights = [_number(item) for item in items]
    return [item for item in heights if item is not None and item > 0]


def normalized_properties(
    properties: dict[str, Any], target: EntityTarget
) -> dict[str, Any]:
    """Urban-API-like attributes norms read, filled from the generator's fields."""
    heights = _level_heights(properties.get("levelHeights"))
    floors = next(
        (
            value
            for value in (
                _number(properties.get(key))
                for key in ("levels", "building:levels", "floors", "storeys")
            )
            if value is not None and value > 0
        ),
        float(len(heights)) if heights else None,
    )
    height = next(
        (
            value
            for value in (
                _number(properties.get(key))
                for key in ("height", "building:height", "height_m")
            )
            if value is not None and value > 0
        ),
        round(sum(heights), 2) if heights else None,
    )
    building = {
        "floors": int(floors) if floors is not None else None,
        "height": height,
        "living_area": _number(
            properties.get("livingSpace") or properties.get("living_space")
        ),
        "footprint_area": _number(properties.get("groundSpace")),
        "total_area": _number(properties.get("building:total_area")),
    }
    result: dict[str, Any] = {
        "building": {
            key: value for key, value in building.items() if value is not None
        },
        "floors": building["floors"],
        "height_m": height,
        "population": _number(properties.get("population")),
        "capacity": _number(properties.get("capacity")),
        "area": _number(properties.get("area")),
    }
    if target.entity_type == "functional_zone":
        result["functional_zone_type"] = {"name": target.name}
    for key in ("FSI_total", "GSI_total", "FSI_living", "GSI_living"):
        if properties.get(key) is not None:
            result[key] = _number(properties.get(key))
    return {key: value for key, value in result.items() if value not in (None, {})}


@dataclass
class EntityLayers:
    """Normalised user objects by entity, split into the design and its context."""

    project: dict[EntityTarget, list[dict[str, Any]]]
    context: dict[EntityTarget, list[dict[str, Any]]]
    border: Any | None  # shapely geometry of the design boundary

    def entities(self, role: Literal["project", "context"]) -> set[EntityTarget]:
        return set(self.project if role == "project" else self.context)

    def counts(self) -> dict[str, dict[str, int]]:
        return {
            target.name: {
                "project": len(self.project.get(target, [])),
                "context": len(self.context.get(target, [])),
            }
            for target in {*self.project, *self.context}
        }


def build_entity_layers(
    files: UserFiles, assignment: dict[str, EntityTarget]
) -> EntityLayers:
    project: dict[EntityTarget, list[dict[str, Any]]] = {}
    context: dict[EntityTarget, list[dict[str, Any]]] = {}
    borders = []
    for layer in files.layers:
        if layer.role == "border":
            borders.extend(_geometries(layer))
            continue
        bucket = project if layer.role == "project" else context
        for index, feature in enumerate(layer.features):
            properties = feature.get("properties") or {}
            target = assignment.get(signature_of(feature_tags(properties)))
            if target is None:
                continue
            bucket.setdefault(target, []).append(
                _normalized_feature(layer, index, feature, target)
            )
    border = unary_union(borders) if borders else None
    return EntityLayers(project=project, context=context, border=border)


def _geometries(layer: UploadedLayer) -> list[Any]:
    result = []
    for feature in layer.features:
        try:
            geometry = shape(feature["geometry"])
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if not geometry.is_empty:
            result.append(geometry if geometry.is_valid else geometry.buffer(0))
    return result


def _normalized_feature(
    layer: UploadedLayer, index: int, feature: dict[str, Any], target: EntityTarget
) -> dict[str, Any]:
    properties = feature.get("properties") or {}
    source_id = feature.get("id", properties.get("id"))
    source_id = index if source_id in (None, "") else source_id
    name = properties.get("name") or (
        f"{target.name} — {layer.name}, №{source_id}"
        if layer.role == "project"
        else None
    )
    return {
        "type": "Feature",
        # The id becomes the object reference of every verdict about it.
        "id": f"user/{layer.name}/{source_id}",
        "geometry": feature.get("geometry"),
        "properties": {
            **normalized_properties(properties, target),
            "name": name,
            "entity": target.name,
            "source_layer": layer.name,
            "user_role": layer.role,
            "is_project": layer.role == "project",
            # Check tools flatten nested properties into columns; the tags and the
            # id identify the object without copying every generator field.
            "source_id": str(source_id),
            "source_tags": signature_of(feature_tags(properties)).replace("|", ", "),
        },
    }
