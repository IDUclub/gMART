"""Deterministic handling of Russian scenario type questions.

The general scenario-data agent still plans over the runtime Urban MCP catalogue.  Questions
about counts by object/service type and ambiguous catalogue scope are different: their
arithmetic, entity identity, dictionary join and clarification are deterministic contracts,
so an LLM must not infer them from a sample.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.agents.services.scenario_data_aggregate import extract_records


class ScenarioEntityKind(StrEnum):
    PHYSICAL_OBJECT = "physical_object"
    SERVICE = "service"


@dataclass(frozen=True)
class ScenarioTypeIntent:
    """A supported deterministic query, or a question needed to disambiguate it."""

    kinds: tuple[ScenarioEntityKind, ...] = ()
    effective_query: str = ""
    clarification: str | None = None


@dataclass(frozen=True)
class TypeDistribution:
    kind: ScenarioEntityKind
    total_unique: int
    available_types: int
    present_types: int
    rows: list[dict[str, Any]]


_COUNT_MARKERS = (
    "сколько",
    "количеств",
    "по тип",
    "по видам",
    "разбив",
    "распредел",
)
_PHYSICAL_MARKERS = ("физическ", "физобъект")
_SERVICE_MARKERS = ("сервис", "услуг")
_GENERIC_OBJECT_MARKERS = ("объект",)
_SERVICE_LIST_MARKERS = (
    "какие тип",
    "какие виды",
    "какие бывают",
    "доступн",
    "перечисл",
    "список",
)
_SERVICE_SCOPE_MARKERS = (
    "в проект",
    "для проект",
    "в сценар",
    "для сценар",
    "на территор",
    "представлен",
    "фактически",
    "в базе",
    "из базы",
    "в справочник",
    "из справочник",
    "общий справочник",
    "полный справочник",
    "в системе",
    "вообще",
)
_BOTH_MARKERS = (
    "и те и те",
    "и то и другое",
    "оба варианта",
    "оба набора",
    "обе группы",
    "все вместе",
)


def classify_type_query(
    user_query: str, history: list[dict[str, Any]] | None = None
) -> ScenarioTypeIntent | None:
    """Recognise Russian questions about project types or ambiguous catalogue scope.

    A bare ``объекты`` is deliberately ambiguous: in the product vocabulary it may mean
    physical objects or the services attached to them.  A short follow-up such as
    ``физические`` is resolved against the last ambiguous user question in chat history.
    """

    direct = _classify_text(user_query)
    if direct is not None:
        return direct

    selected = _selected_kinds(user_query)
    if not selected or not history:
        return None

    for message in reversed(history):
        if str(message.get("role") or "").lower() != "user":
            continue
        previous = _content_text(message.get("content"))
        previous_intent = _classify_text(previous)
        if previous_intent is None or previous_intent.clarification is None:
            continue
        return ScenarioTypeIntent(
            kinds=selected,
            effective_query=f"{previous}\nУточнение пользователя: {user_query}",
        )
    return None


def _classify_text(text: str) -> ScenarioTypeIntent | None:
    lowered = text.lower().replace("ё", "е")
    has_entity = any(
        marker in lowered
        for marker in (
            *_GENERIC_OBJECT_MARKERS,
            *_PHYSICAL_MARKERS,
            *_SERVICE_MARKERS,
        )
    )
    asks_for_service_list = any(
        marker in lowered for marker in _SERVICE_LIST_MARKERS
    ) and any(marker in lowered for marker in _SERVICE_MARKERS)
    has_explicit_scope = any(marker in lowered for marker in _SERVICE_SCOPE_MARKERS)
    if asks_for_service_list and not has_explicit_scope:
        return ScenarioTypeIntent(
            effective_query=text,
            clarification=(
                "Уточните, пожалуйста: нужен полный список типов городских "
                "сервисов из общего справочника или только типы сервисов, "
                "фактически представленные в текущем проекте/сценарии?"
            ),
        )
    if not has_entity or not any(marker in lowered for marker in _COUNT_MARKERS):
        return None

    selected = _selected_kinds(lowered)
    has_physical = ScenarioEntityKind.PHYSICAL_OBJECT in selected
    has_service = ScenarioEntityKind.SERVICE in selected
    has_generic = any(marker in lowered for marker in _GENERIC_OBJECT_MARKERS)

    if not has_physical and not has_service and has_generic:
        return ScenarioTypeIntent(
            effective_query=text,
            clarification=(
                "Уточните, пожалуйста: посчитать физические объекты, сервисы "
                "(услуги) или оба набора?"
            ),
        )
    return ScenarioTypeIntent(kinds=selected, effective_query=text)


def _selected_kinds(text: str) -> tuple[ScenarioEntityKind, ...]:
    lowered = text.lower().replace("ё", "е")
    explicitly_both = re.search(
        r"(?:объект\w*\s+и\s+(?:сервис|услуг)|(?:сервис|услуг)\w*\s+и\s+объект)",
        lowered,
    )
    if explicitly_both or any(marker in lowered for marker in _BOTH_MARKERS):
        return (ScenarioEntityKind.PHYSICAL_OBJECT, ScenarioEntityKind.SERVICE)
    kinds = []
    if any(marker in lowered for marker in _PHYSICAL_MARKERS):
        kinds.append(ScenarioEntityKind.PHYSICAL_OBJECT)
    if any(marker in lowered for marker in _SERVICE_MARKERS):
        kinds.append(ScenarioEntityKind.SERVICE)
    return tuple(kinds)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


_KIND_FIELDS = {
    ScenarioEntityKind.PHYSICAL_OBJECT: {
        "entity_id": "physical_object_id",
        "type_container": "physical_object_type",
        "type_id": "physical_object_type_id",
        "parent": "physical_object_function",
    },
    ScenarioEntityKind.SERVICE: {
        "entity_id": "service_id",
        "type_container": "service_type",
        "type_id": "service_type_id",
        "parent": "urban_function",
    },
}


def build_type_distribution(
    entity_result: Any,
    project_catalog_result: Any,
    kind: ScenarioEntityKind,
    *,
    fallback_catalog_result: Any | None = None,
) -> TypeDistribution:
    """Count unique entities and join every represented type to the Urban MCP dictionary."""

    fields = _KIND_FIELDS[kind]
    entities = extract_records(entity_result) or []
    project_catalog = extract_records(project_catalog_result) or []
    fallback_catalog = extract_records(fallback_catalog_result) or []

    unique_entities = _unique_by(entities, fields["entity_id"])
    counts: Counter[str] = Counter()
    observed: dict[str, dict[str, Any]] = {}
    for entity in unique_entities:
        type_value = entity.get(fields["type_container"])
        if not isinstance(type_value, dict):
            counts["__unknown__"] += 1
            continue
        type_id = _type_id(type_value, fields["type_id"])
        if type_id is None:
            counts["__unknown__"] += 1
            continue
        key = str(type_id)
        counts[key] += 1
        observed.setdefault(key, type_value)

    project_index = _catalog_index(project_catalog, fields)
    fallback_index = _catalog_index(fallback_catalog, fields)
    rows: list[dict[str, Any]] = []
    for key, entry in project_index.items():
        count = counts.get(key, 0)
        if count:
            rows.append(_exact_row(key, entry, count))

    for key, count in counts.items():
        if key in project_index:
            continue
        if key == "__unknown__":
            rows.append(
                {
                    "type_id": "—",
                    "type_name": "Тип не указан",
                    "count": count,
                    "status": "не найден в данных",
                    "possible_types": "—",
                }
            )
            continue
        exact = fallback_index.get(key)
        if exact is not None:
            rows.append(_exact_row(key, exact, count))
            continue

        observed_value = observed.get(key) or {}
        candidates = _related_candidates(
            observed_value, project_catalog + fallback_catalog, fields
        )
        rows.append(
            {
                "type_id": _display_id(key),
                "type_name": "Не определён",
                "count": count,
                "status": "предположение" if candidates else "не найден в справочнике",
                "possible_types": ", ".join(candidates) if candidates else "—",
            }
        )

    rows.sort(
        key=lambda row: (
            0 if int(row["count"]) > 0 else 1,
            -int(row["count"]),
            str(row["type_name"]).lower(),
            str(row["type_id"]),
        )
    )
    return TypeDistribution(
        kind=kind,
        total_unique=len(unique_entities),
        available_types=len(rows),
        present_types=sum(1 for row in rows if int(row["count"]) > 0),
        rows=rows,
    )


def _unique_by(records: list[dict[str, Any]], id_field: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for record in records:
        identifier = record.get(id_field)
        if identifier is None:
            identifier = json.dumps(
                record, ensure_ascii=False, sort_keys=True, default=str
            )
        key = str(identifier)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def _type_id(value: dict[str, Any], field: str) -> Any:
    return value.get(field, value.get("id"))


def _catalog_index(
    catalog: list[dict[str, Any]], fields: dict[str, str]
) -> dict[str, dict[str, Any]]:
    indexed = {}
    for entry in catalog:
        type_id = _type_id(entry, fields["type_id"])
        if type_id is not None:
            indexed[str(type_id)] = entry
    return indexed


def _exact_row(key: str, entry: dict[str, Any], count: int) -> dict[str, Any]:
    return {
        "type_id": _display_id(key),
        "type_name": str(entry.get("name") or "Без названия"),
        "count": count,
        "status": "точное соответствие",
        "possible_types": "—",
    }


def _related_candidates(
    observed: dict[str, Any],
    catalog: list[dict[str, Any]],
    fields: dict[str, str],
) -> list[str]:
    observed_parent = observed.get(fields["parent"])
    if not isinstance(observed_parent, dict) or observed_parent.get("id") is None:
        return []
    parent_id = str(observed_parent["id"])
    candidates = {
        str(entry.get("name"))
        for entry in catalog
        if entry.get("name")
        and isinstance(entry.get(fields["parent"]), dict)
        and str(entry[fields["parent"]].get("id")) == parent_id
    }
    return sorted(candidates)


def _display_id(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def distribution_table(distribution: TypeDistribution) -> dict[str, Any]:
    noun = (
        "физических объектов"
        if distribution.kind == ScenarioEntityKind.PHYSICAL_OBJECT
        else "сервисов"
    )
    return {
        "name": f"scenario_{distribution.kind.value}_types",
        "title": f"Распределение {noun} по типам",
        "columns": [
            {"key": "type_id", "label": "ID типа"},
            {"key": "type_name", "label": "Тип"},
            {"key": "count", "label": "Количество"},
            {"key": "status", "label": "Статус"},
            {"key": "possible_types", "label": "Возможные варианты"},
        ],
        "rows": distribution.rows,
    }


def distribution_answer(scenario_id: int, distributions: list[TypeDistribution]) -> str:
    paragraphs = [f"Результаты для сценария {scenario_id}:"]
    for distribution in distributions:
        entity_phrase = (
            _plural_ru(
                distribution.total_unique,
                "уникальный физический объект",
                "уникальных физических объекта",
                "уникальных физических объектов",
            )
            if distribution.kind == ScenarioEntityKind.PHYSICAL_OBJECT
            else _plural_ru(
                distribution.total_unique,
                "уникальный сервис",
                "уникальных сервиса",
                "уникальных сервисов",
            )
        )
        paragraphs.append(
            f"Найдено {distribution.total_unique} {entity_phrase}. "
            f"В сценарии представлено {distribution.present_types} "
            f"{_plural_ru(distribution.present_types, 'тип', 'типа', 'типов')}. "
            "Полная разбивка представленных типов приведена в таблице."
        )
    return "\n\n".join(paragraphs)


def _plural_ru(value: int, one: str, few: str, many: str) -> str:
    last_two = value % 100
    if 11 <= last_two <= 14:
        return many
    last = value % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many
