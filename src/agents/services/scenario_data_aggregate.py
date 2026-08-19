"""Count what a tool returned instead of showing the model a sample of it.

The scenario-data agent used to hand the LLM a *truncated* view of every tool result — the
first eight rows and a "… ещё N" marker. For a question like "what objects are on the
territory and how many", that is unanswerable by construction: 924 physical objects arrived,
eight were visible, and the model correctly reported the rest as unknown.

Counting is not the model's job. This module reduces a list of records to exact per-field
distributions computed in Python, so the answer carries real numbers and the context stays
small regardless of how many rows came back.

Only *categorical* fields are counted. Identifiers are useless as a breakdown (924 objects
give 924 distinct ids) and are skipped, as are free-text and numeric measurement fields.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

#: Nested dicts are flattened to this depth, so ``physical_object_type.name`` is reached
#: while deep payloads stay out of the summary.
MAX_FLATTEN_DEPTH = 2
#: Fields kept in the breakdown, most-informative first.
MAX_FIELDS = 8
#: A field whose values are nearly all distinct is an identifier, not a category.
MAX_DISTINCT_RATIO = 0.5

_ENTITY_ID_FIELDS = ("physical_object_id", "service_id", "object_geometry_id", "id")

_TECHNICAL_IDENTIFIER_TOKEN = re.compile(
    r"(?<!\w)[\"']?(?:scenario_id|project_id|service_type_id|"
    r"physical_object_type_id|service_id|physical_object_id|id|ids)[\"']?"
    r"\s*[:=]\s*[^,;\s)\]}]+[,;]?\s*",
    flags=re.IGNORECASE,
)
_HUMAN_IDENTIFIER_TOKEN = re.compile(
    r"\b(?:(?:ID|идентификатор)\s+(?:сценария|проекта|типа(?:\s+сервиса|\s+"
    r"физического\s+объекта)?|сервиса|физического\s+объекта)|(?:scenario|project)\s+ID)"
    r"\s*[:=№#-]?\s*[\w-]+[,;]?\s*",
    flags=re.IGNORECASE,
)


def _is_technical_identifier_key(key: str) -> bool:
    tail = key.rsplit(".", 1)[-1].casefold()
    return tail in {"id", "ids", "scenario_id", "project_id"} or tail.endswith(
        ("_id", "_ids")
    )


def _public_value(value: Any) -> Any:
    """Remove internal Urban identifiers while retaining names and aggregates."""

    if isinstance(value, dict):
        return {
            str(key): _public_value(child)
            for key, child in value.items()
            if not _is_technical_identifier_key(str(key))
        }
    if isinstance(value, list):
        return [_public_value(child) for child in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _public_value(json.loads(stripped))
            except (TypeError, ValueError):
                pass
        return _TECHNICAL_IDENTIFIER_TOKEN.sub("", value).strip()
    return value


def sanitize_public_answer(answer: str) -> str:
    """Remove explicit implementation identifiers from user-visible prose."""

    lines = []
    for line in answer.splitlines():
        cleaned = _TECHNICAL_IDENTIFIER_TOKEN.sub("", line)
        cleaned = _HUMAN_IDENTIFIER_TOKEN.sub("", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
        if re.fullmatch(r"(?:метаданные|metadata)\s*[:.]?", cleaned, re.IGNORECASE):
            continue
        lines.append(cleaned)
    return "\n".join(lines).strip()


def extract_records(result: Any) -> list[dict[str, Any]] | None:
    """The list of record dicts inside a tool result, if that is what it is."""

    rows = result
    if isinstance(rows, dict):
        for key in ("result", "results", "items", "features", "data"):
            if isinstance(rows.get(key), list):
                rows = rows[key]
                break
    if not isinstance(rows, list) or not rows:
        return None
    records = [row for row in rows if isinstance(row, dict)]
    if len(records) != len(rows):
        return None
    # GeoJSON features carry their attributes one level down.
    if all(set(r) >= {"type", "properties"} for r in records):
        props = [r.get("properties") or {} for r in records]
        if all(isinstance(p, dict) for p in props):
            return props
    return records


def _flatten(
    record: dict[str, Any], depth: int = 0, prefix: str = ""
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if key in {"geometry", "coordinates"}:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict) and depth < MAX_FLATTEN_DEPTH:
            flat.update(_flatten(value, depth + 1, f"{path}."))
        elif isinstance(value, (str, bool)) or value is None:
            flat[path] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            flat[path] = value
    return flat


def _is_reference(path: str) -> bool:
    """True for a key that points at another entity (``service_type_id``)."""

    tail = path.rsplit(".", 1)[-1].lower()
    return tail == "id" or tail.endswith("_id") or tail.endswith("_ids")


#: Fields that read as a category rather than a measurement. A breakdown by these is what a
#: "what is there and how much" question is actually asking for, so they outrank everything
#: else — ranking purely by "fewest distinct values" let near-constant booleans crowd out
#: ``service_type.name`` and produced a breakdown that answered nothing.
_CATEGORY_HINTS = ("name", "type", "category", "status", "function", "kind", "level")


def _is_category(path: str) -> bool:
    lowered = path.lower()
    return any(hint in lowered for hint in _CATEGORY_HINTS)


def unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate entity rows by their stable Urban API identifier when available."""

    id_field = next(
        (
            field
            for field in _ENTITY_ID_FIELDS
            if any(record.get(field) is not None for record in records)
        ),
        None,
    )
    if id_field is None:
        return records
    seen: set[str] = set()
    unique = []
    for record in records:
        identifier = record.get(id_field)
        if identifier is None:
            # Do not collapse unrelated rows merely because one row lacks the dominant ID.
            unique.append(record)
            continue
        key = str(identifier)
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def unresolved_references(aggregate: dict[str, Any] | None) -> list[str]:
    """Reference fields in ``aggregate`` that have no sibling name to read them by.

    ``service_type_id`` counted 12 ways is a real distribution, but "12 services of type 3"
    is not an answer. When the sibling ``service_type.name`` is absent, the id has to be
    resolved through a dictionary tool — this is what tells the planner to make that call
    instead of stopping.
    """

    if not aggregate:
        return []
    fields = set(aggregate.get("breakdown") or {})
    pending = []
    for path in fields:
        if not _is_reference(path):
            continue
        stem = path.rsplit(".", 1)[-1].removesuffix("_ids").removesuffix("_id")
        if not stem or stem == "id":
            continue
        named = {f"{stem}.name", f"{stem}_name", "name"}
        if not any(candidate in field for field in fields for candidate in named):
            pending.append(path)
    return sorted(pending)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Exact per-field value counts for ``records``.

    Returns ``None`` when nothing categorical is present, so the caller can fall back to the
    ordinary sample-based summary rather than emitting an empty breakdown.
    """

    if not records:
        return None
    records = unique_records(records)
    total = len(records)
    counters: dict[str, Counter] = {}
    for record in records:
        for path, value in _flatten(record).items():
            # Reference keys are deliberately kept: a low-cardinality `service_type_id` is a
            # real distribution and the join key that tells the planner which dictionary to
            # fetch. True row identifiers are dropped below by the distinct == total rule.
            counters.setdefault(path, Counter())[
                "—" if value is None or value == "" else str(value)
            ] += 1

    scored: list[tuple[int, int, str, Counter]] = []
    for path, counter in counters.items():
        distinct = len(counter)
        if distinct <= 1 and total > 1 and counter.most_common(1)[0][0] == "—":
            continue  # a column that is empty everywhere says nothing
        if distinct == total and total > 1:
            # A type can legitimately occur once per entity. Keep nested type/category
            # fields, but drop row IDs and free-text entity names.
            tail = path.rsplit(".", 1)[-1].lower()
            nested_category = "." in path and _is_category(path)
            if (_is_reference(path) and not _is_category(path)) or (
                tail == "name" and not nested_category
            ):
                continue
        if not _is_category(path) and distinct > max(
            1, int(total * MAX_DISTINCT_RATIO)
        ):
            # High-cardinality *and* not category-shaped: a measurement or a note. A
            # category keeps its place even when rows are few and types many, which is
            # exactly the case a ratio rule alone gets wrong.
            continue
        scored.append((0 if _is_category(path) else 1, distinct, path, counter))

    if not scored:
        return None

    # Categories first, then the more informative field (fewest distinct) within each rank.
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    breakdown: dict[str, Any] = {}
    for _rank, distinct, path, counter in scored[:MAX_FIELDS]:
        # Type-count questions require the full project catalogue; silently folding the
        # long tail into "other" produces an answer that cannot name every available type.
        top = counter.most_common()
        entry: dict[str, Any] = {
            "distinct_values": distinct,
            "counts": {value: count for value, count in top},
        }
        breakdown[path] = entry

    return {"total_records": total, "breakdown": breakdown}


def aggregate_result(result: Any) -> dict[str, Any] | None:
    """``aggregate_records`` applied to whatever record list a tool result contains."""

    records = extract_records(result)
    if records is None:
        return None
    return aggregate_records(records)


def answer_records(result: Any, *, limit: int = 100) -> list[dict[str, Any]] | None:
    """Keep a bounded complete id/name catalogue when it fits in model context."""

    records = extract_records(result)
    if records is None or len(records) > limit:
        return None
    selected: list[dict[str, Any]] = []
    for record in records:
        row = {
            str(key): value
            for key, value in record.items()
            if key in {"id", "name", "title", "code"}
            or key.endswith("_id")
            or key.endswith("_name")
        }
        if row:
            selected.append(row)
    return selected or None


def bounded_observation_context(
    observations: list[dict[str, Any]], *, max_chars: int
) -> str:
    """Serialize recent evidence first without ever cutting JSON mid-document."""

    compacted: list[dict[str, Any]] = []
    for observation in reversed(observations):
        item = {
            key: observation[key]
            for key in (
                "context",
                "tool",
                "arguments",
                "layer_count",
                "table_count",
                "summary",
                "satisfies",
                "aggregate",
                "answer_records",
                "unresolved_references",
            )
            if key in observation
        }
        if isinstance(item.get("summary"), str):
            item["summary"] = item["summary"][:3000]
        mapping = observation.get("mapping")
        if isinstance(mapping, dict):
            item["mapping"] = {
                key: mapping[key]
                for key in (
                    "domain",
                    "direction",
                    "requested_values",
                    "source_tool",
                    "matches",
                )
                if key in mapping
            }
            if isinstance(item["mapping"].get("matches"), list):
                item["mapping"]["matches"] = item["mapping"]["matches"][:50]
        artifact = observation.get("artifact")
        if isinstance(artifact, dict):
            item["artifact"] = {
                key: artifact[key]
                for key in ("rows", "columns", "format", "source_tool")
                if key in artifact
            }
        candidate = [*compacted, item]
        payload = json.dumps(
            {"order": "most_recent_first", "observations": candidate},
            ensure_ascii=False,
            default=str,
        )
        if len(payload) > max_chars:
            continue
        compacted = candidate
    return json.dumps(
        {"order": "most_recent_first", "observations": compacted},
        ensure_ascii=False,
        default=str,
    )


def bounded_public_observation_context(
    observations: list[dict[str, Any]], *, max_chars: int
) -> str:
    """Build answer evidence without exposing internal ids or tool arguments.

    The planner receives the full observation context because it must bind verified ids to
    Urban MCP calls. The answer model has a different contract: it needs names, counts and
    layer/table facts, but project/scenario/type ids are implementation metadata for it.
    """

    public_observations: list[dict[str, Any]] = []
    for observation in observations:
        item = {
            key: _public_value(observation[key])
            for key in (
                "context",
                "layer_count",
                "table_count",
                "summary",
                "aggregate",
                "answer_records",
            )
            if key in observation
        }
        mapping = observation.get("mapping")
        if isinstance(mapping, dict):
            matches = mapping.get("matches") or []
            names = [
                str(match.get("name") or match.get("title") or "").strip()
                for match in matches
                if isinstance(match, dict) and (match.get("name") or match.get("title"))
            ]
            if names:
                existing = item.get("answer_records")
                public_records = existing if isinstance(existing, list) else []
                item["answer_records"] = [
                    *public_records,
                    *({"name": name} for name in names[:50]),
                ]
        public_observations.append(item)
    return bounded_observation_context(public_observations, max_chars=max_chars)
