"""Deterministic mapping-tool selection over the runtime Urban MCP catalogue."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data_aggregate import extract_records
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    DataRequirement,
    MappingDirection,
    MappingNeed,
)


@dataclass(frozen=True)
class MappingCall:
    requirement_id: str
    need: MappingNeed
    tool: UrbanMcpTool
    arguments: dict[str, Any]
    intent_text: str


class UrbanMappingResolver:
    """Choose only dictionary calls that can be populated without guessed values."""

    MAX_BOOTSTRAP_CALLS = 3

    def plan_calls(
        self,
        acquisition: AcquisitionPlan,
        tools: list[UrbanMcpTool],
        scenario_id: int | None,
        project_id: int | None = None,
        known_mappings: list[dict[str, Any]] | None = None,
    ) -> list[MappingCall]:
        needs = [
            (requirement, need)
            for requirement in acquisition.requirements
            for need in requirement.mapping_needs
        ]
        calls: list[MappingCall] = []
        seen: set[tuple[str, str, str]] = set()
        for requirement, need in needs:
            if mapping_need_is_resolved(need, known_mappings or []):
                continue
            for lookup_need in _lookup_needs(need):
                for tool in self._rank(lookup_need, tools):
                    arguments = self._arguments(
                        lookup_need, tool, scenario_id, project_id
                    )
                    if arguments is None:
                        continue
                    key = (tool.group, tool.name, repr(sorted(arguments.items())))
                    if key in seen:
                        break
                    calls.append(
                        MappingCall(
                            requirement_id=requirement.requirement_id,
                            need=lookup_need,
                            tool=tool,
                            arguments=arguments,
                            intent_text=" ".join(
                                (acquisition.objective, requirement.description)
                            ),
                        )
                    )
                    seen.add(key)
                    break
                if len(calls) >= self.MAX_BOOTSTRAP_CALLS:
                    break
            if len(calls) >= self.MAX_BOOTSTRAP_CALLS:
                break
        return calls

    @staticmethod
    def _rank(need: MappingNeed, tools: list[UrbanMcpTool]) -> list[UrbanMcpTool]:
        normalized_domain = re.sub(r"[^a-zа-яё0-9]+", " ", need.domain.lower())
        domain_words = set(normalized_domain.split())
        compact_domain = normalized_domain.replace(" ", "")
        tokens = {
            token
            for token in re.findall(r"[a-zа-яё0-9]+", normalized_domain)
            if len(token) > 2
        }

        preferred_names: set[str] = set()
        required_subject_tokens: set[str] = set()
        if (
            "service_type" in need.domain.lower()
            or {"service", "type"}.issubset(domain_words)
            or "servicetype" in compact_domain
        ):
            preferred_names = {"getservicetypes", "getscenarioservicetypes"}
            required_subject_tokens = {"service", "type"}
        elif (
            "physical_object_type" in need.domain.lower()
            or {"physical", "object", "type"}.issubset(domain_words)
            or "physicalobjecttype" in compact_domain
        ):
            preferred_names = {
                "getphysicalobjecttypes",
                "getscenariophysicalobjecttypes",
            }
            required_subject_tokens = {"physical", "object", "type"}

        def is_relevant(tool: UrbanMcpTool) -> bool:
            name_and_title = re.sub(
                r"[^a-zа-яё0-9]+", " ", f"{tool.name} {tool.title}".lower()
            )
            normalized_name = re.sub(r"[^a-z0-9]+", "", tool.name.lower())
            if normalized_name in preferred_names:
                return True
            if required_subject_tokens:
                return all(token in name_and_title for token in required_subject_tokens)
            meaningful = tokens - {"type", "types", "data", "object", "objects"}
            return bool(meaningful) and any(
                token in name_and_title for token in meaningful
            )

        def score(tool: UrbanMcpTool) -> tuple[int, str]:
            text = f"{tool.name} {tool.title} {tool.description}".lower()
            normalized_name = re.sub(r"[^a-z0-9]+", "", tool.name.lower())
            value = sum(
                3 if token in tool.title.lower() else 1
                for token in tokens
                if token in text
            )
            if normalized_name in preferred_names:
                value += 100
                if tool.group == "dictionaries":
                    value += 20
            if tool.group == "dictionaries":
                value += 4
            if "type" in tool.name.lower() or "тип" in text:
                value += 1
            return (-value, tool.name)

        return sorted((tool for tool in tools if is_relevant(tool)), key=score)

    @staticmethod
    def _arguments(
        need: MappingNeed,
        tool: UrbanMcpTool,
        scenario_id: int | None,
        project_id: int | None = None,
    ) -> dict[str, Any] | None:
        schema = tool.input_schema or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        arguments: dict[str, Any] = {}
        if "scenario_id" in properties and scenario_id is not None:
            arguments["scenario_id"] = scenario_id
        if "project_id" in properties and project_id is not None:
            arguments["project_id"] = project_id

        values = list(need.values)
        if values:
            plural_candidates = (
                ("ids", "_ids", "identifiers")
                if need.direction == MappingDirection.ID_TO_NAME
                else ("names", "_names")
            )
            singular_candidates = (
                ("id", "_id", "identifier")
                if need.direction == MappingDirection.ID_TO_NAME
                else ("name", "_name")
            )
            for name, prop in properties.items():
                lowered = name.lower()
                prop_type = (prop or {}).get("type") if isinstance(prop, dict) else None
                if (
                    any(marker in lowered for marker in plural_candidates)
                    and prop_type == "array"
                ):
                    arguments[name] = values
                    break
                if len(values) == 1 and any(
                    lowered == marker or lowered.endswith(marker)
                    for marker in singular_candidates
                ):
                    arguments[name] = values[0]
                    break

        if required - set(arguments):
            return None
        return arguments


_DOMAIN_ALIASES = {
    "service_types": "service_type",
    "service_type": "service_type",
    "physical_object_types": "physical_object_type",
    "physical_object_type": "physical_object_type",
}
_TYPE_DOMAINS = ("physical_object_type", "service_type")
_NON_TYPE_MAPPING_DOMAINS = {
    "functional_zone",
    "functional_zone_type",
    "indicator",
    "project",
    "scenario",
    "soc_group",
    "social_group",
    "territory",
}

_WORD_ENDINGS = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ские",
    "ский",
    "ская",
    "ское",
    "ие",
    "ые",
    "ий",
    "ый",
    "ая",
    "ое",
    "ой",
    "ов",
    "ев",
    "ам",
    "ям",
    "ах",
    "ях",
    "ы",
    "и",
    "а",
    "я",
    "у",
    "ю",
    "е",
    "о",
)


def context_mapping_snapshots(context: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert published chat mappings into the planner's domain-aware snapshots."""

    structured = ((context or {}).get("content") or {}).get("structured") or {}
    raw_mappings = structured.get("mappings") or []
    if isinstance(raw_mappings, dict):
        raw_mappings = [raw_mappings]
    elif not isinstance(raw_mappings, list):
        raw_mappings = []
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for item in raw_mappings:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("domain"), str) and isinstance(
            item.get("matches"), list
        ):
            domain = _canonical_domain(item["domain"])
            for match in item["matches"]:
                if isinstance(match, dict):
                    _append_mapping(
                        by_domain, domain, match.get("id"), match.get("name")
                    )
            continue
        for raw_domain, values in item.items():
            if not isinstance(values, dict):
                continue
            domain = _canonical_domain(str(raw_domain))
            for left, right in values.items():
                if _looks_like_id(left):
                    _append_mapping(by_domain, domain, left, right)
                elif _looks_like_id(right):
                    _append_mapping(by_domain, domain, right, left)

    for message in (context or {}).get("tail") or []:
        if not isinstance(message, dict):
            continue
        for part in message.get("parts") or []:
            if not isinstance(part, dict) or part.get("kind") != "table":
                continue
            payload = part.get("payload") or {}
            if not isinstance(payload, dict):
                continue
            _append_table_mappings(by_domain, payload)

    return [
        {
            "domain": domain,
            "direction": MappingDirection.NAME_TO_ID.value,
            "requested_values": [],
            "source_tool": "chat_context",
            "matches": matches,
        }
        for domain, matches in sorted(by_domain.items())
        if matches
    ]


def enrich_acquisition_mappings(
    acquisition: AcquisitionPlan,
    user_query: str,
    known_mappings: list[dict[str, Any]],
) -> AcquisitionPlan:
    """Restore mapping needs that the model dropped after seeing a known numeric id."""

    requirements: list[DataRequirement] = []
    changed = False
    for requirement in acquisition.requirements:
        haystack = " ".join(
            (user_query, acquisition.objective, requirement.description)
        )
        needs = [
            need
            for need in requirement.mapping_needs
            if _canonical_domain(need.domain) not in {"scenario", "project"}
        ]
        if needs != requirement.mapping_needs:
            # These ids are selected by the UI and bound by the service. Treating
            # them as name/id dictionaries sends the resolver to unrelated tools.
            changed = True
        normalized_needs: list[MappingNeed] = []
        for need in needs:
            domain = _canonical_domain(need.domain)
            if (
                need.direction == MappingDirection.NAME_TO_ID
                and domain not in _TYPE_DOMAINS
                and domain not in _NON_TYPE_MAPPING_DOMAINS
                and not need.values
            ):
                # Some models put the requested class itself into domain
                # (domain="school") and omit values. Treat it as an unproven type;
                # _lookup_needs will verify it against both Urban type ontologies.
                normalized_needs.append(
                    need.model_copy(
                        update={
                            "domain": "physical_object_type",
                            "values": [need.domain],
                        }
                    )
                )
                changed = True
            else:
                normalized_needs.append(need)
        needs = normalized_needs
        if not needs:
            inferred_names = _quoted_type_names(haystack)
            if inferred_names:
                # The domain is deliberately provisional. _lookup_needs expands a
                # named type across both Urban ontologies; the returned mappings then
                # select the verified service/physical-object domain.
                needs.append(
                    MappingNeed(
                        domain="physical_object_type",
                        direction=MappingDirection.NAME_TO_ID,
                        values=inferred_names,
                    )
                )
                changed = True
        mentioned_by_domain: dict[str, list[str]] = {}
        for snapshot in known_mappings:
            domain = _canonical_domain(str(snapshot.get("domain") or ""))
            mentioned = [
                str(match.get("name"))
                for match in snapshot.get("matches") or []
                if isinstance(match, dict)
                and match.get("name")
                and _name_is_mentioned(haystack, str(match["name"]))
            ]
            if mentioned:
                mentioned_by_domain.setdefault(domain, []).extend(mentioned)
        mentioned_by_domain = {
            domain: names
            for domain, names in mentioned_by_domain.items()
            if domain in _TYPE_DOMAINS and names
        }
        explicit_domain = _explicit_type_domain(haystack)
        preferred_domain = explicit_domain
        if preferred_domain is None and len(mentioned_by_domain) == 1:
            preferred_domain = next(iter(mentioned_by_domain))
        if preferred_domain is not None:
            filtered = [
                need
                for need in needs
                if _canonical_domain(need.domain) not in _TYPE_DOMAINS
                or _canonical_domain(need.domain) == preferred_domain
            ]
            if filtered != needs:
                needs = filtered
                changed = True
        for snapshot in known_mappings:
            domain = _canonical_domain(str(snapshot.get("domain") or ""))
            mentioned = [
                str(match.get("name"))
                for match in snapshot.get("matches") or []
                if isinstance(match, dict)
                and match.get("name")
                and _name_is_mentioned(haystack, str(match["name"]))
            ]
            if not mentioned:
                continue
            existing = next(
                (
                    need
                    for need in needs
                    if _canonical_domain(need.domain) == domain
                    and need.direction == MappingDirection.NAME_TO_ID
                ),
                None,
            )
            if existing is None:
                needs.append(
                    MappingNeed(
                        domain=domain,
                        direction=MappingDirection.NAME_TO_ID,
                        values=list(dict.fromkeys(mentioned)),
                    )
                )
                changed = True
            elif not existing.values:
                replacement = existing.model_copy(
                    update={"values": list(dict.fromkeys(mentioned))}
                )
                needs[needs.index(existing)] = replacement
                changed = True
        requirements.append(
            requirement.model_copy(update={"mapping_needs": needs})
            if needs != requirement.mapping_needs
            else requirement
        )
    if not changed:
        return acquisition
    return acquisition.model_copy(update={"requirements": requirements})


def mapping_need_is_resolved(
    need: MappingNeed, known_mappings: list[dict[str, Any]]
) -> bool:
    """Return true only when every explicitly requested value has evidence."""

    if not need.values:
        return False
    domain = _canonical_domain(need.domain)
    matches = [
        match
        for snapshot in known_mappings
        if _canonical_domain(str(snapshot.get("domain") or "")) == domain
        for match in snapshot.get("matches") or []
        if isinstance(match, dict)
    ]
    if need.direction == MappingDirection.NAME_TO_ID:
        return all(
            any(
                _same_name(str(value), str(match.get("name") or ""))
                for match in matches
            )
            for value in need.values
        )
    return all(
        any(str(value) == str(match.get("id")) for match in matches)
        for value in need.values
    )


def _quoted_type_names(text: str) -> list[str]:
    """Extract explicit human-facing type names the planner forgot to map."""

    if not any(marker in text.casefold() for marker in ("тип", "type")):
        return []
    return list(
        dict.fromkeys(
            match.strip()
            for match in re.findall(r"[«\"]([^»\"]{2,80})[»\"]", text)
            if match.strip()
        )
    )


def _lookup_needs(need: MappingNeed) -> list[MappingNeed]:
    """Resolve a named urban type against both ontologies when its domain is unproven."""

    domain = _canonical_domain(need.domain)
    if need.direction != MappingDirection.NAME_TO_ID or domain not in _TYPE_DOMAINS:
        return [need]
    return [
        need.model_copy(update={"domain": candidate, "values": []})
        for candidate in _TYPE_DOMAINS
    ]


def bind_mapping_arguments(
    tool: UrbanMcpTool,
    arguments: dict[str, Any],
    known_mappings: list[dict[str, Any]],
    intent_text: str,
) -> dict[str, Any]:
    """Fill domain-specific id arguments from verified mappings, never from guesses."""

    prepared = dict(arguments)
    properties = (tool.input_schema or {}).get("properties") or {}
    for name, prop in properties.items():
        domain = _domain_for_argument(name)
        if domain is None:
            continue
        matches = [
            match
            for snapshot in known_mappings
            if _canonical_domain(str(snapshot.get("domain") or "")) == domain
            for match in snapshot.get("matches") or []
            if isinstance(match, dict)
            and match.get("id") is not None
            and match.get("name")
            and _name_is_mentioned(intent_text, str(match["name"]))
        ]
        if not matches:
            continue
        ids = list(dict.fromkeys(match["id"] for match in matches))
        prop_type = prop.get("type") if isinstance(prop, dict) else None
        if prop_type == "array" or name.endswith("_ids"):
            prepared[name] = ids
        elif len(ids) == 1:
            prepared[name] = ids[0]
    return prepared


def _canonical_domain(value: str) -> str:
    normalized = re.sub(r"[^a-zа-яё0-9]+", "_", value.casefold()).strip("_")
    return _DOMAIN_ALIASES.get(normalized, normalized.removesuffix("s"))


def _domain_for_argument(name: str) -> str | None:
    lowered = name.casefold()
    if lowered in {"scenario_id", "project_id", "id", "ids"}:
        return None
    if lowered.endswith("_ids"):
        return _canonical_domain(lowered[:-4])
    if lowered.endswith("_id"):
        return _canonical_domain(lowered[:-3])
    return None


def _explicit_type_domain(text: str) -> str | None:
    normalized = text.casefold()
    service_markers = ("сервис", "услуг", "обеспеч", "service")
    physical_markers = (
        "физическ",
        "здани",
        "сооруж",
        "physical object",
        "physical_object",
    )
    service = any(marker in normalized for marker in service_markers)
    physical = any(marker in normalized for marker in physical_markers)
    if service == physical:
        return None
    return "service_type" if service else "physical_object_type"


def _looks_like_id(value: Any) -> bool:
    return isinstance(value, int) or (isinstance(value, str) and value.isdigit())


def _append_mapping(
    target: dict[str, list[dict[str, Any]]], domain: str, raw_id: Any, raw_name: Any
) -> None:
    if not domain or raw_id is None or raw_name is None:
        return
    identifier = int(raw_id) if isinstance(raw_id, str) and raw_id.isdigit() else raw_id
    candidate = {"id": identifier, "name": str(raw_name)}
    bucket = target.setdefault(domain, [])
    if candidate not in bucket:
        bucket.append(candidate)


def _append_table_mappings(
    target: dict[str, list[dict[str, Any]]], payload: dict[str, Any]
) -> None:
    rows = payload.get("rows") or []
    if not isinstance(rows, list):
        return
    raw_name = str(payload.get("name") or "")
    domain = (
        _canonical_domain(raw_name.removeprefix("mapping_"))
        if raw_name.startswith("mapping_")
        else ""
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        id_key = (
            "id"
            if domain and "id" in row
            else next(
                (
                    str(key)
                    for key in row
                    if str(key).endswith("_type_id") and _looks_like_id(row.get(key))
                ),
                "",
            )
        )
        if not domain and id_key:
            domain = _canonical_domain(id_key.removesuffix("_id"))
        name_key = next(
            (key for key in ("name", "title") if row.get(key) is not None), None
        )
        if domain and id_key and name_key:
            _append_mapping(target, domain, row.get(id_key), row.get(name_key))


def _tokens(value: str) -> set[str]:
    return {
        _stem(token)
        for token in re.findall(r"[a-zа-яё0-9]+", value.casefold())
        if token
    }


def _stem(token: str) -> str:
    for ending in _WORD_ENDINGS:
        if len(token) - len(ending) >= 3 and token.endswith(ending):
            return token[: -len(ending)]
    return token


def _name_is_mentioned(text: str, name: str) -> bool:
    name_tokens = _tokens(name)
    return bool(name_tokens) and name_tokens.issubset(_tokens(text))


def _same_name(left: str, right: str) -> bool:
    return _tokens(left) == _tokens(right)


def mapping_snapshot(call: MappingCall, result: Any) -> dict[str, Any]:
    """Bound mapping evidence for planner context; preserve its precise source."""

    records = extract_records(result)
    normalized: list[dict[str, Any]] = []
    requested = {str(value).casefold() for value in call.need.values}
    for record in records or []:
        id_keys = [key for key in record if key == "id" or key.endswith("_id")]
        name_keys = [
            key for key in record if key in {"name", "title"} or key.endswith("_name")
        ]
        if not id_keys and not name_keys:
            continue
        candidate = {
            "id": record.get(id_keys[0]) if id_keys else None,
            "name": record.get(name_keys[0]) if name_keys else None,
        }
        candidate_values = {str(value).casefold() for value in candidate.values()}
        requested_match = any(
            left == right or left in right or right in left
            for left in requested
            for right in candidate_values
        )
        intent_match = bool(candidate.get("name")) and _name_is_mentioned(
            call.intent_text, str(candidate["name"])
        )
        if requested and not requested_match and not intent_match:
            continue
        normalized.append(candidate)
        if len(normalized) >= 100:
            break

    return {
        "domain": call.need.domain,
        "direction": call.need.direction.value,
        "requested_values": call.need.values[:50],
        "source_tool": f"{call.tool.group}.{call.tool.name}",
        "matches": normalized,
        "result": _bound(result),
    }


def _bound(value: Any, depth: int = 0) -> Any:
    if depth >= 3:
        return "…"
    if isinstance(value, dict):
        return {
            str(key): _bound(item, depth + 1) for key, item in list(value.items())[:20]
        }
    if isinstance(value, list):
        return [_bound(item, depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return value[:500]
    return value
