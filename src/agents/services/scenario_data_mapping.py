"""Deterministic mapping-tool selection over the runtime Urban MCP catalogue."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.mcp_clients.urban_mcp_client import UrbanMcpTool
from src.agents.services.scenario_data_aggregate import extract_records
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    MappingDirection,
    MappingNeed,
)


@dataclass(frozen=True)
class MappingCall:
    requirement_id: str
    need: MappingNeed
    tool: UrbanMcpTool
    arguments: dict[str, Any]


class UrbanMappingResolver:
    """Choose only dictionary calls that can be populated without guessed values."""

    MAX_BOOTSTRAP_CALLS = 3

    def plan_calls(
        self,
        acquisition: AcquisitionPlan,
        tools: list[UrbanMcpTool],
        scenario_id: int | None,
    ) -> list[MappingCall]:
        needs = [
            (requirement.requirement_id, need)
            for requirement in acquisition.requirements
            for need in requirement.mapping_needs
        ]
        calls: list[MappingCall] = []
        seen: set[tuple[str, str, str]] = set()
        for requirement_id, need in needs:
            for tool in self._rank(need, tools):
                arguments = self._arguments(need, tool, scenario_id)
                if arguments is None:
                    continue
                key = (tool.group, tool.name, repr(sorted(arguments.items())))
                if key in seen:
                    break
                calls.append(
                    MappingCall(
                        requirement_id=requirement_id,
                        need=need,
                        tool=tool,
                        arguments=arguments,
                    )
                )
                seen.add(key)
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
        if (
            "service_type" in need.domain.lower()
            or {"service", "type"}.issubset(domain_words)
            or "servicetype" in compact_domain
        ):
            preferred_names = {"getservicetypes", "getscenarioservicetypes"}
        elif (
            "physical_object_type" in need.domain.lower()
            or {"physical", "object", "type"}.issubset(domain_words)
            or "physicalobjecttype" in compact_domain
        ):
            preferred_names = {
                "getphysicalobjecttypes",
                "getscenariophysicalobjecttypes",
            }

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

        return sorted(tools, key=score)

    @staticmethod
    def _arguments(
        need: MappingNeed, tool: UrbanMcpTool, scenario_id: int | None
    ) -> dict[str, Any] | None:
        schema = tool.input_schema or {}
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        arguments: dict[str, Any] = {}
        if "scenario_id" in properties and scenario_id is not None:
            arguments["scenario_id"] = scenario_id

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
        if requested and not any(
            left == right or left in right or right in left
            for left in requested
            for right in candidate_values
        ):
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
