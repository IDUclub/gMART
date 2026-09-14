"""LLM-assisted resolution of named Urban object and service types."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from loguru import logger
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from src.agents.services.restriction.restriction_catalog import strip_json_fence
from src.agents.services.scenario_data.scenario_data_aggregate import extract_records
from src.agents.services.scenario_data.scenario_data_mapping import (
    MappingCall,
    _canonical_domain,
    mapping_need_is_resolved,
)
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    MappingDirection,
    MappingNeed,
)

TYPE_DOMAINS = ("physical_object_type", "service_type")
MAX_PATTERN_LENGTH = 160
MAX_MAPPING_CANDIDATES = 120
MAX_CANDIDATES_PER_REQUEST = 40
MAPPING_LLM_RETRIES = 2

_FALLBACK_SUFFIXES = (
    "иями",
    "ами",
    "ями",
    "ого",
    "ему",
    "ому",
    "ыми",
    "ими",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ий",
    "ый",
    "ой",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)


class TypeMappingRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    requirement_id: str
    requested_value: str


class TypeSearchPattern(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    requested_value: str
    pattern: str = Field(min_length=1, max_length=MAX_PATTERN_LENGTH)

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        """Allow useful regexes while rejecting constructs prone to unsafe execution."""

        forbidden = ("(?=", "(?!", "(?<=", "(?<!", "(?P", "(?>", "(?#")
        if any(marker in value for marker in forbidden):
            raise ValueError(
                "lookarounds, named groups and atomic groups are not allowed"
            )
        if re.search(r"\\[1-9]", value):
            raise ValueError("regex backreferences are not allowed")
        if re.search(r"\([^)]*[*+]\s*\)[*+]", value):
            raise ValueError("nested quantified groups are not allowed")
        try:
            re.compile(value, flags=re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return value


class TypeSearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    patterns: list[TypeSearchPattern] = Field(min_length=1, max_length=30)


def _fallback_search_pattern(value: str) -> str:
    """Build a conservative, regex-safe token pattern without calling the LLM."""

    tokens = re.findall(r"[0-9a-zа-яё]+", value.casefold())
    stems = []
    for token in tokens:
        stem = token
        for suffix in _FALLBACK_SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= 3:
                stem = stem[: -len(suffix)]
                break
        stems.append(re.escape(stem))
    return ".*".join(stems) or re.escape(value.strip()) or r"$^"


class TypeMappingCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    requirement_id: str
    requested_value: str
    domain: Literal["physical_object_type", "service_type"]
    type_id: str | int
    name: str
    source_tool: str

    def prompt_entry(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "requirement_id": self.requirement_id,
            "requested_value": self.requested_value,
            "domain": self.domain,
            "name": self.name,
        }


class TypeCandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_ids: list[str] = Field(default_factory=list, max_length=60)
    reason: str = Field(default="", max_length=1000)


class TypeCandidateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    accepted: bool
    reason: str = Field(default="", max_length=500)


class TypeSelectionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assessments: list[TypeCandidateAssessment] = Field(max_length=60)


@dataclass(frozen=True)
class TypeMappingResolution:
    accepted: list[TypeMappingCandidate]
    missing_values: list[str]
    assessment: TypeSelectionAssessment | None = None

    @property
    def complete(self) -> bool:
        return bool(self.accepted) and not self.missing_values


def pending_type_mapping_requests(
    acquisition: AcquisitionPlan,
    known_mappings: list[dict[str, Any]],
) -> list[TypeMappingRequest]:
    """Return each unresolved named type independently, preserving requirement ownership."""

    requests: list[TypeMappingRequest] = []
    seen: set[tuple[str, str]] = set()
    for requirement in acquisition.requirements:
        for need in requirement.mapping_needs:
            if (
                need.direction != MappingDirection.NAME_TO_ID
                or _canonical_domain(need.domain) not in TYPE_DOMAINS
                or mapping_need_is_resolved(need, known_mappings)
            ):
                continue
            for value in need.values:
                key = (requirement.requirement_id, str(value))
                if key in seen:
                    continue
                requests.append(
                    TypeMappingRequest(
                        requirement_id=requirement.requirement_id,
                        requested_value=str(value),
                    )
                )
                seen.add(key)
    return requests


def collect_type_mapping_candidates(
    search_plan: TypeSearchPlan,
    results: list[tuple[MappingCall, Any]],
) -> list[TypeMappingCandidate]:
    """Run generated regexes locally over complete type dictionaries from both domains."""

    candidates: list[TypeMappingCandidate] = []
    seen: set[tuple[str, str, str, str]] = set()
    for pattern in search_plan.patterns:
        matched_for_request = 0
        for call, result in results:
            domain = _canonical_domain(call.need.domain)
            if domain not in TYPE_DOMAINS:
                continue
            for record in extract_records(result) or []:
                identifier, name = _type_record(record, domain)
                if identifier is None or not name:
                    continue
                if re.search(pattern.pattern, name, flags=re.IGNORECASE) is None:
                    continue
                key = (
                    pattern.requirement_id,
                    pattern.requested_value,
                    domain,
                    str(identifier),
                )
                if key in seen:
                    continue
                candidates.append(
                    TypeMappingCandidate(
                        candidate_id=f"candidate_{len(candidates) + 1}",
                        requirement_id=pattern.requirement_id,
                        requested_value=pattern.requested_value,
                        domain=domain,
                        type_id=identifier,
                        name=name,
                        source_tool=f"{call.tool.group}.{call.tool.name}",
                    )
                )
                seen.add(key)
                matched_for_request += 1
                if len(candidates) >= MAX_MAPPING_CANDIDATES:
                    return candidates
                if matched_for_request >= MAX_CANDIDATES_PER_REQUEST:
                    break
            if matched_for_request >= MAX_CANDIDATES_PER_REQUEST:
                break
    return candidates


def verified_mapping_snapshots(
    candidates: list[TypeMappingCandidate],
) -> list[dict[str, Any]]:
    """Convert accepted candidates into the evidence shape consumed by the plan builder."""

    snapshots: list[dict[str, Any]] = []
    for candidate in candidates:
        snapshots.append(
            {
                "domain": candidate.domain,
                "direction": MappingDirection.NAME_TO_ID.value,
                "requested_values": [candidate.requested_value],
                "source_tool": candidate.source_tool,
                "verified_by": "llm_semantic_assessment",
                "matches": [{"id": candidate.type_id, "name": candidate.name}],
            }
        )
    return snapshots


def apply_verified_type_mappings(
    acquisition: AcquisitionPlan,
    candidates: list[TypeMappingCandidate],
) -> AcquisitionPlan:
    """Replace provisional type needs with the names and domains selected by the mapper."""

    by_requirement: dict[str, dict[str, list[str]]] = {}
    for candidate in candidates:
        names = by_requirement.setdefault(candidate.requirement_id, {}).setdefault(
            candidate.domain, []
        )
        if candidate.name not in names:
            names.append(candidate.name)

    requirements = []
    for requirement in acquisition.requirements:
        selected = by_requirement.get(requirement.requirement_id)
        if not selected:
            requirements.append(requirement)
            continue
        selected_requested_values = {
            candidate.requested_value
            for candidate in candidates
            if candidate.requirement_id == requirement.requirement_id
        }
        retained: list[MappingNeed] = []
        for need in requirement.mapping_needs:
            if (
                need.direction != MappingDirection.NAME_TO_ID
                or _canonical_domain(need.domain) not in TYPE_DOMAINS
            ):
                retained.append(need)
                continue
            untouched_values = [
                value
                for value in need.values
                if str(value) not in selected_requested_values
            ]
            if untouched_values:
                retained.append(need.model_copy(update={"values": untouched_values}))
        retained.extend(
            MappingNeed(
                domain=domain,
                direction=MappingDirection.NAME_TO_ID,
                values=names,
            )
            for domain, names in selected.items()
        )
        requirements.append(requirement.model_copy(update={"mapping_needs": retained}))
    return acquisition.model_copy(update={"requirements": requirements})


class UrbanTypeMapper:
    """Generate search regexes, select candidates, then independently verify semantics."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def select_scenario_entities(
        self, model, query, candidates, *, requested_type=None
    ):
        from src.agents.services.scenario_data.scenario_data_selection import (
            ScenarioEntitySelection,
            exact_type_candidate,
            explicit_entity_domain,
            quoted_type,
            selection_messages,
            validate_selection,
        )

        domain = explicit_entity_domain(query)
        if domain:
            candidates = {
                key: value
                for key, value in candidates.items()
                if value["domain"] == domain
            }
        literal = quoted_type(query)
        exact = exact_type_candidate(literal or requested_type, candidates)
        if exact is not None:
            return ScenarioEntitySelection(candidate=exact)
        if literal:
            # A quoted catalogue name is a constraint, not permission to pick a synonym.
            return ScenarioEntitySelection(candidate=None)
        if requested_type:
            query = f"{query}\nЗапрошенный тип: {requested_type}"

        return await self._request_json(
            model,
            selection_messages(query, candidates),
            ScenarioEntitySelection,
            "scenario entity selection",
            post_validate=lambda selection: validate_selection(selection, candidates),
        )

    async def classify_scenario_entity_request(self, model, query):
        from src.agents.services.scenario_data.scenario_data_selection import (
            ScenarioEntityRequest,
            entity_request_messages,
            quoted_type,
        )

        literal = quoted_type(query)
        if literal:
            from src.agents.services.scenario_data.scenario_data_evaluator import (
                wants_layers,
            )

            return ScenarioEntityRequest(
                operation=(
                    "map"
                    if wants_layers(query)
                    else (
                        "count"
                        if re.search(r"сколько|количеств|посчита|подсчита", query, re.I)
                        else "list"
                    )
                ),
                requested_type=literal,
            )

        def validate(request):
            if (
                request.operation != "unsupported"
                and not (request.requested_type or "").strip()
            ):
                raise ValueError("A supported entity request must name its type")
            return request

        return await self._request_json(
            model,
            entity_request_messages(query),
            ScenarioEntityRequest,
            "scenario entity request",
            post_validate=validate,
        )

    async def build_search_plan(
        self,
        model: str,
        user_query: str,
        acquisition: AcquisitionPlan,
        requests: list[TypeMappingRequest],
    ) -> TypeSearchPlan:
        request_payload = [request.model_dump(mode="json") for request in requests]
        prompt = f"""Сформируй локальные регулярные выражения для поиска названий типов
городских сущностей. Для каждого элемента requests верни ровно один pattern. Паттерн
должен находить разумные лексические варианты, синонимы и уточнённые названия, которые
потом будут оценены отдельной моделью. Не добавляй флаги, lookaround, backreference и
исполняемый код. Не меняй requirement_id и requested_value.

Запрос пользователя: {user_query}
Цель: {acquisition.objective}
Requests: {json.dumps(request_payload, ensure_ascii=False)}"""

        expected = {
            (request.requirement_id, request.requested_value) for request in requests
        }

        def validate(plan: TypeSearchPlan) -> TypeSearchPlan:
            actual = {
                (item.requirement_id, item.requested_value) for item in plan.patterns
            }
            if actual != expected or len(plan.patterns) != len(expected):
                raise ValueError(
                    "search patterns must cover every requested value exactly once"
                )
            return plan

        try:
            return await self._request_json(
                model,
                [{"role": "system", "content": prompt}],
                TypeSearchPlan,
                "type search patterns",
                post_validate=validate,
            )
        except ValueError as exc:
            logger.warning(
                "Scenario-data type search planner failed; using safe local "
                f"patterns: {exc}"
            )
            return TypeSearchPlan(
                patterns=[
                    TypeSearchPattern(
                        requirement_id=request.requirement_id,
                        requested_value=request.requested_value,
                        pattern=_fallback_search_pattern(request.requested_value),
                    )
                    for request in requests
                ]
            )

    async def resolve_candidates(
        self,
        model: str,
        user_query: str,
        acquisition: AcquisitionPlan,
        requests: list[TypeMappingRequest],
        candidates: list[TypeMappingCandidate],
    ) -> TypeMappingResolution:
        if not candidates:
            return TypeMappingResolution(
                accepted=[],
                missing_values=[request.requested_value for request in requests],
            )

        selection = await self._select_candidates(
            model, user_query, acquisition, candidates
        )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        selected = [by_id[candidate_id] for candidate_id in selection.candidate_ids]
        if not selected:
            return TypeMappingResolution(
                accepted=[],
                missing_values=[request.requested_value for request in requests],
            )

        assessment = await self._assess_candidates(
            model, user_query, acquisition, selected
        )
        accepted_ids = {
            item.candidate_id for item in assessment.assessments if item.accepted
        }
        accepted = [
            candidate
            for candidate in selected
            if candidate.candidate_id in accepted_ids
        ]
        covered = {
            (candidate.requirement_id, candidate.requested_value)
            for candidate in accepted
        }
        missing = [
            request.requested_value
            for request in requests
            if (request.requirement_id, request.requested_value) not in covered
        ]
        return TypeMappingResolution(
            accepted=accepted,
            missing_values=list(dict.fromkeys(missing)),
            assessment=assessment,
        )

    async def _select_candidates(
        self,
        model: str,
        user_query: str,
        acquisition: AcquisitionPlan,
        candidates: list[TypeMappingCandidate],
    ) -> TypeCandidateSelection:
        candidate_payload = json.dumps(
            [item.prompt_entry() for item in candidates], ensure_ascii=False
        )
        prompt = f"""Выбери из candidates типы, наиболее точно соответствующие запросу.
Одновременно выбери правильный домен: physical_object_type или service_type. Возвращай
только candidate_id из списка, ничего не придумывай. Можно выбрать несколько кандидатов:
это обязательно, когда пользователь запросил несколько типов, и допустимо, когда одному
понятию действительно соответствуют несколько справочных типов. Не выбирай лишь похожие
по словам, но отличающиеся по смыслу записи. Пустой список означает, что подходящего типа
среди кандидатов нет.

Запрос пользователя: {user_query}
Цель: {acquisition.objective}
Candidates: {candidate_payload}"""
        available = {candidate.candidate_id for candidate in candidates}

        def validate(selection: TypeCandidateSelection) -> TypeCandidateSelection:
            if len(selection.candidate_ids) != len(set(selection.candidate_ids)):
                raise ValueError("candidate_ids must be unique")
            unknown = set(selection.candidate_ids) - available
            if unknown:
                raise ValueError(f"unknown candidate_ids: {sorted(unknown)}")
            return selection

        return await self._request_json(
            model,
            [{"role": "system", "content": prompt}],
            TypeCandidateSelection,
            "type candidate selection",
            post_validate=validate,
        )

    async def _assess_candidates(
        self,
        model: str,
        user_query: str,
        acquisition: AcquisitionPlan,
        selected: list[TypeMappingCandidate],
    ) -> TypeSelectionAssessment:
        selected_payload = json.dumps(
            [item.prompt_entry() for item in selected], ensure_ascii=False
        )
        prompt = f"""Независимо проверь каждый выбранный справочный тип. accepted=true только
если название и домен по смыслу соответствуют тому, что запросил пользователь. Совпадение
отдельных слов недостаточно. Не заменяй выбор и не добавляй кандидатов. Верни assessment
для каждого candidate_id ровно один раз.

Запрос пользователя: {user_query}
Цель: {acquisition.objective}
Выбранные типы: {selected_payload}"""
        expected = {candidate.candidate_id for candidate in selected}

        def validate(assessment: TypeSelectionAssessment) -> TypeSelectionAssessment:
            actual = {item.candidate_id for item in assessment.assessments}
            if actual != expected or len(assessment.assessments) != len(expected):
                raise ValueError(
                    "assessment must cover every selected candidate exactly once"
                )
            return assessment

        return await self._request_json(
            model,
            [{"role": "system", "content": prompt}],
            TypeSelectionAssessment,
            "type candidate assessment",
            post_validate=validate,
        )

    async def _request_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema,
        label: str,
        *,
        post_validate=None,
    ):
        error = ""
        for attempt in range(MAPPING_LLM_RETRIES + 1):
            call: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 1800 if attempt == 0 else 3000,
                },
            }
            if attempt < MAPPING_LLM_RETRIES:
                call["format"] = schema.model_json_schema()
            if attempt:
                call["reasoning_effort"] = "medium"
                call["messages"] = messages + [
                    {
                        "role": "user",
                        "content": (f"Исправь JSON: {error}. Верни только JSON."),
                    }
                ]
            response = await self.llm_client.chat(**call)
            raw = (response.get("message") or {}).get("content") or ""
            if not raw.strip():
                done_reason = response.get("done_reason") or "unknown"
                error = f"empty model response (done_reason={done_reason})"
                logger.warning(
                    f"Invalid scenario-data {label}, attempt {attempt + 1}: {error}"
                )
                continue
            try:
                parsed = schema.model_validate(json.loads(strip_json_fence(raw)))
                return post_validate(parsed) if post_validate else parsed
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
                logger.warning(
                    f"Invalid scenario-data {label}, attempt {attempt + 1}: {error}"
                )
        raise ValueError(f"invalid scenario-data {label} after retries: {error}")


def _type_record(record: dict[str, Any], domain: str) -> tuple[Any, str]:
    identifier = record.get(f"{domain}_id", record.get("id"))
    if identifier is None:
        identifier = next(
            (
                value
                for key, value in record.items()
                if key.endswith("_type_id") and value is not None
            ),
            None,
        )
    name = str(record.get("name") or record.get("title") or "").strip()
    return identifier, name
