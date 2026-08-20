"""Registry-dispatched execution of versioned CheckPlans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from src.agents.services.compliance_registry import (
    DEFAULT_COMPLIANCE_REGISTRY,
    TemplateRegistry,
    UnsupportedSchemaError,
    UnsupportedTemplateError,
)
from src.agents.services.compliance_requirements import ComplianceDataGate
from src.agents.services.restriction_tool_executor import RestrictionToolExecutor
from src.agents.services.service_entities.compliance import (
    CheckPlan,
    ComplianceResult,
    ComplianceSummary,
    DeclaredRequirements,
    VerificationCoverage,
)


@dataclass
class ComplianceExecution:
    plan: CheckPlan | None
    result: ComplianceResult
    tool_calls: list[dict[str, Any]]
    layers: dict[str, dict[str, Any]]
    timings_ms: dict[str, float]


class ComplianceTemplateExecutor:
    def __init__(
        self,
        registry: TemplateRegistry = DEFAULT_COMPLIANCE_REGISTRY,
        data_gate: ComplianceDataGate | None = None,
    ) -> None:
        self.registry = registry
        self.data_gate = data_gate or ComplianceDataGate()
        self.tools = RestrictionToolExecutor()

    async def execute(
        self,
        mcp_client,
        raw_plan: dict[str, Any],
        scenario_id: int,
    ) -> ComplianceExecution:
        timings_ms: dict[str, float] = {}
        validation_started = perf_counter()
        restriction_id = str(
            (raw_plan.get("source") or {}).get("restriction_id") or "unknown"
        )
        try:
            plan, params = self.registry.validate_plan(raw_plan)
        except (
            UnsupportedSchemaError,
            UnsupportedTemplateError,
            ValidationError,
        ) as exc:
            timings_ms["check_plan_validation"] = (
                perf_counter() - validation_started
            ) * 1000
            return ComplianceExecution(
                plan=None,
                result=self._outcome(
                    restriction_id=restriction_id,
                    template=str(raw_plan.get("template") or "unknown"),
                    template_version=int(raw_plan.get("template_version") or 1),
                    verification_status="unsupported",
                    missing=[str(exc)],
                    warnings=["check_plan_validation_failed"],
                ),
                tool_calls=[],
                layers={},
                timings_ms=timings_ms,
            )
        timings_ms["check_plan_validation"] = (
            perf_counter() - validation_started
        ) * 1000
        if plan.planner_status == "unsupported":
            return ComplianceExecution(
                plan=plan,
                result=self._outcome(
                    restriction_id=plan.source.restriction_id,
                    template=plan.template,
                    template_version=plan.template_version,
                    verification_status="unsupported",
                    missing=["planner_status:unsupported"],
                ),
                tool_calls=[],
                layers={},
                timings_ms=timings_ms,
            )

        resolution_started = perf_counter()
        effective = self.registry.effective_requirements(plan)
        requirements = DeclaredRequirements(
            layers=effective["layers"], attributes=effective["attributes"]
        )
        if effective["missing_registry_roles"]:
            timings_ms["requirements_resolution"] = (
                perf_counter() - resolution_started
            ) * 1000
            return ComplianceExecution(
                plan=plan,
                result=self._outcome(
                    restriction_id=plan.source.restriction_id,
                    template=plan.template,
                    template_version=plan.template_version,
                    verification_status="unverifiable",
                    missing=[
                        f"declared_requirement:{role}"
                        for role in effective["missing_registry_roles"]
                    ],
                    effective_requirements=requirements,
                ),
                tool_calls=[],
                layers={},
                timings_ms=timings_ms,
            )

        layers, retrieval_calls = await self._retrieve_layers(
            mcp_client, requirements, scenario_id
        )
        resolution = self.data_gate.resolve(plan, layers, requirements)
        timings_ms["requirements_resolution"] = (
            perf_counter() - resolution_started
        ) * 1000
        if not resolution.executable:
            return ComplianceExecution(
                plan=plan,
                result=self._outcome(
                    restriction_id=plan.source.restriction_id,
                    template=plan.template,
                    template_version=plan.template_version,
                    verification_status="unverifiable",
                    missing=resolution.missing,
                    effective_requirements=requirements,
                    resolved_requirements=resolution.resolved,
                ),
                tool_calls=retrieval_calls,
                layers=layers,
                timings_ms=timings_ms,
            )

        entry = self.registry.get(plan.template, plan.template_version)
        limit_error = self._limit_error(entry, resolution.layers)
        if limit_error:
            return ComplianceExecution(
                plan=plan,
                result=self._outcome(
                    restriction_id=plan.source.restriction_id,
                    template=plan.template,
                    template_version=plan.template_version,
                    verification_status="unverifiable",
                    missing=[limit_error],
                    effective_requirements=requirements,
                    resolved_requirements=resolution.resolved,
                ),
                tool_calls=retrieval_calls,
                layers=layers,
                timings_ms=timings_ms,
            )

        tool_name, arguments = self._tool_call(plan, params, resolution)
        execution_started = perf_counter()
        raw_result = await self.tools.execute_named_tool(
            mcp_client,
            tool_name,
            {**arguments, "layers": resolution.layers},
        )
        timings_ms["template_execution"] = (perf_counter() - execution_started) * 1000
        coverage = VerificationCoverage.model_validate(raw_result["coverage"])
        summary = ComplianceSummary.model_validate(raw_result["summary"])
        if coverage.applicable_objects == 0:
            verification = "not_applicable"
            compliance = "unknown"
        elif coverage.checked_objects == 0:
            verification = "unverifiable"
            compliance = "unknown"
        elif coverage.unchecked_objects:
            verification = "partial"
            compliance = "violated" if summary.violated_objects else "passed"
        else:
            verification = "complete"
            compliance = "violated" if summary.violated_objects else "passed"
        result = ComplianceResult(
            restriction_id=plan.source.restriction_id,
            template=plan.template,
            template_version=plan.template_version,
            verification_status=verification,
            compliance_status=compliance,
            coverage=coverage,
            summary=summary,
            effective_requirements=requirements,
            resolved_requirements=resolution.resolved,
            missing_requirements=[],
            warnings=(
                ["Проверена только часть применимых объектов"]
                if verification == "partial"
                else []
            ),
            source={
                **plan.source.model_dump(mode="json"),
                "planner_status": plan.planner_status,
                "check_plan": plan.model_dump(mode="json"),
            },
            evidence=raw_result.get("evidence") or [],
            violated_features=raw_result.get("violated_objects"),
            passed_features=raw_result.get("passed_objects"),
        )
        stored_arguments = {
            key: value for key, value in arguments.items() if key != "layers"
        }
        tool_calls = [
            *retrieval_calls,
            {"function": {"name": tool_name, "arguments": stored_arguments}},
        ]
        return ComplianceExecution(
            plan=plan,
            result=result,
            tool_calls=tool_calls,
            layers=layers,
            timings_ms=timings_ms,
        )

    async def _retrieve_layers(
        self, mcp_client, requirements: DeclaredRequirements, scenario_id: int
    ) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
        layers: dict[str, dict[str, Any]] = {}
        calls: list[dict[str, Any]] = []
        for entity_type, tool_name, argument_name in (
            ("service", "GetServices", "services_names"),
            ("physical_object", "GetPhysicalObjects", "physical_objects_names"),
        ):
            names = list(
                dict.fromkeys(
                    item.entity
                    for item in requirements.layers
                    if item.entity_type.value == entity_type
                )
            )
            if not names:
                continue
            arguments = {argument_name: names, "scenario_id": scenario_id}
            layers.update(
                await self.tools.execute_named_tool(mcp_client, tool_name, arguments)
            )
            calls.append({"function": {"name": tool_name, "arguments": arguments}})

        for requirement in (
            item
            for item in requirements.layers
            if item.entity_type.value == "functional_zone"
        ):
            zone_names = (
                None
                if requirement.entity == "functional_zones"
                else [requirement.entity]
            )
            arguments = {
                "scenario_id": scenario_id,
                "zone_type_names": zone_names,
            }
            response = await self.tools.execute_named_tool(
                mcp_client, "GetFunctionalZones", arguments
            )
            if "functional_zones" in response:
                layers[requirement.entity] = response["functional_zones"]
            calls.append(
                {"function": {"name": "GetFunctionalZones", "arguments": arguments}}
            )
        return layers, calls

    @staticmethod
    def _tool_call(plan: CheckPlan, params, resolution) -> tuple[str, dict[str, Any]]:
        def layer(role: str) -> str:
            if role not in resolution.role_layers:
                raise ValueError(f"Layer role {role!r} is unresolved")
            return resolution.role_layers[role]

        common = {
            "restriction_id": plan.source.restriction_id,
            "template_version": plan.template_version,
            "provenance": plan.source.model_dump(mode="json"),
            "input_revision": ComplianceTemplateExecutor._input_revision(
                resolution.profiles
            ),
        }
        if plan.template == "distance_from_source":
            return "CheckDistanceFromSource", {
                **common,
                **params.model_dump(mode="json", exclude_none=True),
                "source_layer": layer(params.source_layer),
                "targets": [layer(role) for role in params.targets],
            }
        if plan.template == "distance_table":
            return "CheckDistanceTable", {
                **common,
                **params.model_dump(
                    mode="json",
                    exclude={"attribute_role", "null_policy", "out_of_range_policy"},
                ),
                "source_layer": layer(params.source_layer),
                "targets": [layer(role) for role in params.targets],
                "attribute_field": resolution.selected_fields[params.attribute_role],
            }
        if plan.template == "presence_within":
            return "CheckPresenceWithin", {
                **common,
                **params.model_dump(mode="json"),
                "objects_layer": layer(params.objects_layer),
                "required_neighbor_layers": [
                    layer(role) for role in params.required_neighbor_layers
                ],
            }
        if plan.template == "zonal_attribute_threshold":
            threshold = params.threshold_source
            return "CheckZonalAttributeThreshold", {
                **common,
                "objects_layer": layer(params.objects_layer),
                "zones_layer": layer(params.zones_layer),
                "object_attribute": resolution.selected_fields[params.attribute_role],
                "operator": params.operator,
                "constant_threshold": (
                    threshold.value if threshold.kind == "constant" else None
                ),
                "zone_threshold_attribute": (
                    resolution.selected_fields[threshold.role]
                    if threshold.kind == "attribute_role"
                    else None
                ),
                "join_predicate": params.join_predicate,
                "result_mode": params.result_mode,
            }
        if plan.template == "zonal_ratio":
            return "CheckZonalRatio", {
                **common,
                "zones_layer": layer(params.zones_layer),
                "numerator_layer": layer(params.numerator.layer),
                "operator": params.operator,
                "threshold": params.threshold,
                "result_mode": params.result_mode,
                "invalid_geometry_policy": params.invalid_geometry_policy,
            }
        raise UnsupportedTemplateError(plan.template)

    @staticmethod
    def _input_revision(profiles: dict[str, dict[str, Any]]) -> str | None:
        revisions = [
            f"{role}:{profile['revision']}"
            for role, profile in sorted(profiles.items())
            if profile.get("revision")
        ]
        if not revisions:
            return None
        digest = hashlib.sha256("\n".join(revisions).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    @staticmethod
    def _limit_error(entry, layers: dict[str, dict[str, Any]]) -> str | None:
        for name, layer in layers.items():
            count = len(layer.get("features") or [])
            if count > entry.max_features:
                return f"layer:{name}:feature_limit:{entry.max_features}"
        payload_size = len(json.dumps(layers, ensure_ascii=False).encode("utf-8"))
        if payload_size > entry.max_payload_bytes:
            return f"payload_limit:{entry.max_payload_bytes}"
        return None

    @staticmethod
    def _outcome(
        *,
        restriction_id: str,
        template: str,
        template_version: int,
        verification_status: str,
        missing: list[str],
        warnings: list[str] | None = None,
        effective_requirements: DeclaredRequirements | None = None,
        resolved_requirements: list | None = None,
    ) -> ComplianceResult:
        return ComplianceResult(
            restriction_id=restriction_id,
            template=template,
            template_version=template_version,
            verification_status=verification_status,
            compliance_status="unknown",
            coverage=VerificationCoverage(
                applicable_objects=0,
                checked_objects=0,
                unchecked_objects=0,
                fill_rate=0,
            ),
            summary=ComplianceSummary(violated_objects=0, passed_objects=0),
            effective_requirements=effective_requirements or DeclaredRequirements(),
            resolved_requirements=resolved_requirements or [],
            missing_requirements=missing,
            warnings=warnings or [],
        )
