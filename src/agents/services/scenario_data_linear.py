"""Plan-first executor for the scenario-data feature-flag pilot."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from fastmcp import Client
from idu_service_auth import KeycloakTokenClient
from loguru import logger

from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    StructuredPartRequest,
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.mcp_clients.urban_mcp_client import UrbanMcpClient, UrbanMcpTool
from src.agents.services.scenario_data_aggregate import (
    aggregate_result,
    answer_records,
    extract_records,
    sanitize_public_answer,
    unresolved_references,
)
from src.agents.services.scenario_data_execution_context import (
    ScenarioExecutionContext,
)
from src.agents.services.scenario_data_mapping import (
    UrbanMappingResolver,
    bind_mapping_arguments,
    context_mapping_snapshots,
    enrich_acquisition_mappings,
    ensure_entity_retrieval_outputs,
    mapping_snapshot,
)
from src.agents.services.service_entities.scenario_data_plan import (
    ExecutionLedger,
    ExecutionRecord,
    PlanStep,
    PlanStepKind,
    StepStatus,
)
from src.common.service_auth import ServiceTokenAuth, user_id_from_jwt

if TYPE_CHECKING:
    from src.agents.services.scenario_data_service import ScenarioDataService

MAX_URBAN_CALLS = 10
MAX_WORKSPACE_CALLS = 20
MAX_REPLANS = 3
ACTIVE_DEADLINE_SECONDS = 5 * 60
ABSOLUTE_DEADLINE_SECONDS = 15 * 60


class ScenarioDataLinearWorkflow:
    """Build a complete plan, execute it sequentially and validate before answering."""

    def __init__(
        self,
        owner: "ScenarioDataService",
        *,
        workspace_enabled: bool = False,
        idu_mcp_url: str | None = None,
        service_auth: KeycloakTokenClient | None = None,
    ) -> None:
        self.owner = owner
        self.workspace_enabled = workspace_enabled and bool(idu_mcp_url)
        self.idu_mcp_url = idu_mcp_url
        self.service_auth = service_auth
        self.mapping_resolver = UrbanMappingResolver()

    async def run(
        self,
        *,
        request_id: str,
        urban_mcp_client: UrbanMcpClient,
        token_ref: list[str],
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int | None,
        project_id: int | None,
        chat_id: str | None,
        history: list[dict],
        context: dict[str, Any],
        tools: list[UrbanMcpTool],
        observations: list[dict[str, Any]],
        parts: list,
        persist_history: bool,
    ) -> AsyncGenerator[dict[str, Any], None]:
        started = time.monotonic()
        ledger = ExecutionLedger()
        fingerprints: set[str] = set()
        artifact_handles: dict[str, str] = {}
        bootstrap_satisfied: set[str] = set()
        mappings = context_mapping_snapshots(context)
        for snapshot in mappings:
            observations.append(
                {
                    "context": "Подтверждённый маппинг из истории чата",
                    "mapping": snapshot,
                    "summary": (
                        f"Домен {snapshot['domain']}: "
                        f"{len(snapshot.get('matches') or [])} пар name/id"
                    ),
                }
            )

        yield self._event(
            request_id,
            "status",
            {"status": "planning", "text": "Составляю план получения данных…"},
        )
        acquisition = await self._bounded_llm(
            request_id,
            started,
            self.owner.plan_builder.build_acquisition_plan(
                model,
                user_query,
                history,
                scenario_id,
                context,
                project_id=project_id,
            ),
        )
        acquisition = enrich_acquisition_mappings(acquisition, user_query, mappings)
        acquisition = ensure_entity_retrieval_outputs(acquisition, user_query)
        execution_context = ScenarioExecutionContext.from_acquisition(
            acquisition,
            user_query=user_query,
            scenario_id=scenario_id,
            project_id=project_id,
        )
        execution_context.update_mappings(mappings)
        plan_payload = acquisition.model_dump(mode="json")
        yield self._event(request_id, "plan_created", plan_payload)
        parts.append(StructuredPartRequest(kind="plan", payload=plan_payload))
        if acquisition.clarification:
            clarification = sanitize_public_answer(acquisition.clarification)
            yield self._event(
                request_id,
                "clarification_required",
                {"text": clarification},
            )
            for event in self.owner._answer_events(clarification):
                yield self.owner._buf(request_id, event)
            parts.append(
                TextPartRequest(kind="text", payload=TextPayload(text=clarification))
            )
            await self.owner._complete_pipeline(
                request_id,
                token_ref[0],
                chat_id,
                parts,
                scenario_id=scenario_id,
                persist_history=persist_history,
                context_model=model,
            )
            return

        mapping_calls = self.mapping_resolver.plan_calls(
            acquisition,
            tools,
            scenario_id,
            project_id=project_id,
            known_mappings=mappings,
        )
        if not mapping_calls:
            mapping_calls = self.mapping_resolver.plan_entity_discovery_calls(
                acquisition,
                user_query,
                tools,
                scenario_id,
                project_id=project_id,
            )
        if mapping_calls:
            yield self._event(
                request_id,
                "mapping_started",
                {
                    "count": len(mapping_calls),
                    "text": "Получаю актуальные справочники…",
                },
            )
        for call in mapping_calls:
            if ledger.urban_calls >= MAX_URBAN_CALLS:
                break
            mapping_step_id = f"mapping_{ledger.urban_calls + 1}"
            mapping_arguments = self.owner._prepare_arguments(
                call.tool,
                call.arguments,
                scenario_id,
                project_id=project_id,
            )
            mapping_fingerprint = json.dumps(
                [
                    PlanStepKind.URBAN_TOOL.value,
                    call.tool.group,
                    call.tool.name,
                    mapping_arguments,
                ],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            fingerprints.add(mapping_fingerprint)
            mapping_attempt = execution_context.start_attempt(
                revision=0,
                step_id=mapping_step_id,
                purpose=f"Разрешить {call.need.domain}: {call.need.values}",
                kind=PlanStepKind.URBAN_TOOL.value,
                group=call.tool.group,
                tool_name=call.tool.name,
                arguments=mapping_arguments,
                satisfies=[],
            )
            yield self._event(
                request_id,
                "step_context",
                mapping_attempt.model_dump(mode="json"),
            )
            result = None
            try:
                async for event, value in self._execute_urban(
                    request_id,
                    urban_mcp_client,
                    token_ref,
                    call.tool,
                    mapping_arguments,
                    scenario_id,
                    ledger,
                    parts,
                    project_id=project_id,
                    step_id=mapping_step_id,
                ):
                    yield event
                    if value is not None:
                        result = value
            except PipelineSuspendedError:
                raise
            except Exception as exc:
                logger.warning(f"Scenario-data mapping {mapping_step_id} failed: {exc}")
                execution_context.fail_attempt(mapping_attempt, exc)
                failure_observation = execution_context.attempt_observation(
                    mapping_attempt
                )
                observations.append(failure_observation)
                ledger.records.append(
                    ExecutionRecord(
                        step_id=mapping_step_id,
                        revision=0,
                        status=StepStatus.FAILED,
                        call_fingerprint=mapping_fingerprint,
                        observation_index=len(observations) - 1,
                        error=str(exc),
                    )
                )
                yield self._event(
                    request_id,
                    "step_failed",
                    mapping_attempt.model_dump(mode="json"),
                )
                continue
            snapshot = mapping_snapshot(call, result)
            if (
                self.workspace_enabled
                and chat_id
                and ledger.workspace_calls < MAX_WORKSPACE_CALLS
            ):
                mapping_step = PlanStep(
                    step_id=mapping_step_id,
                    purpose=f"Маппинг {call.need.domain}",
                    group=call.tool.group,
                    tool_name=call.tool.name,
                    arguments=call.arguments,
                    satisfies=[call.requirement_id],
                    expected_output="Актуальный справочник",
                )
                try:
                    artifact, artifact_events = await self._create_workspace_artifact(
                        request_id,
                        result,
                        chat_id,
                        token_ref,
                        mapping_step,
                    )
                    for event in artifact_events:
                        yield event
                    if artifact:
                        ledger.workspace_calls += 1
                        artifact_handles[mapping_step_id] = artifact["handle"]
                        snapshot["artifact"] = artifact
                except PipelineSuspendedError:
                    raise
                except Exception as exc:
                    logger.warning(f"Could not cache mapping {mapping_step_id}: {exc}")
            ledger.records.append(
                ExecutionRecord(
                    step_id=mapping_step_id,
                    revision=0,
                    status=StepStatus.COMPLETED,
                    call_fingerprint=mapping_fingerprint,
                )
            )
            bootstrap_satisfied.add(call.requirement_id)
            mappings.append(snapshot)
            mapping_observation = {
                "context": "Актуальный маппинг",
                "mapping": snapshot,
                "summary": f"Маппинг получен из {snapshot['source_tool']}",
                "resolved_reference_domain": call.need.domain,
            }
            execution_context.complete_attempt(mapping_attempt, mapping_observation)
            mapping_observation["step_context"] = mapping_attempt.model_dump(
                mode="json"
            )
            observations.append(mapping_observation)
        if mapping_calls:
            yield self._event(
                request_id,
                "mapping_completed",
                {"count": len(mappings), "text": "Актуальные справочники получены"},
            )
            acquisition = enrich_acquisition_mappings(acquisition, user_query, mappings)
            acquisition = ensure_entity_retrieval_outputs(acquisition, user_query)
            execution_context.update_acquisition(acquisition)
            execution_context.update_mappings(mappings)

        revision = 1
        try:
            plan = await self._bounded_llm(
                request_id,
                started,
                self.owner.plan_builder.build_execution_plan(
                    model,
                    user_query,
                    acquisition,
                    tools,
                    mappings,
                    scenario_id=scenario_id,
                    project_id=project_id,
                    revision=revision,
                    observations=observations,
                    completed_fingerprints=sorted(fingerprints),
                    completed_step_ids=sorted(ledger.completed_step_ids),
                    workspace_enabled=self.workspace_enabled,
                    execution_context=execution_context.planner_snapshot(
                        urban_calls=ledger.urban_calls,
                        workspace_calls=ledger.workspace_calls,
                        replans=ledger.replans,
                    ),
                ),
            )
        except ValueError as exc:
            logger.warning(f"Could not build initial scenario-data plan: {exc}")
            reasons = [f"не удалось составить исполнимый план: {exc}"]
            answer = execution_context.failure_note(reasons)
            execution_snapshot = execution_context.planner_snapshot(
                urban_calls=ledger.urban_calls,
                workspace_calls=ledger.workspace_calls,
                replans=ledger.replans,
            )
            failure = {
                "code": "scenario_data_plan_invalid",
                "reasons": reasons,
                "urban_calls": ledger.urban_calls,
                "workspace_calls": ledger.workspace_calls,
                "replans": ledger.replans,
                "execution_context": execution_snapshot,
            }
            yield self._event(request_id, "pipeline_failed", failure)
            parts.append(StructuredPartRequest(kind="failure", payload=failure))
            answer = sanitize_public_answer(answer)
            for event in self.owner._answer_events(answer):
                yield self.owner._buf(request_id, event)
            parts.append(TextPartRequest(kind="text", payload=TextPayload(text=answer)))
            await self.owner._complete_pipeline(
                request_id,
                token_ref[0],
                chat_id,
                parts,
                scenario_id=scenario_id,
                persist_history=persist_history,
                context_model=model,
            )
            return

        answer = ""
        validation_reasons: list[str] = []
        while True:
            revision_payload = {
                **plan.model_dump(mode="json"),
                "execution_context": execution_context.planner_snapshot(
                    urban_calls=ledger.urban_calls,
                    workspace_calls=ledger.workspace_calls,
                    replans=ledger.replans,
                ),
            }
            yield self._event(request_id, "plan_revision_created", revision_payload)
            parts.append(
                StructuredPartRequest(kind="plan_revision", payload=revision_payload)
            )

            plan_failed = False
            for step in plan.steps:
                if await self._deadline_exceeded(request_id, started):
                    validation_reasons = ["превышен лимит активной работы 5 минут"]
                    plan_failed = True
                    break
                if await self.owner.state_store.is_cancelled(request_id):
                    raise asyncio.CancelledError
                if (
                    step.kind == PlanStepKind.URBAN_TOOL
                    and ledger.urban_calls >= MAX_URBAN_CALLS
                ):
                    validation_reasons = ["исчерпан лимит из 10 вызовов Urban MCP"]
                    plan_failed = True
                    break
                if (
                    step.kind == PlanStepKind.WORKSPACE
                    and ledger.workspace_calls >= MAX_WORKSPACE_CALLS
                ):
                    validation_reasons = ["исчерпан лимит из 20 workspace-операций"]
                    plan_failed = True
                    break
                if not set(step.depends_on).issubset(ledger.completed_step_ids):
                    validation_reasons = [
                        f"не выполнены зависимости шага {step.step_id}"
                    ]
                    plan_failed = True
                    break
                attempt = execution_context.start_step(
                    plan.revision, step, dict(step.arguments)
                )
                yield self._event(
                    request_id,
                    "step_context",
                    attempt.model_dump(mode="json"),
                )
                try:
                    resolved_arguments = (
                        self._resolve_workspace_arguments(
                            step.arguments, artifact_handles
                        )
                        if step.kind == PlanStepKind.WORKSPACE
                        else step.arguments
                    )
                except ValueError as exc:
                    validation_reasons = [str(exc)]
                    execution_context.fail_attempt(attempt, exc)
                    failed_observation = execution_context.attempt_observation(attempt)
                    observations.append(failed_observation)
                    ledger.records.append(
                        ExecutionRecord(
                            step_id=step.step_id,
                            revision=plan.revision,
                            status=StepStatus.FAILED,
                            observation_index=len(observations) - 1,
                            error=str(exc),
                        )
                    )
                    yield self._event(
                        request_id,
                        "step_failed",
                        attempt.model_dump(mode="json"),
                    )
                    plan_failed = True
                    break
                fingerprint = self._fingerprint(step, scenario_id, resolved_arguments)
                if fingerprint in fingerprints:
                    error = ValueError(
                        f"план повторяет уже проверенный вызов {step.tool_name} "
                        "с теми же аргументами"
                    )
                    validation_reasons = [str(error)]
                    execution_context.fail_attempt(attempt, error)
                    failed_observation = execution_context.attempt_observation(attempt)
                    observations.append(failed_observation)
                    ledger.records.append(
                        ExecutionRecord(
                            step_id=step.step_id,
                            revision=plan.revision,
                            status=StepStatus.FAILED,
                            call_fingerprint=fingerprint,
                            observation_index=len(observations) - 1,
                            error=str(error),
                        )
                    )
                    yield self._event(
                        request_id,
                        "step_failed",
                        attempt.model_dump(mode="json"),
                    )
                    plan_failed = True
                    break
                fingerprints.add(fingerprint)
                try:
                    if step.kind == PlanStepKind.WORKSPACE:
                        if not self.workspace_enabled or not chat_id:
                            raise ValueError("workspace недоступен для этого чата")
                        result = None
                        async for event, value in self._execute_workspace(
                            request_id,
                            token_ref,
                            step,
                            {**resolved_arguments, "chat_id": chat_id},
                            ledger,
                            parts,
                        ):
                            yield event
                            if value is not None:
                                result = value
                        observation, result_events = self._consume_workspace_result(
                            request_id, step, result, parts
                        )
                    else:
                        tool = urban_mcp_client.get_tool(
                            step.group or "", step.tool_name
                        )
                        resolved_arguments = bind_mapping_arguments(
                            tool,
                            resolved_arguments,
                            mappings,
                            " ".join(
                                (
                                    user_query,
                                    acquisition.objective,
                                    step.purpose,
                                )
                            ),
                        )
                        arguments = self.owner._prepare_arguments(
                            tool,
                            resolved_arguments,
                            scenario_id,
                            project_id=project_id,
                        )
                        attempt.arguments = dict(arguments)
                        effective_fingerprint = self._fingerprint(
                            step, scenario_id, arguments
                        )
                        if effective_fingerprint != fingerprint:
                            fingerprints.discard(fingerprint)
                            if effective_fingerprint in fingerprints:
                                raise ValueError(
                                    f"вызов {step.tool_name} с подготовленными "
                                    "аргументами уже был проверен"
                                )
                            fingerprints.add(effective_fingerprint)
                            fingerprint = effective_fingerprint
                        yield self._event(
                            request_id,
                            "step_context",
                            attempt.model_dump(mode="json"),
                        )
                        result = None
                        async for event, value in self._execute_urban(
                            request_id,
                            urban_mcp_client,
                            token_ref,
                            tool,
                            arguments,
                            scenario_id,
                            ledger,
                            parts,
                            project_id=project_id,
                            step_id=step.step_id,
                            plan_step=step,
                            fingerprint=fingerprint,
                        ):
                            yield event
                            if value is not None:
                                result = value
                        observation, result_events = await self._consume_result(
                            request_id,
                            step,
                            result,
                            chat_id,
                            token_ref,
                            ledger,
                            parts,
                        )
                        observation["arguments"] = arguments
                    for event in result_events:
                        yield event
                    execution_context.complete_attempt(attempt, observation)
                    observation["step_context"] = attempt.model_dump(mode="json")
                    observations.append(observation)
                    artifact = observation.get("artifact") or {}
                    if isinstance(artifact.get("handle"), str):
                        artifact_handles[step.step_id] = artifact["handle"]
                    ledger.records.append(
                        ExecutionRecord(
                            step_id=step.step_id,
                            revision=plan.revision,
                            status=StepStatus.COMPLETED,
                            call_fingerprint=fingerprint,
                            observation_index=len(observations) - 1,
                        )
                    )
                    yield self._event(
                        request_id,
                        "step_completed",
                        {"step_id": step.step_id, "revision": plan.revision},
                    )
                except PipelineSuspendedError:
                    raise
                except Exception as exc:
                    logger.warning(f"Scenario-data step {step.step_id} failed: {exc}")
                    if step.kind == PlanStepKind.WORKSPACE and not step.satisfies:
                        execution_context.skip_attempt(attempt, exc)
                        skipped_observation = execution_context.attempt_observation(
                            attempt
                        )
                        ledger.records.append(
                            ExecutionRecord(
                                step_id=step.step_id,
                                revision=plan.revision,
                                status=StepStatus.SKIPPED,
                                call_fingerprint=fingerprint,
                                observation_index=len(observations),
                                error=str(exc),
                            )
                        )
                        observations.append(skipped_observation)
                        yield self._event(
                            request_id,
                            "step_completed",
                            {
                                "step_id": step.step_id,
                                "revision": plan.revision,
                                "status": "skipped",
                            },
                        )
                        continue
                    execution_context.fail_attempt(attempt, exc)
                    failed_observation = execution_context.attempt_observation(attempt)
                    ledger.records.append(
                        ExecutionRecord(
                            step_id=step.step_id,
                            revision=plan.revision,
                            status=StepStatus.FAILED,
                            call_fingerprint=fingerprint,
                            observation_index=len(observations),
                            error=str(exc),
                        )
                    )
                    observations.append(failed_observation)
                    yield self._event(
                        request_id,
                        "step_failed",
                        attempt.model_dump(mode="json"),
                    )
                    validation_reasons = [f"шаг {step.step_id} завершился ошибкой"]
                    plan_failed = True
                    break

            yield self._event(
                request_id,
                "validation_started",
                {"revision": plan.revision, "text": "Проверяю полноту результата…"},
            )
            if not plan_failed:
                deterministic_reasons = [
                    *self._plan_completion_reasons(
                        acquisition, plan, ledger, bootstrap_satisfied
                    ),
                    *self._required_output_reasons(plan, observations),
                ]
                if deterministic_reasons:
                    validation_reasons = deterministic_reasons
                    plan_failed = True
            if not plan_failed:
                answer = await self._bounded_llm(
                    request_id,
                    started,
                    self.owner._draft_answer(
                        model, user_query, observations, temperature, history
                    ),
                )
                grounded_layer_result = bool(plan.required_output.layers)
                if grounded_layer_result:
                    validation_reasons = []
                    sufficient = True
                else:
                    verdict = await self._bounded_llm(
                        request_id,
                        started,
                        self.owner.evaluator.evaluate(
                            model, user_query, observations, answer
                        ),
                    )
                    validation_reasons = verdict.reasons
                    sufficient = verdict.sufficient
                if sufficient:
                    validation_payload = {
                        "sufficient": True,
                        "revision": plan.revision,
                        "criteria": plan.required_output.model_dump(mode="json"),
                        "execution_context": execution_context.planner_snapshot(
                            urban_calls=ledger.urban_calls,
                            workspace_calls=ledger.workspace_calls,
                            replans=ledger.replans,
                        ),
                    }
                    yield self._event(
                        request_id, "validation_completed", validation_payload
                    )
                    parts.append(
                        StructuredPartRequest(
                            kind="validation", payload=validation_payload
                        )
                    )
                    break

            if (
                ledger.replans >= MAX_REPLANS
                or ledger.urban_calls >= MAX_URBAN_CALLS
                or ledger.workspace_calls >= MAX_WORKSPACE_CALLS
                or await self._deadline_exceeded(request_id, started)
            ):
                if not answer:
                    if await self._deadline_exceeded(request_id, started):
                        answer = (
                            "Не удалось завершить сбор данных в установленный срок. "
                            "Уточните требуемый набор данных или сократите область запроса."
                        )
                    else:
                        answer = await self._bounded_llm(
                            request_id,
                            started,
                            self.owner._draft_answer(
                                model,
                                user_query,
                                observations,
                                temperature,
                                history,
                            ),
                        )
                failure_note = execution_context.failure_note(validation_reasons)
                answer = (
                    f"{answer.rstrip()}\n\n{failure_note}"
                    if answer.strip()
                    else failure_note
                )
                execution_snapshot = execution_context.planner_snapshot(
                    urban_calls=ledger.urban_calls,
                    workspace_calls=ledger.workspace_calls,
                    replans=ledger.replans,
                )
                failure = {
                    "code": "scenario_data_budget_exhausted",
                    "reasons": validation_reasons,
                    "urban_calls": ledger.urban_calls,
                    "workspace_calls": ledger.workspace_calls,
                    "replans": ledger.replans,
                    "execution_context": execution_snapshot,
                }
                yield self._event(request_id, "pipeline_failed", failure)
                parts.append(StructuredPartRequest(kind="failure", payload=failure))
                break

            ledger.replans += 1
            revision += 1
            yield self._event(
                request_id,
                "replanning",
                {"revision": revision, "reasons": validation_reasons},
            )
            try:
                plan = await self._bounded_llm(
                    request_id,
                    started,
                    self.owner.plan_builder.build_execution_plan(
                        model,
                        user_query,
                        acquisition,
                        tools,
                        mappings,
                        scenario_id=scenario_id,
                        project_id=project_id,
                        revision=revision,
                        reason="; ".join(validation_reasons) or "недостаточно данных",
                        observations=observations,
                        completed_fingerprints=sorted(fingerprints),
                        completed_step_ids=sorted(ledger.completed_step_ids),
                        workspace_enabled=self.workspace_enabled,
                        execution_context=execution_context.planner_snapshot(
                            urban_calls=ledger.urban_calls,
                            workspace_calls=ledger.workspace_calls,
                            replans=ledger.replans,
                        ),
                    ),
                )
            except ValueError as exc:
                logger.warning(f"Could not build scenario-data replan: {exc}")
                reasons = validation_reasons or [
                    f"не удалось составить новую ревизию плана: {exc}"
                ]
                failure_note = execution_context.failure_note(reasons)
                answer = (
                    f"{answer.rstrip()}\n\n{failure_note}"
                    if answer.strip()
                    else failure_note
                )
                execution_snapshot = execution_context.planner_snapshot(
                    urban_calls=ledger.urban_calls,
                    workspace_calls=ledger.workspace_calls,
                    replans=ledger.replans,
                )
                failure = {
                    "code": "scenario_data_replan_invalid",
                    "reasons": reasons,
                    "urban_calls": ledger.urban_calls,
                    "workspace_calls": ledger.workspace_calls,
                    "replans": ledger.replans,
                    "execution_context": execution_snapshot,
                }
                yield self._event(request_id, "pipeline_failed", failure)
                parts.append(StructuredPartRequest(kind="failure", payload=failure))
                break

        answer = sanitize_public_answer(answer)
        for event in self.owner._answer_events(answer):
            yield self.owner._buf(request_id, event)
        if answer.strip():
            parts.append(
                TextPartRequest(kind="text", payload=TextPayload(text=answer.strip()))
            )
        await self.owner._complete_pipeline(
            request_id,
            token_ref[0],
            chat_id,
            parts,
            scenario_id=scenario_id,
            persist_history=persist_history,
            context_model=model,
        )

    async def _execute_urban(
        self,
        request_id: str,
        client: UrbanMcpClient,
        token_ref: list[str],
        tool: UrbanMcpTool,
        arguments: dict[str, Any],
        scenario_id: int | None,
        ledger: ExecutionLedger,
        parts: list,
        *,
        project_id: int | None = None,
        step_id: str,
        plan_step: PlanStep | None = None,
        fingerprint: str | None = None,
    ):
        source = f"URBAN_MCP/{tool.group}"
        yield self._event(
            request_id,
            "step_started",
            {
                "step_id": step_id,
                "purpose": plan_step.purpose if plan_step else "Получение маппинга",
                "tool_name": tool.name,
            },
        ), None
        call = {"tool_name": tool.name, "arguments": arguments, "group": tool.group}
        yield self.owner._buf(
            request_id, self.owner._tool_call_event(call, source)
        ), None
        parts.append(
            ToolCallPartRequest(
                kind="tool_call",
                payload=ToolCallPayload(
                    execution_mode="sequential",
                    calls=[
                        ToolCall(
                            step=ledger.urban_calls + 1,
                            tool_name=tool.name,
                            arguments=arguments,
                        )
                    ],
                ),
                mcp_source=source,
            )
        )
        result_box: list[Any] = []
        meta = {}
        if scenario_id is not None:
            meta["scenario_id"] = scenario_id
        if project_id is not None:
            meta["project_id"] = project_id
        try:
            async for event in self.owner._retryable_operation(
                request_id,
                client,
                token_ref,
                lambda: client.execute_tool(
                    tool.group,
                    tool.name,
                    arguments,
                    meta=meta,
                ),
                result_box,
                retry_transient=True,
            ):
                yield self.owner._buf(request_id, event), None
        finally:
            # Failed read-only calls are attempts too and must consume the bounded budget.
            ledger.urban_calls += 1
        yield self._event(
            request_id,
            "status",
            {"status": "tool_execution", "text": f"Получены данные: {tool.title}"},
        ), self.owner._unwrap_result(result_box[0])

    async def _consume_result(
        self,
        request_id: str,
        step: PlanStep,
        result: Any,
        chat_id: str | None,
        token_ref: list[str],
        ledger: ExecutionLedger,
        parts: list,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        layer_count = 0
        for path, feature_collection in self.owner._feature_collections(result):
            layer_count += 1
            name = step.layer_name or step.purpose
            if path:
                name = f"{name} · {path}"
            # Raw geometry is SSE truth for the active browser, never a chat/context part.
            events.append(
                self._event(
                    request_id,
                    "feature_collection",
                    {"name": name, "feature_collection": feature_collection},
                )
            )
        table = self.owner._table_from_result(
            result,
            name=f"urban_{step.group}_{step.tool_name}",
            title=step.purpose,
        )
        if table is not None:
            events.append(self._event(request_id, "table", table))
            parts.append(self.owner._table_part(table))

        observation: dict[str, Any] = {
            "tool": f"{step.group}.{step.tool_name}",
            "arguments": step.arguments,
            "layer_count": layer_count,
            "table_count": int(table is not None),
            "summary": self.owner._result_summary(result),
            "satisfies": step.satisfies,
        }
        aggregate = aggregate_result(result)
        if aggregate is not None:
            observation["aggregate"] = aggregate
            pending = unresolved_references(aggregate)
            if pending:
                observation["unresolved_references"] = pending
        records_for_answer = answer_records(result)
        if records_for_answer is not None:
            observation["answer_records"] = records_for_answer

        if (
            self.workspace_enabled
            and chat_id
            and ledger.workspace_calls < MAX_WORKSPACE_CALLS
        ):
            try:
                artifact, artifact_events = await self._create_workspace_artifact(
                    request_id, result, chat_id, token_ref, step
                )
                events.extend(artifact_events)
                if artifact:
                    ledger.workspace_calls += 1
                    observation["artifact"] = artifact
                    parts.append(
                        StructuredPartRequest(kind="artifact_ref", payload=artifact)
                    )
                    events.append(self._event(request_id, "artifact_created", artifact))
            except PipelineSuspendedError:
                raise
            except Exception as exc:
                observation["workspace_error"] = str(exc)
                logger.warning(f"Could not cache result of {step.step_id}: {exc}")
        return observation, events

    async def _create_workspace_artifact(
        self,
        request_id: str,
        result: Any,
        chat_id: str,
        token_ref: list[str],
        step: PlanStep,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        records = extract_records(result)
        feature_collection = next(
            (collection for _, collection in self.owner._feature_collections(result)),
            None,
        )
        if records is None and feature_collection is None:
            return None, []
        arguments: dict[str, Any] = {"chat_id": chat_id}
        if feature_collection is not None:
            arguments["feature_collection"] = feature_collection
        else:
            arguments["records"] = records
        client = self._workspace_client(token_ref[0])
        result_box: list[Any] = []
        events: list[dict[str, Any]] = []
        async for event in self.owner._retryable_operation(
            request_id,
            client,
            token_ref,
            lambda: client.execute_tool("WorkspaceCreate", arguments),
            result_box,
        ):
            events.append(self.owner._buf(request_id, event))
        artifact = self.owner._unwrap_result(result_box[0])
        if not isinstance(artifact, dict):
            return None, events
        return (
            {
                **artifact,
                "source_step": step.step_id,
                "source_tool": f"{step.group}.{step.tool_name}",
            },
            events,
        )

    async def _execute_workspace(
        self,
        request_id: str,
        token_ref: list[str],
        step: PlanStep,
        arguments: dict[str, Any],
        ledger: ExecutionLedger,
        parts: list,
    ):
        yield self._event(
            request_id,
            "step_started",
            {
                "step_id": step.step_id,
                "purpose": step.purpose,
                "tool_name": step.tool_name,
            },
        ), None
        call = {"tool_name": step.tool_name, "arguments": arguments}
        yield self.owner._buf(
            request_id, self.owner._tool_call_event(call, "IDU_MCP/workspace")
        ), None
        parts.append(
            ToolCallPartRequest(
                kind="tool_call",
                payload=ToolCallPayload(
                    execution_mode="sequential",
                    calls=[
                        ToolCall(
                            step=ledger.workspace_calls + 1,
                            tool_name=step.tool_name,
                            arguments=arguments,
                        )
                    ],
                ),
                mcp_source="IDU_MCP/workspace",
            )
        )
        client = self._workspace_client(token_ref[0])
        result_box: list[Any] = []
        async for event in self.owner._retryable_operation(
            request_id,
            client,
            token_ref,
            lambda: client.execute_tool(step.tool_name, arguments),
            result_box,
        ):
            yield self.owner._buf(request_id, event), None
        ledger.workspace_calls += 1
        yield self._event(
            request_id,
            "status",
            {"status": "workspace", "text": f"Обработан набор: {step.purpose}"},
        ), self.owner._unwrap_result(result_box[0])

    def _consume_workspace_result(
        self,
        request_id: str,
        step: PlanStep,
        result: Any,
        parts: list,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        layer_count = 0
        if isinstance(result, dict):
            collection = result.get("feature_collection")
            if isinstance(collection, dict):
                layer_count = 1
                events.append(
                    self._event(
                        request_id,
                        "feature_collection",
                        {
                            "name": step.layer_name or step.purpose,
                            "feature_collection": collection,
                        },
                    )
                )
        table = self.owner._table_from_result(
            result, name=f"workspace_{step.tool_name}", title=step.purpose
        )
        if table is not None:
            events.append(self._event(request_id, "table", table))
            parts.append(self.owner._table_part(table))
        observation: dict[str, Any] = {
            "tool": f"workspace.{step.tool_name}",
            "arguments": step.arguments,
            "summary": self.owner._result_summary(result),
            "satisfies": step.satisfies,
            "layer_count": layer_count,
            "table_count": int(table is not None),
        }
        if isinstance(result, dict) and isinstance(result.get("handle"), str):
            artifact = {
                **result,
                "source_step": step.step_id,
                "source_tool": f"workspace.{step.tool_name}",
            }
            observation["artifact"] = artifact
            parts.append(StructuredPartRequest(kind="artifact_ref", payload=artifact))
            events.append(self._event(request_id, "artifact_created", artifact))
        return observation, events

    def _workspace_client(self, token: str) -> IduMcpClient:
        if self.service_auth is None:
            raise RuntimeError("Service auth is not configured")
        return IduMcpClient(
            Client(
                self.idu_mcp_url,
                auth=ServiceTokenAuth(self.service_auth, user_id_from_jwt(token)),
            )
        )

    @classmethod
    def _resolve_workspace_arguments(
        cls, value: Any, artifact_handles: dict[str, str]
    ) -> Any:
        if isinstance(value, str) and value.startswith("$artifact:"):
            step_id = value.removeprefix("$artifact:")
            try:
                return artifact_handles[step_id]
            except KeyError as exc:
                raise ValueError(f"artifact шага {step_id} ещё не создан") from exc
        if isinstance(value, list):
            return [
                cls._resolve_workspace_arguments(item, artifact_handles)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: cls._resolve_workspace_arguments(item, artifact_handles)
                for key, item in value.items()
            }
        return value

    def _event(
        self, request_id: str, event_type: str, content: dict[str, Any]
    ) -> dict[str, Any]:
        return self.owner._buf(request_id, {"type": event_type, "content": content})

    async def _deadline_exceeded(self, request_id: str, started: float) -> bool:
        state = await self.owner.state_store.get_state(request_id) or {}
        wall = (
            time.time() - float(state["started_at"])
            if state.get("started_at")
            else time.monotonic() - started
        )
        active = wall - float(state.get("token_wait_seconds", 0))
        return wall >= ABSOLUTE_DEADLINE_SECONDS or active >= ACTIVE_DEADLINE_SECONDS

    @staticmethod
    def _plan_completion_reasons(
        acquisition, plan, ledger, bootstrap_satisfied: set[str]
    ) -> list[str]:
        completed = ledger.completed_step_ids
        missing_steps = [
            step.step_id for step in plan.steps if step.step_id not in completed
        ]
        covered = set(bootstrap_satisfied) | {
            requirement
            for step in plan.steps
            if step.step_id in completed
            for requirement in step.satisfies
        }
        missing_requirements = [
            item.requirement_id
            for item in acquisition.requirements
            if item.requirement_id not in covered
        ]
        reasons = []
        if missing_steps:
            reasons.append(f"не завершены шаги плана: {', '.join(missing_steps)}")
        if missing_requirements:
            reasons.append("не закрыты требования: " + ", ".join(missing_requirements))
        return reasons

    @staticmethod
    def _required_output_reasons(plan, observations: list[dict[str, Any]]) -> list[str]:
        if plan.required_output.layers and not any(
            int(item.get("layer_count") or 0) > 0 for item in observations
        ):
            return [
                "требуется географический слой, но ни один выполненный шаг не "
                "вернул геометрию"
            ]
        if plan.required_output.tables and not any(
            int(item.get("table_count") or 0) > 0 for item in observations
        ):
            return [
                "требуется таблица, но ни один выполненный шаг не вернул табличные "
                "данные"
            ]
        return []

    async def _bounded_llm(self, request_id: str, started: float, awaitable):
        """Bound LLM calls while still reacting promptly to an explicit cancel."""

        task = asyncio.create_task(awaitable)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=1.0)
                if done:
                    return task.result()
                if await self.owner.state_store.is_cancelled(request_id):
                    task.cancel()
                    raise asyncio.CancelledError
                if await self._deadline_exceeded(request_id, started):
                    task.cancel()
                    raise TimeoutError("scenario-data workflow deadline exceeded")
        finally:
            if not task.done():
                task.cancel()

    @staticmethod
    def _fingerprint(
        step: PlanStep,
        scenario_id: int | None,
        resolved_arguments: dict[str, Any] | None = None,
    ) -> str:
        arguments = dict(resolved_arguments or step.arguments)
        if scenario_id is not None and step.kind == PlanStepKind.URBAN_TOOL:
            arguments.setdefault("scenario_id", scenario_id)
        return json.dumps(
            [step.kind.value, step.group, step.tool_name, arguments],
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
