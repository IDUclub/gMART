"""Concurrent scenario checks with ordered orchestration events."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.dto.pzz_request_dto import PzzInputs
from src.agents.services.orchestrator import orchestrator_events as events
from src.agents.services.pipeline_state import PipelineStatus
from src.agents.services.service_entities.orchestrator_plan import (
    OrchestratorAgent,
    OrchestratorPlan,
    OrchestratorPlanMode,
    OrchestratorStep,
)
from src.agents.services.urban_analysis.urban_analysis_report import (
    REPORT_STEP_MAX_CHARS,
    ReportResult,
    generate_report,
    report_file_event,
)
from src.agents.services.urban_analysis.urban_analysis_zones import resolve_zones

ANALYSIS_STEPS = (
    (
        OrchestratorAgent.COMPLIANCE,
        "Проверь объекты выбранного сценария на соответствие всем применимым нормам.",
    ),
    (
        OrchestratorAgent.PROVISION,
        "Сравни обеспеченность по всем сервисам до и после проекта.",
    ),
    (
        OrchestratorAgent.PZZ,
        "Проверь здания сценария на соответствие функциональным зонам ПЗЗ.",
    ),
    (
        OrchestratorAgent.URBAN_ANALYSIS_REPORT,
        "Сформируй общий отчёт по результатам градостроительного анализа.",
    ),
)


@dataclass
class UrbanAnalysisResult:
    """Per-run data for a transport's persistence, never shared across requests."""

    steps: list[dict] = field(default_factory=list)
    table_parts: list = field(default_factory=list)
    zones: dict = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.FAILED


class UrbanAnalysisService:
    def __init__(
        self,
        restriction_service,
        provision_service,
        pzz_service,
        urban_api_client,
        state_store,
        file_store=None,
        *,
        llm_client=None,
    ):
        self.restriction_service = restriction_service
        self.provision_service = provision_service
        self.pzz_service = pzz_service
        self.urban_api_client = urban_api_client
        self.state_store = state_store
        self.file_store = file_store
        self.llm_client = (
            llm_client if llm_client is not None else restriction_service.llm_client
        )

    async def run_urban_analysis_pipeline(
        self,
        idu_mcp_client,
        effects_mcp_client,
        dvd_mcp_client,
        normgraph_mcp_client,
        token: str,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int,
        pzz_mcp_client=None,
        pzz_inputs: PzzInputs | dict | None = None,
        request_id: str | None = None,
        *,
        result: UrbanAnalysisResult | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run fixed checks and report without ChatStorage or a transport dependency.

        A future SSE controller or A2A executor wraps this generator, owns the
        outer request lifecycle, and persists ``result`` and file events. Closing
        the generator cancels and awaits every local check task.
        """
        if scenario_id is None:
            raise ValueError("urban analysis requires scenario_id")
        result = result if result is not None else UrbanAnalysisResult()
        request_id = request_id or self.state_store.new_request_id()
        steps = [
            OrchestratorStep(agent=agent, task=task) for agent, task in ANALYSIS_STEPS
        ]
        step_ids = [self.state_store.new_request_id() for _ in steps]
        yield events.status_event(
            "urban_analysis",
            "Проверка норм, обеспеченности и ПЗЗ выполняется параллельно…",
        )
        yield events.plan_event(
            OrchestratorPlan(mode=OrchestratorPlanMode.URBAN_ANALYSIS, steps=steps)
        )
        queue: asyncio.Queue = asyncio.Queue()
        backlogs = [deque() for _ in range(3)]
        observations = [events.StepObservation() for _ in range(3)]
        report_steps = []

        async def pump(index: int) -> None:
            try:
                pipeline = self._check_pipeline(
                    index,
                    step=steps[index],
                    request_id=step_ids[index],
                    idu_mcp_client=idu_mcp_client,
                    effects_mcp_client=effects_mcp_client,
                    dvd_mcp_client=dvd_mcp_client,
                    normgraph_mcp_client=normgraph_mcp_client,
                    pzz_mcp_client=pzz_mcp_client,
                    pzz_inputs=pzz_inputs,
                    token=token,
                    model=model,
                    temperature=temperature,
                    user_query=user_query,
                    scenario_id=scenario_id,
                    result=result,
                )
                async with aclosing(pipeline):
                    async for item in pipeline:
                        queue.put_nowait((index, "event", item))
                        if item.get("type") in events.TERMINAL_STATUSES:
                            # Do not consume any draft or further events after a
                            # terminal outcome, even if the producer keeps yielding.
                            break
            except Exception as exc:
                queue.put_nowait((index, "error", exc))
            finally:
                queue.put_nowait((index, "end", None))

        tasks = [
            asyncio.create_task(
                pump(index), name=f"urban-analysis:{request_id}:{index + 1}"
            )
            for index in range(3)
        ]
        files = []
        try:
            for index, step in enumerate(steps[:3]):
                yield events.step_started_event(
                    index + 1, step, step_ids[index], step.task
                )
                observation = observations[index]
                while True:
                    # Check the common queue even when replaying a backlog, so
                    # another check's token refresh is never hidden behind it.
                    if not queue.empty() or not backlogs[index]:
                        incoming, kind, item = await queue.get()
                        if incoming < index:
                            continue
                        if kind == "event" and item.get("type") == "token_expired":
                            item = {
                                **item,
                                "content": {
                                    **(item.get("content") or {}),
                                    "request_id": step_ids[incoming],
                                },
                            }
                            yield events.step_event(incoming + 1, steps[incoming], item)
                            continue
                        backlogs[incoming].append((kind, item))
                    if not backlogs[index]:
                        continue
                    kind, item = backlogs[index].popleft()
                    if kind == "end":
                        break
                    if kind == "error":
                        if isinstance(item, PipelineSuspendedError):
                            observation.status = "suspended"
                            observation.reason = (
                                "Шаг приостановлен. Результат ещё не подтверждён."
                            )
                        else:
                            logger.opt(exception=item).error(
                                "Urban analysis check failed"
                            )
                            observation.status = "failed"
                            observation.reason = "Не удалось выполнить проверку: внутренняя ошибка сервиса."
                        break
                    if observation.observe(item):
                        yield events.step_event(index + 1, step, item)
                    if item.get("type") in events.TERMINAL_STATUSES:
                        break
                summary = observation.summary()
                yield events.step_finished_event(
                    index + 1, step, observation.status, summary
                )
                result.steps.append(
                    events.summary_step(index + 1, step, observation.status, summary)
                )
                report_steps.append(
                    events.summary_step(
                        index + 1,
                        step,
                        observation.status,
                        observation.summary(REPORT_STEP_MAX_CHARS),
                    )
                )
                files.extend(observation.files)
                if observation.status == "completed":
                    result.table_parts.extend(observation.tables)
            result.status = (
                PipelineStatus.DONE
                if any(s["status"] == "completed" for s in result.steps)
                else PipelineStatus.FAILED
            )
            report = ReportResult()
            step = steps[3]
            yield events.step_started_event(4, step, step_ids[3], step.task)
            async for item in generate_report(
                self.llm_client,
                model=model,
                temperature=temperature,
                steps=report_steps,
                zones=result.zones,
                result=report,
            ):
                yield events.step_event(4, step, item)
            result.steps.append(events.summary_step(4, step, "completed", report.text))
            yield events.step_finished_event(4, step, "completed", report.text)
            file = await report_file_event(
                self.file_store,
                token=token,
                scenario_id=scenario_id,
                steps=result.steps,
                text=report.text,
            )
            if file:
                files.append(file)
            yield events.final_event(result.steps)
            for item in files:
                yield item
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_pipeline(
        self,
        index,
        *,
        step,
        request_id,
        idu_mcp_client,
        effects_mcp_client,
        dvd_mcp_client,
        normgraph_mcp_client,
        pzz_mcp_client,
        pzz_inputs,
        token,
        model,
        temperature,
        user_query,
        scenario_id,
        result,
    ):
        common = dict(
            token=token,
            model=model,
            temperature=temperature,
            user_query=step.task,
            scenario_id=scenario_id,
            request_id=request_id,
            persist_history=False,
        )
        if index == 0:
            if normgraph_mcp_client is None:
                yield {
                    "type": "error",
                    "content": {"message": "Сервис NormGraph не настроен."},
                }
                return
            pipeline = self.restriction_service.run_compliance_pipeline(
                mcp_client=idu_mcp_client,
                normgraph_mcp_client=normgraph_mcp_client,
                dvd_mcp_client=dvd_mcp_client,
                conversation_key=None,
                upload_ids=None,
                **common,
            )
        elif index == 1:
            pipeline = self.provision_service.run_provision_pipeline(
                idu_mcp_client=idu_mcp_client,
                effects_mcp_client=effects_mcp_client,
                **common,
            )
        else:
            if pzz_mcp_client is None or self.pzz_service is None:
                yield {
                    "type": "error",
                    "content": {"message": "Сервис ПЗЗ не настроен."},
                }
                return
            zones = await resolve_zones(
                self.urban_api_client,
                self.pzz_service,
                token=token,
                scenario_id=scenario_id,
                model=model,
                user_query=user_query,
                pzz_inputs=pzz_inputs,
            )
            result.zones = zones.report_data()
            if zones.status != "completed":
                yield {
                    "type": (
                        "clarification"
                        if zones.status == "needs_clarification"
                        else "error"
                    ),
                    "content": {"message": zones.reason, "question": zones.reason},
                }
                return
            yield {
                "type": "status",
                "content": {
                    "status": "zones_selected",
                    "text": (
                        f"Функциональные зоны: {zones.inputs.year}, источник {zones.inputs.source}."
                    ),
                    "year": zones.inputs.year,
                    "source": zones.inputs.source,
                },
            }
            pipeline = self.pzz_service.run_pzz_pipeline(
                pzz_mcp_client=pzz_mcp_client,
                pzz_api_client=getattr(pzz_mcp_client, "api_client", None),
                inputs=zones.inputs,
                **common,
            )
        async with aclosing(pipeline):
            async for item in pipeline:
                yield item
