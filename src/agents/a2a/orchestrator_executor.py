from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from loguru import logger
from python_a2a.models.task import TaskState

from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.provision_executor import ProvisionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.orchestrator_service import (
    OrchestratorIntent,
    OrchestratorService,
)

A2AData = dict[str, Any]
A2AEventData = dict[str, Any]


class OrchestratorAgentExecutor:
    """
    Executor for the orchestrator A2A agent.
    Classifies user intent via LLM and delegates to restriction and/or provision executors.

    Stream order per request:
      1. Orchestrator task creation event
      2. StatusUpdate: WAITING — "Определяю тип запроса..."
      3. LLM classification (hidden from stream)
      4. StatusUpdate: WAITING — routing notification
      5. All events from RestrictionAgentExecutor (if needs_restriction)
      6. All events from ProvisionAgentExecutor  (if needs_provision)
      7. StatusUpdate: COMPLETED / FAILED on the orchestrator task

    Attributes:
        DEFAULT_MODEL (str): Default Ollama model name.
        DEFAULT_TEMPERATURE (float): Default LLM sampling temperature.
        orchestrator_service (OrchestratorService): LLM-based intent classifier.
        restriction_executor (RestrictionAgentExecutor): Restriction sub-agent executor.
        provision_executor (ProvisionAgentExecutor): Provision sub-agent executor.
        task_store (A2ATaskStore): Task store for orchestrator-level tasks only.
    """

    DEFAULT_MODEL = "gpt-oss:20b"
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        orchestrator_service: OrchestratorService,
        restriction_executor: RestrictionAgentExecutor,
        provision_executor: ProvisionAgentExecutor,
        task_store: A2ATaskStore,
    ) -> None:
        """
        OrchestratorAgentExecutor initialization function.
        Args:
            orchestrator_service (OrchestratorService): LLM intent classifier.
            restriction_executor (RestrictionAgentExecutor): Restriction sub-agent executor.
            provision_executor (ProvisionAgentExecutor): Provision sub-agent executor.
            task_store (A2ATaskStore): Task store for orchestrator tasks.
        """
        self.orchestrator_service = orchestrator_service
        self.restriction_executor = restriction_executor
        self.provision_executor = provision_executor
        self.task_store = task_store

    async def execute(
        self,
        params: A2AData,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
    ) -> A2AData:
        """
        Non-streaming execution: drain the stream and return the final orchestrator task.
        Args:
            params (A2AData): A2A method params.
            idu_mcp_client (IduMcpClient): MCP client for IDU geospatial tools.
            effects_mcp_client (EffectsMcpClient): MCP client for effects calculation.
        Returns:
            A2AData: Final serialized orchestrator task.
        """
        task_id = ""
        async for event in self.stream(params, idu_mcp_client, effects_mcp_client):
            if "task" in event:
                task_id = event["task"].get("id", "")
        return self.task_store.get_task(task_id) or {}

    async def stream(
        self,
        params: A2AData,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
    ) -> AsyncGenerator[A2AEventData, None]:
        """
        Streaming execution: yield orchestrator status events and all sub-agent events.
        Args:
            params (A2AData): A2A method params.
            idu_mcp_client (IduMcpClient): MCP client for IDU geospatial tools.
            effects_mcp_client (EffectsMcpClient): MCP client for effects calculation.
        Yields:
            A2AEventData: A2A task, status update, or artifact update event.
        """
        execution = self._prepare_execution(params)
        task_id = execution["task_id"]
        context_id = execution["context_id"]

        task = self.task_store.create_task(
            task_id,
            context_id,
            execution["message"],
            execution["metadata"],
        )
        yield {"task": task}

        # ── Phase 1: intent classification ───────────────────────────────────
        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message(context_id, task_id, "Определяю тип запроса..."),
        )
        yield self._status_update(task_id, context_id, status, final=False)

        try:
            intent = await self.orchestrator_service.classify_intent(
                execution["user_query"], execution["model"]
            )
        except Exception as exc:
            logger.warning(f"OrchestratorExecutor: classification error: {exc}")
            intent = OrchestratorIntent(needs_restriction=True, needs_provision=False)

        if intent.is_empty:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id,
                    task_id,
                    "Запрос не относится к геопространственному анализу. "
                    "Пожалуйста, уточните задачу: задайте зоны ограничений "
                    "или запросите расчёт обеспеченности сервисами.",
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)
            return

        # ── Phase 1.5: query decomposition (compound requests only) ──────────
        if intent.is_compound:
            status = self.task_store.set_status(
                task_id,
                TaskState.WAITING,
                self._agent_message(
                    context_id,
                    task_id,
                    "Составной запрос: разбиваю на подзадачи для каждого агента...",
                ),
            )
            yield self._status_update(task_id, context_id, status, final=False)
            restriction_query, provision_query = (
                await self.orchestrator_service.decompose_query(
                    execution["user_query"], execution["model"]
                )
            )
            intent.restriction_query = restriction_query
            intent.provision_query = provision_query

        # ── Phase 2: routing notification ─────────────────────────────────────
        labels = []
        if intent.needs_restriction:
            labels.append("restriction-creation-agent")
        if intent.needs_provision:
            labels.append("provision-effects-agent")
        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message(
                context_id,
                task_id,
                f"Перенаправляю запрос: {', '.join(labels)}.",
            ),
        )
        yield self._status_update(task_id, context_id, status, final=False)

        # ── Phase 3: delegation ───────────────────────────────────────────────
        try:
            if intent.needs_restriction:
                async for event in self.restriction_executor.stream(
                    self._make_sub_params(execution, query=intent.restriction_query),
                    idu_mcp_client,
                ):
                    yield event

            if intent.needs_provision:
                if execution.get("project_id") is None:
                    status = self.task_store.set_status(
                        task_id,
                        TaskState.WAITING,
                        self._agent_message(
                            context_id,
                            task_id,
                            "Расчёт обеспеченности пропущен: параметр project_id не передан. "
                            "Укажите project_id в запросе для получения эффектов обеспеченности.",
                        ),
                    )
                    yield self._status_update(task_id, context_id, status, final=False)
                else:
                    async for event in self.provision_executor.stream(
                        self._make_sub_params(execution, query=intent.provision_query),
                        idu_mcp_client,
                        effects_mcp_client,
                    ):
                        yield event

        except Exception as exc:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id, task_id, f"Ошибка при выполнении агента: {exc}"
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)
            return

        # ── Phase 4: final orchestrator status ────────────────────────────────
        status = self.task_store.set_status(
            task_id,
            TaskState.COMPLETED,
            self._agent_message(context_id, task_id, "Оркестрация завершена."),
        )
        yield self._status_update(task_id, context_id, status, final=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _prepare_execution(self, params: A2AData) -> A2AData:
        """
        Extract and validate orchestrator execution context from raw A2A params.
        Mirrors the _prepare_execution pattern used in RestrictionAgentExecutor
        and ProvisionAgentExecutor.
        Args:
            params (A2AData): Raw A2A method params.
        Returns:
            A2AData: Validated execution context dict.
        Raises:
            ValueError: If scenario_id or user query is missing.
        """
        message = self._extract_message(params)
        user_query = self._extract_text(message)
        request_data = self._extract_request_data(params, message)

        scenario_id = request_data.get("scenario_id")
        if scenario_id is None:
            raise ValueError(
                "scenario_id is required in params.metadata or a data part"
            )

        user_query = request_data.get("request") or user_query
        user_query = re.sub(
            r"\b[\w-]*id\b\s*[:=]\s*[\w-]+", "", user_query, flags=re.IGNORECASE
        )
        user_query = re.sub(r"\s{2,}", " ", user_query).strip()
        if not user_query:
            raise ValueError("User message text is required")

        task_id = str(params.get("id") or params.get("taskId") or uuid4())
        context_id = str(
            params.get("contextId")
            or params.get("context_id")
            or message.get("contextId")
            or uuid4()
        )
        model = str(request_data.get("model") or self.DEFAULT_MODEL)
        temperature = float(request_data.get("temperature", self.DEFAULT_TEMPERATURE))
        raw_project_id = request_data.get("project_id")
        project_id = int(raw_project_id) if raw_project_id is not None else None

        return {
            "task_id": task_id,
            "context_id": context_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": user_query}],
            },
            "metadata": request_data,
            "model": model,
            "temperature": temperature,
            "scenario_id": int(scenario_id),
            "project_id": project_id,
            "user_query": user_query,
        }

    @staticmethod
    def _make_sub_params(execution: A2AData, *, query: str | None = None) -> A2AData:
        """
        Build A2A params for a sub-executor call with a fresh task ID.

        ``query`` overrides the user-visible text passed to the sub-executor.
        For compound requests it carries the decomposed sub-query so each agent
        only sees context relevant to its own task.  When ``None`` the original
        ``execution["user_query"]`` is used unchanged.

        scenario_id and token values are forwarded opaquely — they are never
        surfaced in LLM prompts; the sub-executors extract them from metadata.

        Args:
            execution (A2AData): Validated orchestrator execution context.
            query (str | None): Focused sub-query for this specific sub-executor.
                Falls back to the original user query when not provided.
        Returns:
            A2AData: Sub-params dict ready for RestrictionAgentExecutor / ProvisionAgentExecutor.
        """
        effective_query = query or execution["user_query"]
        metadata: dict[str, Any] = {
            "scenario_id": execution["scenario_id"],
            "request": effective_query,
            "model": execution["model"],
            "temperature": execution["temperature"],
        }
        if execution.get("project_id") is not None:
            metadata["project_id"] = execution["project_id"]
        return {
            "id": str(uuid4()),
            "contextId": execution["context_id"],
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": effective_query}],
                "metadata": metadata,
            },
        }

    @staticmethod
    def _extract_message(params: A2AData) -> A2AData:
        message = params.get("message")
        if isinstance(message, dict):
            return dict(message)
        direct_text = params.get("request") or params.get("text")
        if direct_text:
            return {
                "role": "user",
                "parts": [{"type": "text", "text": str(direct_text)}],
            }
        raise ValueError("params.message is required")

    @staticmethod
    def _extract_request_data(params: A2AData, message: A2AData) -> A2AData:
        data: dict[str, Any] = {}
        for source in (params.get("metadata"), message.get("metadata")):
            if isinstance(source, dict):
                data.update(source)
        for part in message.get("parts", []):
            part_data = part.get("data") if isinstance(part, dict) else None
            if isinstance(part_data, dict):
                data.update(part_data)
        for key in (
            "scenario_id",
            "scenarioId",
            "project_id",
            "projectId",
            "model",
            "temperature",
            "request",
            "userQuery",
        ):
            if key in params:
                data[key] = params[key]
        if "scenarioId" in data and "scenario_id" not in data:
            data["scenario_id"] = data["scenarioId"]
        if "projectId" in data and "project_id" not in data:
            data["project_id"] = data["projectId"]
        if "userQuery" in data and "request" not in data:
            data["request"] = data["userQuery"]
        return data

    @staticmethod
    def _extract_text(message: A2AData) -> str:
        return "".join(
            str(part["text"])
            for part in message.get("parts", [])
            if isinstance(part, dict) and part.get("text")
        ).strip()

    @staticmethod
    def _status_update(
        task_id: str, context_id: str, status: A2AData, final: bool
    ) -> A2AEventData:
        return {
            "statusUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "status": status,
                "final": final,
            }
        }

    @staticmethod
    def _agent_message(context_id: str, task_id: str, text: str) -> A2AData:
        return {
            "role": "agent",
            "kind": "message",
            "parts": [{"type": "text", "text": text}],
        }
