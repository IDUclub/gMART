from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from python_a2a.models.task import TaskState

from src.agents.a2a.a2a_format import sanitized_user_message
from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.common.exceptions.token_exceptions import PipelineSuspendedError
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient

A2AData = dict[str, Any]
A2AEventData = dict[str, Any]


@dataclass(frozen=True)
class ComplianceMcpClients:
    """Scenario layers come from idu_mcp, the norms from NormGraph (optional)."""

    idu: "IduMcpClient"
    normgraph: "NormGraphMcpClient | None"


class ComplianceAgentExecutor:
    """Executor for A2A compliance-check tasks."""

    # None means "whatever the provider serves" — resolved by
    # BaseLlmService.resolve_model, so A2A and REST share one default.
    DEFAULT_MODEL = None
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        restriction_service: RestrictionParserService,
        task_store: A2ATaskStore,
    ) -> None:
        self.restriction_service = restriction_service
        self.task_store = task_store

    async def execute(
        self,
        params: A2AData,
        clients: ComplianceMcpClients,
        token: str,
    ) -> A2AData:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        async for _ in self._run_pipeline(execution, clients, token):
            pass
        return self.task_store.get_task(task["id"]) or task

    async def stream(
        self,
        params: A2AData,
        clients: ComplianceMcpClients,
        token: str,
    ) -> AsyncGenerator[A2AEventData, None]:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        # First frame of a task lifecycle stream is the Task object itself
        # (kind: "task"), per A2A 0.3 SendStreamingMessageSuccessResponse.
        yield task
        async for event in self._run_pipeline(execution, clients, token):
            yield event

    async def _run_pipeline(
        self,
        execution: A2AData,
        clients: ComplianceMcpClients,
        token: str,
    ) -> AsyncGenerator[A2AEventData, None]:
        task_id = execution["task_id"]
        context_id = execution["context_id"]
        layer_count = 0

        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message(
                context_id, task_id, "Запуск проверки нормативного соответствия."
            ),
        )
        yield self._status_update(task_id, context_id, status, final=False)

        try:
            async for item in self.restriction_service.run_compliance_pipeline(
                mcp_client=clients.idu,
                normgraph_mcp_client=clients.normgraph,
                token=token,
                temperature=execution["temperature"],
                model=execution["model"],
                user_query=execution["user_query"],
                scenario_id=execution["scenario_id"],
                # chat_id is passed for read-only history context; A2A tasks
                # must leave no trace in ChatStorage.
                chat_id=execution["chat_id"],
                request_id=task_id,
                persist_history=False,
                # A document choice is answered by the next message of the context.
                conversation_key=f"a2a-compliance:{context_id}",
            ):
                if item.get("type") == "feature_collection":
                    layer_count += 1
                event = self._pipeline_item_to_event(
                    task_id, context_id, item, layer_count
                )
                if event is None:
                    continue
                yield event
                if item.get("type") in {"error", "pipeline_suspended", "clarification"}:
                    return

            status = self.task_store.set_status(
                task_id,
                TaskState.COMPLETED,
                self._agent_message(context_id, task_id, "Проверка завершена."),
            )
            yield self._status_update(task_id, context_id, status, final=True)

        except PipelineSuspendedError:
            return
        except Exception as exc:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id,
                    task_id,
                    f"Сбой проверки нормативного соответствия: {exc}",
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)

    def _pipeline_item_to_event(
        self,
        task_id: str,
        context_id: str,
        item: A2AData,
        layer_number: int,
    ) -> A2AEventData | None:
        item_type = item.get("type")
        content = item.get("content") or {}

        if item_type == "status":
            status = self.task_store.set_status(
                task_id,
                TaskState.WAITING,
                self._agent_message(context_id, task_id, content.get("text", "")),
            )
            return self._status_update(task_id, context_id, status, final=False)

        if item_type == "chunk":
            text = content.get("text") or ""
            if not text:
                return None
            artifact = self._text_artifact(text)
            self.task_store.add_or_append_artifact(task_id, artifact, append=True)
            return self._artifact_update(task_id, context_id, artifact, append=True)

        if item_type == "clarification":
            question = content.get("question") or "Требуется уточнение"
            status = self.task_store.set_status(
                task_id,
                TaskState.INPUT_REQUIRED,
                {
                    **self._agent_message(context_id, task_id, question),
                    "parts": [
                        {"type": "text", "text": question},
                        {"type": "data", "data": content},
                    ],
                },
            )
            return self._status_update(task_id, context_id, status, final=True)

        if item_type == "feature_collection":
            artifact = self._data_artifact(
                f"compliance-layer-{layer_number}",
                content.get("name", "layer"),
                content.get("feature_collection") or {},
                "application/vnd.geo+json",
            )
            self.task_store.add_or_append_artifact(task_id, artifact, append=False)
            return self._artifact_update(task_id, context_id, artifact, append=False)

        if item_type in {"compliance_summary", "file"}:
            artifact = self._data_artifact(
                f"compliance-{item_type.removeprefix('compliance_')}",
                item_type,
                content,
                "application/json",
            )
            self.task_store.add_or_append_artifact(task_id, artifact, append=False)
            return self._artifact_update(task_id, context_id, artifact, append=False)

        if item_type in {"error", "pipeline_suspended"}:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id,
                    task_id,
                    content.get("message", "Ошибка проверки соответствия"),
                ),
            )
            return self._status_update(task_id, context_id, status, final=True)

        # tool_call / check_plan / per-norm results / service events stay internal;
        # the summary artifact carries every per-norm result.
        return None

    def _prepare_execution(self, params: A2AData) -> A2AData:
        message = self._extract_message(params)
        raw_text = self._extract_text(message)
        request_data = self._extract_request_data(params, message)
        # Same scenario-context contract as the restriction agent: structured
        # fields first, then an inline ``scenario_id=...`` that is hidden from the LLM.
        scenario_id = RestrictionAgentExecutor._resolve_scenario_id(
            request_data, raw_text
        )
        user_query = RestrictionAgentExecutor._hide_inline_ids(raw_text)
        if not user_query:
            raise A2AInvalidParamsError("Message text is required")

        task_id = params.get("id") or params.get("taskId") or str(uuid4())
        context_id = (
            params.get("contextId")
            or params.get("context_id")
            or message.get("contextId")
            or str(uuid4())
        )
        chat_id = request_data.get("chat_id") or request_data.get("chatId")

        return {
            "task_id": task_id,
            "context_id": context_id,
            "message": self._sanitize_user_message(message),
            "metadata": request_data,
            "model": request_data.get("model") or self.DEFAULT_MODEL,
            "temperature": float(
                request_data.get("temperature", self.DEFAULT_TEMPERATURE)
            ),
            "scenario_id": scenario_id,
            "chat_id": str(chat_id) if chat_id else None,
            "user_query": user_query,
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
        raise A2AInvalidParamsError("params.message is required")

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
            "model",
            "temperature",
            "scenario_id",
            "scenarioId",
            "chat_id",
            "chatId",
        ):
            if key in params:
                data[key] = params[key]
        return data

    @staticmethod
    def _extract_text(message: A2AData) -> str:
        return "".join(
            str(part["text"])
            for part in message.get("parts", [])
            if isinstance(part, dict) and part.get("text")
        ).strip()

    @staticmethod
    def _sanitize_user_message(message: A2AData) -> A2AData:
        parts = [
            {
                "type": "text",
                "text": RestrictionAgentExecutor._hide_inline_ids(str(part["text"])),
            }
            for part in message.get("parts", [])
            if isinstance(part, dict) and part.get("text")
        ]
        return sanitized_user_message(parts, message.get("messageId"))

    @staticmethod
    def _status_update(
        task_id: str, context_id: str, status: A2AData, final: bool
    ) -> A2AEventData:
        return {
            "kind": "status-update",
            "taskId": task_id,
            "contextId": context_id,
            "status": status,
            "final": final,
        }

    @staticmethod
    def _artifact_update(
        task_id: str, context_id: str, artifact: A2AData, append: bool
    ) -> A2AEventData:
        return {
            "kind": "artifact-update",
            "taskId": task_id,
            "contextId": context_id,
            "artifact": artifact,
            "append": append,
            "lastChunk": not append,
        }

    @staticmethod
    def _agent_message(context_id: str, task_id: str, text: str) -> A2AData:
        return {
            "kind": "message",
            "messageId": str(uuid4()),
            "role": "agent",
            "parts": [{"type": "text", "text": text}],
        }

    @staticmethod
    def _text_artifact(text: str) -> A2AData:
        return {
            "artifactId": "compliance-answer",
            "name": "compliance-response",
            "description": "Итог проверки нормативного соответствия",
            "parts": [{"type": "text", "text": text}],
            "metadata": {"mediaType": "text/plain", "append": True},
        }

    @staticmethod
    def _data_artifact(
        artifact_id: str, name: str, data: A2AData, media_type: str
    ) -> A2AData:
        # Layer names are Russian; ids are numbered so two layers never collide.
        return {
            "artifactId": artifact_id,
            "name": name,
            "parts": [
                {"type": "data", "data": data, "metadata": {"mediaType": media_type}}
            ],
            "metadata": {"mediaType": media_type},
        }
