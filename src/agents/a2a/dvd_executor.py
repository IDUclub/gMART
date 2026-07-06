from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from python_a2a.models.task import TaskState

from src.agents.a2a.a2a_format import sanitized_user_message
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.services.dvd_rag_service import DvdRagService

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient

A2AData = dict[str, Any]
A2AEventData = dict[str, Any]


class DocumentQaAgentExecutor:
    """Executor for A2A regulatory-documents QA (RAG) tasks."""

    DEFAULT_MODEL = "gpt-oss:20b"
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        dvd_rag_service: DvdRagService,
        task_store: A2ATaskStore,
    ) -> None:
        self.dvd_rag_service = dvd_rag_service
        self.task_store = task_store

    async def execute(
        self,
        params: A2AData,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
    ) -> A2AData:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        async for _ in self._run_pipeline(execution, dvd_mcp_client, token):
            pass
        return self.task_store.get_task(task["id"]) or task

    async def stream(
        self,
        params: A2AData,
        dvd_mcp_client: "DvdMcpClient",
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
        async for event in self._run_pipeline(execution, dvd_mcp_client, token):
            yield event

    async def _run_pipeline(
        self,
        execution: A2AData,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
    ) -> AsyncGenerator[A2AEventData, None]:
        task_id = execution["task_id"]
        context_id = execution["context_id"]

        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message(
                context_id, task_id, "Запуск агента по нормативной документации."
            ),
        )
        yield self._status_update(task_id, context_id, status, final=False)

        try:
            async for item in self.dvd_rag_service.run_document_qa_pipeline(
                dvd_mcp_client=dvd_mcp_client,
                token=token,
                model=execution["model"],
                temperature=execution["temperature"],
                user_query=execution["user_query"],
                scenario_id=execution["scenario_id"],
                chat_id=execution["chat_id"],
            ):
                event = self._pipeline_item_to_event(task_id, context_id, item)
                if event is None:
                    continue
                yield event
                if item.get("type") == "error":
                    return

            status = self.task_store.set_status(
                task_id,
                TaskState.COMPLETED,
                self._agent_message(context_id, task_id, "Ответ сформирован."),
            )
            yield self._status_update(task_id, context_id, status, final=True)

        except Exception as exc:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id,
                    task_id,
                    f"Сбой агента по нормативной документации: {exc}",
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)

    def _pipeline_item_to_event(
        self, task_id: str, context_id: str, item: A2AData
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
            iteration = content.get("iteration", 1)
            artifact = self._text_artifact(text, iteration)
            self.task_store.add_or_append_artifact(task_id, artifact, append=True)
            return self._artifact_update(task_id, context_id, artifact, append=True)

        if item_type == "error":
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id, task_id, content.get("message", "Ошибка агента")
                ),
            )
            return self._status_update(task_id, context_id, status, final=True)

        if item_type == "warning":
            status = self.task_store.set_status(
                task_id,
                TaskState.WAITING,
                self._agent_message(context_id, task_id, content.get("message", "")),
            )
            return self._status_update(task_id, context_id, status, final=False)

        # tool_call / service_event / pipeline_started are internal — not surfaced here.
        return None

    def _prepare_execution(self, params: A2AData) -> A2AData:
        message = self._extract_message(params)
        user_query = self._extract_text(message)
        if not user_query:
            raise A2AInvalidParamsError("Message text is required")
        request_data = self._extract_request_data(params, message)

        task_id = params.get("id") or params.get("taskId") or str(uuid4())
        context_id = (
            params.get("contextId")
            or params.get("context_id")
            or message.get("contextId")
            or str(uuid4())
        )
        scenario_id = request_data.get("scenario_id") or request_data.get("scenarioId")
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
            "scenario_id": int(scenario_id) if scenario_id is not None else None,
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
            {"type": "text", "text": str(part["text"])}
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
    def _text_artifact(text: str, iteration: int) -> A2AData:
        # A new artifactId per draft iteration so a rejected draft is not concatenated
        # with the next one; the final accepted answer is the last iteration's artifact.
        return {
            "artifactId": f"document-qa-answer-{iteration}",
            "name": "document-qa-response",
            "description": "Ответ агента по нормативной документации",
            "parts": [{"type": "text", "text": text}],
            "metadata": {
                "mediaType": "text/plain",
                "iteration": iteration,
                "append": True,
            },
        }
