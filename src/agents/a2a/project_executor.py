from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from pydantic import ValidationError
from python_a2a.models.task import TaskState

from src.agents.a2a.a2a_format import sanitized_user_message
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import A2AInvalidParamsError
from src.agents.common.exceptions.base_exceptions import AgentsBaseException
from src.agents.dto.project_request_dto import ProjectInputs
from src.agents.services.compilance.compliance_uploads import LAYER_ROLES
from src.agents.services.project_files.project_files_service import (
    ProjectFilesService,
)
from src.common.service_auth import user_id_from_jwt

A2AData = dict[str, Any]
A2AEventData = dict[str, Any]
_ARTIFACT_EVENTS = {
    "project_created",
    "scenario_created",
    "project_ready",
    "project_summary",
}


class ProjectAgentExecutor:
    """Executor for A2A tasks that create a project from files."""

    DEFAULT_MODEL = None
    DEFAULT_TEMPERATURE = 1.0

    def __init__(self, project_service: ProjectFilesService, task_store: A2ATaskStore):
        self.project_service = project_service
        self.task_store = task_store

    async def execute(self, params: A2AData, token: str) -> A2AData:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        async for _ in self._run_pipeline(execution, token):
            pass
        return self.task_store.get_task(task["id"]) or task

    async def stream(
        self, params: A2AData, token: str
    ) -> AsyncGenerator[A2AEventData, None]:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        yield task
        async for event in self._run_pipeline(execution, token):
            yield event

    async def _run_pipeline(
        self, execution: A2AData, token: str
    ) -> AsyncGenerator[A2AEventData, None]:
        task_id = execution["task_id"]
        context_id = execution["context_id"]
        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message("Запуск агента создания проекта из файлов."),
        )
        yield self._status_update(task_id, context_id, status, final=False)
        try:
            upload_ids = list(execution["upload_ids"])
            if execution["inline_files"]:
                upload_ids += await self._store_inline_files(
                    execution["inline_files"], token
                )
            async for item in self.project_service.run_project_pipeline(
                token=token,
                model=execution["model"],
                temperature=execution["temperature"],
                user_query=execution["user_query"],
                chat_id=execution["chat_id"],
                request_id=task_id,
                # A2A tasks leave no trace in ChatStorage; the question of a task
                # is answered by the next message of its context.
                persist_history=False,
                conversation_key=f"a2a-project:{context_id}",
                upload_ids=upload_ids or None,
                inputs=execution["inputs"],
            ):
                event = self._pipeline_item_to_event(task_id, context_id, item)
                if event is None:
                    continue
                yield event
                if item.get("type") in {"error", "clarification"}:
                    return
            status = self.task_store.set_status(
                task_id, TaskState.COMPLETED, self._agent_message("Готово.")
            )
            yield self._status_update(task_id, context_id, status, final=True)
        except Exception as exc:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(f"Сбой агента создания проекта: {exc}"),
            )
            yield self._status_update(task_id, context_id, status, final=True)

    def _pipeline_item_to_event(
        self, task_id: str, context_id: str, item: A2AData
    ) -> A2AEventData | None:
        item_type = item.get("type")
        content = item.get("content") or {}
        if item_type in {"status", "warning"}:
            status = self.task_store.set_status(
                task_id,
                TaskState.WAITING,
                self._agent_message(content.get("text") or content.get("message", "")),
            )
            return self._status_update(task_id, context_id, status, final=False)
        if item_type == "chunk":
            text = content.get("text") or ""
            if not text:
                return None
            artifact = {
                "artifactId": "project-answer",
                "name": "project-response",
                "description": "Ответ агента создания проекта",
                "parts": [{"type": "text", "text": text + "\n\n"}],
                "metadata": {"mediaType": "text/plain", "append": True},
            }
            self.task_store.add_or_append_artifact(task_id, artifact, append=True)
            return self._artifact_update(task_id, context_id, artifact, append=True)
        if item_type == "error":
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(content.get("message", "Ошибка агента")),
            )
            return self._status_update(task_id, context_id, status, final=True)
        if item_type == "clarification":
            question = content.get("question") or "Требуется уточнение"
            status = self.task_store.set_status(
                task_id,
                TaskState.INPUT_REQUIRED,
                {
                    **self._agent_message(question),
                    "parts": [
                        {"type": "text", "text": question},
                        {"type": "data", "data": content},
                    ],
                },
            )
            return self._status_update(task_id, context_id, status, final=True)
        if item_type in _ARTIFACT_EVENTS:
            artifact = {
                "artifactId": f"project-{item_type}",
                "name": item_type,
                "parts": [{"type": "data", "data": content}],
            }
            self.task_store.add_or_append_artifact(task_id, artifact, append=False)
            return self._artifact_update(task_id, context_id, artifact, append=False)
        return None

    async def _store_inline_files(self, files: list[A2AData], token: str) -> list[str]:
        """Store GeoJSON layers passed in the message as uploads of the caller."""
        try:
            owner = user_id_from_jwt(token)
        except ValueError as exc:
            raise A2AInvalidParamsError(
                "Inline files need a caller token with a user id"
            ) from exc
        upload_ids = []
        for index, item in enumerate(files, start=1):
            name = str(item.get("name") or f"layer_{index}")
            if not name.casefold().endswith((".geojson", ".json")):
                name += ".geojson"
            try:
                summary = await self.project_service.uploads.create(
                    owner,
                    [(name, json.dumps(item["geojson"]).encode("utf-8"))],
                    role=item.get("role"),
                )
            except AgentsBaseException as exc:
                raise A2AInvalidParamsError(f"files[{index - 1}]: {exc.message}")
            upload_ids.append(summary["upload_id"])
        return upload_ids

    def _prepare_execution(self, params: A2AData) -> A2AData:
        message = self._extract_message(params)
        user_query = self._extract_text(message)
        if not user_query:
            raise A2AInvalidParamsError("Message text is required")
        data = self._extract_request_data(params, message)
        task_id = params.get("id") or params.get("taskId") or str(uuid4())
        context_id = (
            params.get("contextId")
            or params.get("context_id")
            or message.get("contextId")
            or str(uuid4())
        )
        upload_ids = data.get("upload_ids") or []
        if not isinstance(upload_ids, list) or not all(
            isinstance(item, str) for item in upload_ids
        ):
            raise A2AInvalidParamsError("upload_ids must be a list of strings")
        try:
            inputs = ProjectInputs.model_validate(
                data.get("inputs")
                or {
                    key: value
                    for key, value in data.items()
                    if key in ProjectInputs.model_fields
                }
            )
        except ValidationError as exc:
            raise A2AInvalidParamsError(f"inputs: {exc.errors()[0]['msg']}") from exc
        chat_id = data.get("chat_id") or data.get("chatId")
        metadata = {key: value for key, value in data.items() if key != "files"}
        return {
            "task_id": task_id,
            "context_id": context_id,
            "message": sanitized_user_message(
                [
                    {"type": "text", "text": str(part["text"])}
                    for part in message.get("parts", [])
                    if isinstance(part, dict) and part.get("text")
                ],
                message.get("messageId"),
            ),
            "metadata": metadata,
            "model": data.get("model") or self.DEFAULT_MODEL,
            "temperature": float(data.get("temperature", self.DEFAULT_TEMPERATURE)),
            "chat_id": str(chat_id) if chat_id else None,
            "user_query": user_query,
            "upload_ids": upload_ids,
            "inline_files": self._inline_files(data),
            "inputs": inputs,
        }

    @staticmethod
    def _inline_files(data: A2AData) -> list[A2AData]:
        files = data.get("files") or []
        if not isinstance(files, list):
            raise A2AInvalidParamsError("files must be a list of {name, role, geojson}")
        for index, item in enumerate(files):
            if not isinstance(item, dict) or not isinstance(item.get("geojson"), dict):
                raise A2AInvalidParamsError(f"files[{index}].geojson is required")
            if item.get("role") is not None and item["role"] not in LAYER_ROLES:
                raise A2AInvalidParamsError(
                    f"files[{index}].role must be one of {', '.join(LAYER_ROLES)}"
                )
        return files

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
            "inputs",
            "upload_ids",
            "files",
            *ProjectInputs.model_fields,
            "model",
            "temperature",
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
    def _agent_message(text: str) -> A2AData:
        return {
            "kind": "message",
            "messageId": str(uuid4()),
            "role": "agent",
            "parts": [{"type": "text", "text": text}],
        }
