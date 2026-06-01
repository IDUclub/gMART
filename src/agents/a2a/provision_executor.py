from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from python_a2a.models.task import TaskState

from src.agents.a2a.task_store import A2ATaskStore
from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.provsion_service import ProvisionService

A2AData = dict[str, Any]
A2AEventData = dict[str, Any]


class ProvisionAgentExecutor:
    """Executor for A2A provision effects tasks."""

    DEFAULT_MODEL = "gpt-oss:20b"
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        provision_service: ProvisionService,
        task_store: A2ATaskStore,
    ) -> None:
        self.provision_service = provision_service
        self.task_store = task_store

    async def execute(
        self,
        params: A2AData,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
    ) -> A2AData:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        async for _ in self._run_pipeline(
            execution, idu_mcp_client, effects_mcp_client
        ):
            pass
        return self.task_store.get_task(task["id"]) or task

    async def stream(
        self,
        params: A2AData,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
    ) -> AsyncGenerator[A2AEventData, None]:
        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        yield {"task": task}
        async for event in self._run_pipeline(
            execution, idu_mcp_client, effects_mcp_client
        ):
            yield event

    async def _run_pipeline(
        self,
        execution: A2AData,
        idu_mcp_client: IduMcpClient,
        effects_mcp_client: EffectsMcpClient,
    ) -> AsyncGenerator[A2AEventData, None]:
        task_id = execution["task_id"]
        context_id = execution["context_id"]
        emitted_text = False

        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message(
                context_id, task_id, "Starting the provision effects pipeline."
            ),
        )
        yield self._status_update(task_id, context_id, status, final=False)

        try:
            async for item in self.provision_service.run_provision_pipeline(
                idu_mcp_client=idu_mcp_client,
                effects_mcp_client=effects_mcp_client,
                temperature=execution["temperature"],
                model=execution["model"],
                user_query=execution["user_query"],
                scenario_id=execution["scenario_id"],
            ):
                event = self._pipeline_item_to_event(task_id, context_id, item)
                if event is None:
                    continue
                if "artifactUpdate" in event:
                    emitted_text = (
                        emitted_text
                        or event["artifactUpdate"]["artifact"].get("artifactId")
                        == "provision-agent-text"
                    )
                yield event
                if item.get("type") == "error":
                    return

            status = self.task_store.set_status(
                task_id,
                TaskState.COMPLETED,
                self._agent_message(
                    context_id, task_id, "Provision effects pipeline completed."
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)

        except Exception as exc:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id, task_id, f"Provision effects pipeline failed: {exc}"
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)
            if not emitted_text:
                artifact = self._text_artifact(
                    "Provision effects pipeline failed.", append=False
                )
                self.task_store.add_or_append_artifact(task_id, artifact, append=False)
                yield self._artifact_update(task_id, context_id, artifact, append=False)

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
            artifact = self._text_artifact(text, append=True)
            self.task_store.add_or_append_artifact(task_id, artifact, append=True)
            return self._artifact_update(task_id, context_id, artifact, append=True)

        if item_type == "feature_collection":
            artifact = self._geojson_artifact(
                content.get("name", "layer"),
                content.get("feature_collection") or {},
            )
            self.task_store.add_or_append_artifact(task_id, artifact, append=False)
            return self._artifact_update(task_id, context_id, artifact, append=False)

        if item_type == "error":
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id,
                    task_id,
                    content.get("message", "Provision pipeline error"),
                ),
            )
            return self._status_update(task_id, context_id, status, final=True)

        return None

    def _prepare_execution(self, params: A2AData) -> A2AData:
        message = self._extract_message(params)
        user_query = self._extract_text(message)
        request_data = self._extract_request_data(params, message)

        project_id = request_data.get("project_id")
        if project_id is None:
            raise ValueError("project_id is required in params.metadata or a data part")

        raw_text = request_data.get("request") or user_query
        scenario_id = self._extract_scenario_id_from_text(str(raw_text))
        if scenario_id is None:
            raise ValueError(
                "scenario_id is required in the request text (e.g. scenario_id=772)"
            )

        user_query = self._hide_inline_ids(str(raw_text))
        if not user_query:
            raise ValueError("User message text is required")

        task_id = params.get("id") or params.get("taskId") or str(uuid4())
        context_id = (
            params.get("contextId")
            or params.get("context_id")
            or message.get("contextId")
            or str(uuid4())
        )

        message = self._sanitize_user_message(message)

        return {
            "task_id": task_id,
            "context_id": context_id,
            "message": message,
            "metadata": request_data,
            "model": request_data.get("model") or self.DEFAULT_MODEL,
            "temperature": float(
                request_data.get("temperature", self.DEFAULT_TEMPERATURE)
            ),
            "scenario_id": int(scenario_id),
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
        raise ValueError("params.message is required")

    @staticmethod
    def _extract_scenario_id_from_text(text: str) -> int | None:
        match = re.search(
            r"\b(?:scenario_id|scenarioId)\b\s*[:=]\s*(\d+)", text, re.IGNORECASE
        )
        return int(match.group(1)) if match else None

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
            "project_id",
            "projectId",
            "model",
            "temperature",
            "request",
            "userQuery",
        ):
            if key in params:
                data[key] = params[key]

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
    def _sanitize_user_message(message: A2AData) -> A2AData:
        return {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": ProvisionAgentExecutor._hide_inline_ids(str(part["text"])),
                }
                for part in message.get("parts", [])
                if isinstance(part, dict) and part.get("text")
            ],
        }

    @staticmethod
    def _hide_inline_ids(text: str) -> str:
        text = re.sub(r"\b[\w-]*id\b\s*[:=]\s*[\w-]+", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", text).strip()

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
    def _artifact_update(
        task_id: str, context_id: str, artifact: A2AData, append: bool
    ) -> A2AEventData:
        return {
            "artifactUpdate": {
                "taskId": task_id,
                "contextId": context_id,
                "artifact": artifact,
                "append": append,
                "lastChunk": not append,
            }
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
    def _text_artifact(text: str, append: bool) -> A2AData:
        return {
            "artifactId": "provision-agent-text",
            "name": "provision-agent-response",
            "description": "Text response from the provision effects agent",
            "parts": [{"type": "text", "text": text}],
            "metadata": {"mediaType": "text/plain", "append": append},
        }

    @staticmethod
    def _geojson_artifact(layer_name: str, feature_collection: A2AData) -> A2AData:
        safe_name = (
            re.sub(r"[^a-zA-Z0-9_-]+", "-", layer_name).strip("-").lower() or "layer"
        )
        return {
            "artifactId": f"geojson-{safe_name}",
            "name": layer_name,
            "description": "GeoJSON layer produced by the provision effects pipeline",
            "parts": [
                {
                    "type": "data",
                    "data": feature_collection,
                    "mediaType": "application/vnd.geo+json",
                }
            ],
            "metadata": {
                "layerName": layer_name,
                "mediaType": "application/vnd.geo+json",
            },
        }
