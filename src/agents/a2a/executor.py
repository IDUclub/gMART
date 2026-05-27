from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from python_a2a.models.task import TaskState

from src.agents.a2a.task_store import A2ATaskStore
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)

A2AData = dict[str, Any]
A2AEventData = dict[str, Any]


class RestrictionAgentExecutor:
    """
    Executor for A2A restriction creation tasks.
    Attributes:
        DEFAULT_MODEL (str): Default model for restriction pipeline.
        DEFAULT_TEMPERATURE (float): Default LLM temperature.
        restriction_service (RestrictionParserService): Restriction pipeline service.
        task_store (A2ATaskStore): A2A task storage.
    """

    DEFAULT_MODEL = "gpt-oss:20b"
    DEFAULT_TEMPERATURE = 1.0

    def __init__(
        self,
        restriction_service: RestrictionParserService,
        task_store: A2ATaskStore,
    ) -> None:
        """
        RestrictionAgentExecutor initialization function.
        Args:
            restriction_service (RestrictionParserService): Restriction pipeline service.
            task_store (A2ATaskStore): A2A task storage.
        """

        self.restriction_service = restriction_service
        self.task_store = task_store

    async def execute(
        self,
        params: A2AData,
        mcp_client: IduMcpClient,
    ) -> A2AData:
        """
        Function executes restriction pipeline and returns final A2A task.
        Args:
            params (A2AData): A2A method params.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Returns:
            A2AData: Final serialized A2A task.
        """

        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )

        async for _ in self._run_pipeline(execution, mcp_client):
            pass

        return self.task_store.get_task(task["id"]) or task

    async def stream(
        self,
        params: A2AData,
        mcp_client: IduMcpClient,
    ) -> AsyncGenerator[A2AEventData, None]:
        """
        Function executes restriction pipeline and streams A2A events.
        Args:
            params (A2AData): A2A method params.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Yields:
            A2AEventData: A2A task, status update or artifact update event.
        """

        execution = self._prepare_execution(params)
        task = self.task_store.create_task(
            execution["task_id"],
            execution["context_id"],
            execution["message"],
            execution["metadata"],
        )
        yield {"task": task}

        async for event in self._run_pipeline(execution, mcp_client):
            yield event

    async def _run_pipeline(
        self,
        execution: A2AData,
        mcp_client: IduMcpClient,
    ) -> AsyncGenerator[A2AEventData, None]:
        """
        Function runs restriction pipeline and converts chunks to A2A events.
        Args:
            execution (A2AData): Prepared execution context.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Yields:
            A2AEventData: A2A status update or artifact update event.
        """

        task_id = execution["task_id"]
        context_id = execution["context_id"]
        emitted_text = False

        status = self.task_store.set_status(
            task_id,
            TaskState.WAITING,
            self._agent_message(
                context_id,
                task_id,
                "Starting the restriction creation pipeline.",
            ),
        )
        yield self._status_update(task_id, context_id, status, final=False)

        try:
            async for (
                item
            ) in self.restriction_service.run_restriction_execution_pipline(
                mcp_client=mcp_client,
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
                        == "restriction-agent-text"
                    )
                yield event
                if item.get("type") == "error":
                    return

            status = self.task_store.set_status(
                task_id,
                TaskState.COMPLETED,
                self._agent_message(
                    context_id,
                    task_id,
                    "Restriction creation pipeline completed.",
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)

        except Exception as exc:
            status = self.task_store.set_status(
                task_id,
                TaskState.FAILED,
                self._agent_message(
                    context_id,
                    task_id,
                    f"Restriction creation pipeline failed: {exc}",
                ),
            )
            yield self._status_update(task_id, context_id, status, final=True)
            if not emitted_text:
                artifact = self._text_artifact(
                    "Restriction creation pipeline failed.",
                    append=False,
                )
                self.task_store.add_or_append_artifact(task_id, artifact, append=False)
                yield self._artifact_update(task_id, context_id, artifact, append=False)

    def _pipeline_item_to_event(
        self,
        task_id: str,
        context_id: str,
        item: A2AData,
    ) -> A2AEventData | None:
        """
        Function converts restriction pipeline item to A2A event.
        Args:
            task_id (str): A2A task id.
            context_id (str): A2A context id.
            item (A2AData): Restriction pipeline item.
        Returns:
            A2AEventData | None: A2A event or None for empty chunks.
        """

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
                    content.get("message", "Restriction pipeline error"),
                ),
            )
            return self._status_update(task_id, context_id, status, final=True)

        return None

    def _prepare_execution(self, params: A2AData) -> A2AData:
        """
        Function extracts execution context from A2A params.
        Args:
            params (A2AData): A2A method params.
        Returns:
            A2AData: Prepared execution context.
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
        user_query = self._hide_inline_ids(str(user_query))
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
        """
        Function extracts message from A2A params.
        Args:
            params (A2AData): A2A method params.
        Returns:
            A2AData: A2A message.
        """

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
    def _extract_request_data(
        params: A2AData,
        message: A2AData,
    ) -> A2AData:
        """
        Function extracts business metadata from A2A params and message.
        Args:
            params (A2AData): A2A method params.
            message (A2AData): A2A message.
        Returns:
            A2AData: Business metadata.
        """

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
            "model",
            "temperature",
            "request",
            "userQuery",
        ):
            if key in params:
                data[key] = params[key]

        if "scenarioId" in data and "scenario_id" not in data:
            data["scenario_id"] = data["scenarioId"]
        if "userQuery" in data and "request" not in data:
            data["request"] = data["userQuery"]

        return data

    @staticmethod
    def _extract_text(message: A2AData) -> str:
        """
        Function extracts text from A2A message parts.
        Args:
            message (A2AData): A2A message.
        Returns:
            str: Message text.
        """

        text_parts = []
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                text_parts.append(str(part["text"]))
        return "".join(text_parts).strip()

    @staticmethod
    def _sanitize_user_message(message: A2AData) -> A2AData:
        """
        Function removes ids and metadata from message stored in task history.
        Args:
            message (A2AData): Raw A2A message.
        Returns:
            A2AData: Sanitized user message.
        """

        return {
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": RestrictionAgentExecutor._hide_inline_ids(
                        str(part["text"])
                    ),
                }
                for part in message.get("parts", [])
                if isinstance(part, dict) and part.get("text")
            ],
        }

    @staticmethod
    def _hide_inline_ids(text: str) -> str:
        """
        Function removes inline id-like assignments from user text.
        Args:
            text (str): User text.
        Returns:
            str: Sanitized text.
        """

        text = re.sub(r"\b[\w-]*id\b\s*[:=]\s*[\w-]+", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s{2,}", " ", text).strip()

    @staticmethod
    def _status_update(
        task_id: str,
        context_id: str,
        status: A2AData,
        final: bool,
    ) -> A2AEventData:
        """
        Function builds A2A status update event.
        Args:
            task_id (str): A2A task id.
            context_id (str): A2A context id.
            status (A2AData): A2A task status.
            final (bool): Whether event is final.
        Returns:
            A2AEventData: A2A status update event.
        """

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
        task_id: str,
        context_id: str,
        artifact: A2AData,
        append: bool,
    ) -> A2AEventData:
        """
        Function builds A2A artifact update event.
        Args:
            task_id (str): A2A task id.
            context_id (str): A2A context id.
            artifact (A2AData): A2A artifact.
            append (bool): Whether artifact should be appended.
        Returns:
            A2AEventData: A2A artifact update event.
        """

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
        """
        Function builds A2A agent message.
        Args:
            context_id (str): A2A context id.
            task_id (str): A2A task id.
            text (str): Message text.
        Returns:
            A2AData: A2A agent message.
        """

        return {
            "kind": "message",
            "messageId": str(uuid4()),
            "role": "agent",
            "parts": [{"type": "text", "text": text}],
        }

    @staticmethod
    def _text_artifact(text: str, append: bool) -> A2AData:
        """
        Function builds text A2A artifact.
        Args:
            text (str): Artifact text.
            append (bool): Whether artifact should be appended.
        Returns:
            A2AData: Text artifact.
        """

        return {
            "artifactId": "restriction-agent-text",
            "name": "restriction-agent-response",
            "description": "Text response from the restriction creation agent",
            "parts": [{"type": "text", "text": text}],
            "metadata": {
                "mediaType": "text/plain",
                "append": append,
            },
        }

    @staticmethod
    def _geojson_artifact(layer_name: str, feature_collection: A2AData) -> A2AData:
        """
        Function builds GeoJSON A2A artifact.
        Args:
            layer_name (str): GIS layer name.
            feature_collection (A2AData): GeoJSON feature collection.
        Returns:
            A2AData: GeoJSON artifact.
        """

        safe_name = (
            re.sub(r"[^a-zA-Z0-9_-]+", "-", layer_name).strip("-").lower() or "layer"
        )
        return {
            "artifactId": f"geojson-{safe_name}",
            "name": layer_name,
            "description": "GeoJSON layer produced by the restriction creation pipeline",
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
