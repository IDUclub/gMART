from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from python_a2a.models.task import TaskState

from src.agents.a2a.a2a_format import (
    apply_history_length,
    normalize_response,
    utc_now_rfc3339,
)
from src.agents.a2a.dvd_agent import DocumentQaA2AAgent
from src.agents.a2a.dvd_executor import DocumentQaAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import (
    A2AInvalidParamsError,
    A2AInvalidRequestError,
    A2AJsonRpcError,
    A2AMethodNotFoundError,
    A2AStreamingEndpointRequiredError,
    A2ATaskNotFoundError,
)
from src.agents.services.dvd_rag_service import DvdRagService

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient

A2AData = dict[str, Any]
A2AResponse = A2AData | list[A2AData]


class DocumentQaA2AService:
    """A2A JSON-RPC service for the regulatory-documents QA (RAG) agent."""

    STREAMING_METHODS = {
        "SendStreamingMessage",
        "message/stream",
        "tasks/sendSubscribe",
    }

    def __init__(
        self,
        dvd_rag_service: DvdRagService,
        task_store: A2ATaskStore | None = None,
        agent: DocumentQaA2AAgent | None = None,
    ) -> None:
        self.agent = agent or DocumentQaA2AAgent()
        self.task_store = task_store or A2ATaskStore()
        self.executor = DocumentQaAgentExecutor(dvd_rag_service, self.task_store)

    def get_agent_card(self, base_url: str) -> A2AData:
        return self.agent.get_agent_card(base_url)

    def is_streaming_request(self, payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and payload.get("method") in self.STREAMING_METHODS
        )

    async def handle_json_rpc(
        self,
        payload: Any,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
    ) -> A2AResponse:
        if isinstance(payload, list):
            return [
                (
                    await self._handle_single_json_rpc(item, dvd_mcp_client, token)
                    if isinstance(item, dict)
                    else self._error(None, A2AInvalidRequestError())
                )
                for item in payload
            ]
        if not isinstance(payload, dict):
            return self._error(None, A2AInvalidRequestError())
        return await self._handle_single_json_rpc(payload, dvd_mcp_client, token)

    async def stream_json_rpc(
        self,
        payload: Any,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
    ) -> AsyncGenerator[A2AData, None]:
        if not isinstance(payload, dict):
            yield self._error(None, A2AInvalidRequestError())
            return

        request_id = payload.get("id")
        try:
            self._validate_json_rpc(payload)
            method = payload.get("method")
            params = self._extract_params(payload)
            if method not in self.STREAMING_METHODS:
                result = await self._dispatch(method, params, dvd_mcp_client, token)
                yield self._success(request_id, result)
                return

            async for event in self.executor.stream(params, dvd_mcp_client, token):
                yield self._success(request_id, event)

        except A2AJsonRpcError as exc:
            yield self._success(
                request_id, self._terminal_failed_event(payload, exc.message)
            )
            yield self._error(request_id, exc)
        except Exception as exc:
            yield self._success(
                request_id, self._terminal_failed_event(payload, str(exc))
            )
            yield self._error(request_id, A2AJsonRpcError(-32000, str(exc)))

    async def _handle_single_json_rpc(
        self,
        payload: A2AData,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
    ) -> A2AData:
        request_id = payload.get("id")
        try:
            self._validate_json_rpc(payload)
            method = payload.get("method")
            params = self._extract_params(payload)
            result = await self._dispatch(method, params, dvd_mcp_client, token)
            return self._success(request_id, result)
        except A2AJsonRpcError as exc:
            return self._error(request_id, exc)
        except Exception as exc:
            return self._error(request_id, A2AJsonRpcError(-32000, str(exc)))

    async def _dispatch(
        self,
        method: str | None,
        params: A2AData,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
    ) -> A2AResponse:
        if method in {"SendMessage", "message/send", "tasks/send"}:
            task = await self.executor.execute(params, dvd_mcp_client, token)
            configuration = params.get("configuration") or {}
            return apply_history_length(task, configuration.get("historyLength"))
        if method in self.STREAMING_METHODS:
            raise A2AStreamingEndpointRequiredError()
        if method in {"GetTask", "tasks/get"}:
            return self._get_task(params)
        if method in {"ListTasks", "tasks/list"}:
            return self.task_store.list_tasks(
                include_artifacts=bool(params.get("includeArtifacts", True))
            )
        if method in {"CancelTask", "tasks/cancel"}:
            return self._cancel_task(params)
        if method in {"GetExtendedAgentCard", "agent/getAuthenticatedExtendedCard"}:
            base_url = str(params.get("baseUrl", "")).rstrip("/")
            return (
                self.get_agent_card(base_url) if base_url else self.get_agent_card("")
            )
        raise A2AMethodNotFoundError(method)

    def _get_task(self, params: A2AData) -> A2AData:
        task_id = self._task_id_from_params(params)
        task = self.task_store.get_task(task_id)
        if task is None:
            raise A2ATaskNotFoundError(task_id)
        return apply_history_length(task, params.get("historyLength"))

    def _cancel_task(self, params: A2AData) -> A2AData:
        task_id = self._task_id_from_params(params)
        task = self.task_store.get_task(task_id)
        if task is None:
            raise A2ATaskNotFoundError(task_id)
        if task["status"]["state"] not in {
            TaskState.COMPLETED.value,
            TaskState.FAILED.value,
            TaskState.CANCELED.value,
        }:
            self.task_store.set_status(task_id, TaskState.CANCELED)
        return self.task_store.get_task(task_id) or task

    @staticmethod
    def _validate_json_rpc(payload: A2AData) -> None:
        if payload.get("jsonrpc") != "2.0":
            raise A2AInvalidRequestError()

    @staticmethod
    def _extract_params(payload: A2AData) -> A2AData:
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise A2AInvalidParamsError()
        return params

    @staticmethod
    def _task_id_from_params(params: A2AData) -> str:
        task_id = params.get("id") or params.get("taskId")
        if not task_id:
            raise A2AInvalidParamsError("Task id is required")
        return str(task_id)

    @staticmethod
    def _success(request_id: Any, result: Any) -> A2AData:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": normalize_response(result),
        }

    @staticmethod
    def _terminal_failed_event(payload: A2AData, message_text: str) -> A2AData:
        """Build a terminal failed status-update for the streaming error path."""
        params = payload.get("params") if isinstance(payload, dict) else None
        params = params if isinstance(params, dict) else {}
        task_id = params.get("id") or params.get("taskId") or str(uuid4())
        context_id = params.get("contextId") or params.get("context_id") or str(uuid4())
        return {
            "kind": "status-update",
            "taskId": task_id,
            "contextId": context_id,
            "status": {
                "state": TaskState.FAILED.value,
                "timestamp": utc_now_rfc3339(),
                "message": {
                    "kind": "message",
                    "messageId": str(uuid4()),
                    "role": "agent",
                    "parts": [{"kind": "text", "text": message_text}],
                },
            },
            "final": True,
        }

    @staticmethod
    def _error(request_id: Any, exc: A2AJsonRpcError) -> A2AData:
        error = {"code": exc.code, "message": exc.message}
        if exc.data is not None:
            error["data"] = exc.data
        return {"jsonrpc": "2.0", "id": request_id, "error": error}
