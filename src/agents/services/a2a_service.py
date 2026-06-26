from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from python_a2a.models.task import TaskState

from src.agents.a2a.a2a_format import (
    apply_history_length,
    normalize_response,
    utc_now_rfc3339,
)
from src.agents.a2a.agent import RestrictionA2AAgent
from src.agents.a2a.executor import RestrictionAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.common.exceptions.a2a_exceptions import (
    A2AInvalidParamsError,
    A2AInvalidRequestError,
    A2AJsonRpcError,
    A2AMethodNotFoundError,
    A2AStreamingEndpointRequiredError,
    A2ATaskNotFoundError,
)
from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
from src.agents.services.restriction_parser_service import (
    RestrictionParserService,
)

A2AData = dict[str, Any]
A2AResponse = A2AData | list[A2AData]


class A2AService:
    """
    Service for A2A JSON-RPC requests.
    Attributes:
        STREAMING_METHODS (set[str]): A2A methods that should return SSE stream.
        agent (RestrictionA2AAgent): A2A agent entity.
        task_store (A2ATaskStore): A2A task storage.
        executor (RestrictionAgentExecutor): Restriction agent executor.
    """

    STREAMING_METHODS = {
        "SendStreamingMessage",
        "message/stream",
        "tasks/sendSubscribe",
    }

    def __init__(
        self,
        restriction_service: RestrictionParserService,
        task_store: A2ATaskStore | None = None,
        agent: RestrictionA2AAgent | None = None,
    ) -> None:
        """
        A2AService initialization function.
        Args:
            restriction_service (RestrictionParserService): Restriction pipeline service.
            task_store (A2ATaskStore | None): Optional task storage.
            agent (RestrictionA2AAgent | None): Optional A2A agent entity.
        """

        self.agent = agent or RestrictionA2AAgent()
        self.task_store = task_store or A2ATaskStore()
        self.executor = RestrictionAgentExecutor(restriction_service, self.task_store)

    def get_agent_card(self, base_url: str) -> A2AData:
        """
        Function returns A2A agent card.
        Args:
            base_url (str): Public server base url.
        Returns:
            A2AData: Serialized A2A agent card.
        """

        return self.agent.get_agent_card(base_url)

    def is_streaming_request(self, payload: Any) -> bool:
        """
        Function checks if payload should be handled as A2A stream.
        Args:
            payload (Any): Incoming payload.
        Returns:
            bool: True if request method is streaming.
        """

        return (
            isinstance(payload, dict)
            and payload.get("method") in self.STREAMING_METHODS
        )

    async def handle_json_rpc(
        self,
        payload: Any,
        mcp_client: IduMcpClient,
    ) -> A2AResponse:
        """
        Function handles regular A2A JSON-RPC payload.
        Args:
            payload (Any): JSON-RPC request or batch request.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Returns:
            A2AResponse: JSON-RPC response or batch response.
        """

        if isinstance(payload, list):
            return [
                (
                    await self._handle_single_json_rpc(item, mcp_client)
                    if isinstance(item, dict)
                    else self._error(None, A2AInvalidRequestError())
                )
                for item in payload
            ]
        if not isinstance(payload, dict):
            return self._error(None, A2AInvalidRequestError())
        return await self._handle_single_json_rpc(payload, mcp_client)

    async def stream_json_rpc(
        self,
        payload: Any,
        mcp_client: IduMcpClient,
    ) -> AsyncGenerator[A2AData, None]:
        """
        Function handles streaming A2A JSON-RPC payload.
        Args:
            payload (Any): JSON-RPC request.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Yields:
            A2AData: JSON-RPC response envelope with A2A event.
        """

        if not isinstance(payload, dict):
            yield self._error(None, A2AInvalidRequestError())
            return

        request_id = payload.get("id")
        try:
            self._validate_json_rpc(payload)
            method = payload.get("method")
            params = self._extract_params(payload)
            if method not in self.STREAMING_METHODS:
                result = await self._dispatch(method, params, mcp_client)
                yield self._success(request_id, result)
                return

            async for event in self.executor.stream(params, mcp_client):
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
        mcp_client: IduMcpClient,
    ) -> A2AData:
        """
        Function handles single A2A JSON-RPC request.
        Args:
            payload (A2AData): JSON-RPC request.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Returns:
            A2AData: JSON-RPC response.
        """

        request_id = payload.get("id")
        try:
            self._validate_json_rpc(payload)
            method = payload.get("method")
            params = self._extract_params(payload)
            result = await self._dispatch(method, params, mcp_client)
            return self._success(request_id, result)

        except A2AJsonRpcError as exc:
            return self._error(request_id, exc)
        except Exception as exc:
            return self._error(request_id, A2AJsonRpcError(-32000, str(exc)))

    async def _dispatch(
        self,
        method: str | None,
        params: A2AData,
        mcp_client: IduMcpClient,
    ) -> A2AResponse:
        """
        Function dispatches A2A method to corresponding handler.
        Args:
            method (str | None): A2A method name.
            params (A2AData): A2A method params.
            mcp_client (IduMcpClient): MCP client for geospatial tools.
        Returns:
            A2AResponse: A2A method result.
        """

        if method in {"SendMessage", "message/send", "tasks/send"}:
            task = await self.executor.execute(params, mcp_client)
            configuration = params.get("configuration") or {}
            return apply_history_length(task, configuration.get("historyLength"))
        if method in self.STREAMING_METHODS:
            raise A2AStreamingEndpointRequiredError()
        if method in {"GetTask", "tasks/get"}:
            return self._get_task(params)
        if method in {"ListTasks", "tasks/list"}:
            include_artifacts = bool(params.get("includeArtifacts", True))
            return self.task_store.list_tasks(include_artifacts=include_artifacts)
        if method in {"CancelTask", "tasks/cancel"}:
            return self._cancel_task(params)
        if method in {
            "GetExtendedAgentCard",
            "agent/getAuthenticatedExtendedCard",
        }:
            base_url = str(params.get("baseUrl", "")).rstrip("/")
            return (
                self.get_agent_card(base_url) if base_url else self.get_agent_card("")
            )
        raise A2AMethodNotFoundError(method)

    def _get_task(self, params: A2AData) -> A2AData:
        """
        Function returns A2A task by id.
        Args:
            params (A2AData): A2A method params.
        Returns:
            A2AData: Serialized A2A task.
        """

        task_id = self._task_id_from_params(params)
        task = self.task_store.get_task(task_id)
        if task is None:
            raise A2ATaskNotFoundError(task_id)
        return apply_history_length(task, params.get("historyLength"))

    def _cancel_task(self, params: A2AData) -> A2AData:
        """
        Function cancels A2A task by id.
        Args:
            params (A2AData): A2A method params.
        Returns:
            A2AData: Serialized A2A task.
        """

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
        """
        Function validates JSON-RPC protocol version.
        Args:
            payload (A2AData): JSON-RPC request.
        """

        if payload.get("jsonrpc") != "2.0":
            raise A2AInvalidRequestError()

    @staticmethod
    def _extract_params(payload: A2AData) -> A2AData:
        """
        Function extracts JSON-RPC params.
        Args:
            payload (A2AData): JSON-RPC request.
        Returns:
            A2AData: JSON-RPC params.
        """

        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise A2AInvalidParamsError()
        return params

    @staticmethod
    def _task_id_from_params(params: A2AData) -> str:
        """
        Function extracts task id from A2A params.
        Args:
            params (A2AData): A2A method params.
        Returns:
            str: A2A task id.
        """

        task_id = params.get("id") or params.get("taskId")
        if not task_id:
            raise A2AInvalidParamsError("Task id is required")
        return str(task_id)

    @staticmethod
    def _success(request_id: Any, result: Any) -> A2AData:
        """
        Function creates JSON-RPC success response.
        Args:
            request_id (Any): JSON-RPC request id.
            result (Any): A2A method result.
        Returns:
            A2AData: JSON-RPC success response.
        """

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": normalize_response(result),
        }

    @staticmethod
    def _terminal_failed_event(payload: A2AData, message_text: str) -> A2AData:
        """
        Function builds a terminal failed status-update for the streaming error path.

        Streaming clients subscribed to ``message/stream`` must receive a terminal event
        even when the request is rejected before the pipeline starts; an empty SSE stream
        would hang a spec-compliant client.
        Args:
            payload (A2AData): The originating JSON-RPC request.
            message_text (str): Human-readable failure reason.
        Returns:
            A2AData: A2A ``statusUpdate`` event with ``final`` set and ``failed`` state.
        """

        params = payload.get("params") if isinstance(payload, dict) else None
        params = params if isinstance(params, dict) else {}
        task_id = params.get("id") or params.get("taskId") or str(uuid4())
        context_id = params.get("contextId") or params.get("context_id") or str(uuid4())
        return {
            "statusUpdate": {
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
        }

    @staticmethod
    def _error(request_id: Any, exc: A2AJsonRpcError) -> A2AData:
        """
        Function creates JSON-RPC error response.
        Args:
            request_id (Any): JSON-RPC request id.
            exc (A2AJsonRpcError): A2A JSON-RPC error.
        Returns:
            A2AData: JSON-RPC error response.
        """

        error = {"code": exc.code, "message": exc.message}
        if exc.data is not None:
            error["data"] = exc.data
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error,
        }
