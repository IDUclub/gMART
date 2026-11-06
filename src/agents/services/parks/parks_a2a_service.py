"""A2A 0.3 JSON-RPC adapter over the Redis-authoritative park workflow."""

from uuid import uuid4

from pydantic import ValidationError

from src.agents.__version__ import APP_VERSION
from src.agents.a2a.a2a_format import utc_now_rfc3339
from src.agents.dto.parks_request_dto import ParksInputs


class ParksA2AService:
    STREAMING_METHODS = {
        "message/stream",
        "SendStreamingMessage",
        "tasks/sendSubscribe",
    }

    def __init__(self, service):
        self.service = service

    def get_agent_card(self, base_url):
        return {
            "name": "parks-agent",
            "description": "Согласование, генерация и анализ парков через gMART.",
            "url": base_url.rstrip("/") + "/parks/a2a",
            "version": APP_VERSION,
            "protocolVersion": "0.3.0",
            "preferredTransport": "JSONRPC",
            "capabilities": {"streaming": True, "pushNotifications": False},
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "security": [{"bearer": []}],
            "skills": [
                {
                    "id": "generate-park",
                    "name": "Генерация парка",
                    "description": "Передайте ParksInputs в metadata.inputs или DataPart. Ответ на согласование — новое сообщение с тем же contextId. request_id восстанавливает поток.",
                    "tags": ["parks", "geojson", "generation"],
                    "examples": ["Создай парк с детской площадкой"],
                }
            ],
        }

    def is_streaming_request(self, payload):
        return (
            isinstance(payload, dict)
            and payload.get("method") in self.STREAMING_METHODS
        )

    @staticmethod
    def _response(request_id, result):
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id, code, text):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": text},
        }

    async def handle_json_rpc(self, payload, token):
        if isinstance(payload, list):
            return [await self.handle_json_rpc(item, token) for item in payload]
        result = None
        async for response in self.stream_json_rpc(payload, token):
            result = response
        if (
            result
            and "result" in result
            and result["result"].get("kind") == "status-update"
        ):
            request_id = result["result"]["taskId"]
            return self._response(
                payload.get("id"), await self._task(request_id, token)
            )
        return result

    @staticmethod
    def _task_status(status):
        return {
            "awaiting_confirmation": "input-required",
            "done": "completed",
            "failed": "failed",
            "submission_unknown": "failed",
        }.get(status, "working")

    async def _task(self, request_id, token):
        state = await self.service.get_run(request_id, token)
        status = self._task_status(state["status"])
        events = await self.service.store.events(request_id, 0)
        artifacts = [
            self._artifact(event)
            for event in events
            if event["type"] == "feature_collection"
            or (event["type"] == "chunk" and event["content"].get("text"))
        ]
        context = (state["conversation"] or request_id).removeprefix("a2a:")
        latest = next(
            (
                event
                for event in reversed(events)
                if event["type"] in {"clarification", "error"}
            ),
            None,
        )
        task_status = {"state": status, "timestamp": utc_now_rfc3339()}
        if latest:
            data = latest["content"]
            task_status["message"] = {
                "kind": "message",
                "messageId": str(uuid4()),
                "role": "agent",
                "parts": [
                    {
                        "kind": "text",
                        "text": data.get("question") or data.get("message", ""),
                    },
                    {"kind": "data", "data": data},
                ],
            }
        return {
            "kind": "task",
            "id": request_id,
            "contextId": context,
            "status": task_status,
            "artifacts": artifacts,
        }

    @staticmethod
    def _artifact(event):
        content = event["content"]
        part = (
            {"kind": "text", "text": content["text"]}
            if event["type"] == "chunk"
            else {"kind": "data", "data": content}
        )
        return {
            "artifactId": str(content.get("event_id", uuid4())),
            "name": event["type"],
            "parts": [part],
        }

    async def stream_json_rpc(self, payload, token):
        rpc_id = payload.get("id") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
            yield self._error(rpc_id, -32600, "Invalid Request")
            return
        try:
            params = payload.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError()
            method = payload.get("method")
            if method in {"tasks/get", "GetTask"}:
                yield self._response(
                    rpc_id,
                    await self._task(
                        str(params.get("id") or params.get("taskId") or ""), token
                    ),
                )
                return
            if method not in self.STREAMING_METHODS | {
                "message/send",
                "SendMessage",
                "tasks/send",
            }:
                yield self._error(rpc_id, -32601, "Method not found")
                return
            message = params.get("message") or {}
            metadata = dict(params.get("metadata") or {})
            metadata.update(message.get("metadata") or {})
            for part in message.get("parts", []):
                if isinstance(part.get("data"), dict):
                    metadata.update(part["data"])
            text = "".join(
                part.get("text", "") for part in message.get("parts", [])
            ) or params.get("request", "")
            request_id = metadata.get("request_id")
            if not text and not request_id:
                raise ValueError()
            context = str(
                params.get("contextId") or message.get("contextId") or uuid4()
            )
            if request_id:
                stored = await self.service.get_run(request_id, token)
                context = (stored["conversation"] or request_id).removeprefix("a2a:")
            inputs = ParksInputs.model_validate(metadata.get("inputs") or {})
            task_id = request_id
            waiting = False
            failed = False
            async for event in self.service.run_parks_pipeline(
                token=token,
                model=metadata.get("model"),
                temperature=float(metadata.get("temperature", 1)),
                user_query=text,
                request_id=request_id,
                conversation_key="a2a:" + context,
                persist_history=False,
                inputs=inputs,
                after=int(metadata.get("after", 0)),
            ):
                if event["type"] == "pipeline_started":
                    task_id = event["content"]["request_id"]
                task_id = task_id or request_id
                content = event["content"]
                if event["type"] == "feature_collection" or (
                    event["type"] == "chunk" and content.get("text")
                ):
                    yield self._response(
                        rpc_id,
                        {
                            "kind": "artifact-update",
                            "taskId": task_id,
                            "contextId": context,
                            "artifact": self._artifact(event),
                            "append": False,
                            "lastChunk": True,
                        },
                    )
                elif event["type"] in {"status", "clarification", "error"}:
                    waiting = event["type"] == "clarification"
                    failed = event["type"] == "error"
                    yield self._response(
                        rpc_id,
                        {
                            "kind": "status-update",
                            "taskId": task_id,
                            "contextId": context,
                            "status": {
                                "state": (
                                    "input-required"
                                    if waiting
                                    else "failed" if failed else "working"
                                ),
                                "timestamp": utc_now_rfc3339(),
                                "message": {
                                    "kind": "message",
                                    "messageId": str(uuid4()),
                                    "role": "agent",
                                    "parts": [{"kind": "data", "data": content}],
                                },
                            },
                            "final": False,
                        },
                    )
            if task_id:
                stored = await self.service.get_run(task_id, token)
                yield self._response(
                    rpc_id,
                    {
                        "kind": "status-update",
                        "taskId": task_id,
                        "contextId": context,
                        "status": {
                            "state": self._task_status(stored["status"]),
                            "timestamp": utc_now_rfc3339(),
                        },
                        "final": True,
                    },
                )
        except LookupError:
            yield self._error(rpc_id, -32001, "Task not found")
        except (ValueError, TypeError, AttributeError, ValidationError):
            yield self._error(rpc_id, -32602, "Invalid params")
