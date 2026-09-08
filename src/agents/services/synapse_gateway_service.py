from __future__ import annotations

import asyncio
import hashlib
import json
import random
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from loguru import logger

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    StatusPartRequest,
    StatusPayload,
    StructuredPartRequest,
    TextPartRequest,
    TextPayload,
)
from src.agents.api_clients.synapse_client import (
    SynapseApiClient,
    SynapseResponseError,
    SynapseSseEvent,
    SynapseUnavailableError,
)
from src.agents.dto.synapse_request_dto import SynapseRunRequestDTO
from src.agents.services.synapse_run_store import SynapseRunStore


class SynapseGatewayError(RuntimeError):
    pass


class SynapseGatewayConflict(SynapseGatewayError):
    pass


class SynapseConfigurationRequired(SynapseGatewayError):
    pass


class SynapseStartUnknownError(SynapseGatewayError):
    pass


class SynapseRunNotFound(SynapseGatewayError):
    pass


class SynapseGatewayService:
    TERMINAL_EVENTS = {
        "project_completed": "done",
        "project_failed": "failed",
        "project_stopped": "cancelled",
    }

    def __init__(
        self,
        client: SynapseApiClient,
        store: SynapseRunStore,
        chat_storage: ChatStorageApiClient,
        *,
        workflow_id: str | None = None,
        run_config_id: str | None = None,
        reconnect_max_seconds: float = 30.0,
    ) -> None:
        self.client = client
        self.store = store
        self.chat_storage = chat_storage
        self.workflow_id = workflow_id
        self.run_config_id = run_config_id
        self.reconnect_max_seconds = reconnect_max_seconds
        self._tasks: dict[str, asyncio.Task] = {}
        self._owner = str(uuid4())

    @staticmethod
    def build_prompt(request_id: str, payload: SynapseRunRequestDTO) -> str:
        context = {
            "request_id": request_id,
            "scenario_id": payload.scenario_id,
            "urban_project_id": payload.project_id,
            "selected_object_ids": payload.metadata.get("selected_object_ids", []),
            "selected_layer_ids": payload.metadata.get("selected_layer_ids", []),
        }
        lines = ["[IDU_CONTEXT_V1]"]
        for key, value in context.items():
            lines.append(
                f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            )
        lines.extend(
            [
                "[/IDU_CONTEXT_V1]",
                "",
                "[USER_REQUEST]",
                payload.request.strip(),
                "[/USER_REQUEST]",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def payload_hash(payload: SynapseRunRequestDTO) -> str:
        canonical = json.dumps(
            payload.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    async def start_run(
        self,
        *,
        user_id: str,
        idempotency_key: str,
        payload: SynapseRunRequestDTO,
    ) -> dict[str, Any]:
        if not payload.chat_id:
            payload = self._with_resolved_configuration(payload)
        candidate_request_id = str(uuid4())
        request_id, claimed = await self.store.claim_idempotency(
            user_id=user_id,
            key=idempotency_key,
            payload_hash=self.payload_hash(payload),
            request_id=candidate_request_id,
        )
        if not claimed:
            state = await self.store.get_state(request_id)
            if not state:
                raise SynapseGatewayConflict(
                    "Idempotent run was claimed but its state is unavailable"
                )
            if state.get("status") == "start_unknown":
                state = await self._recover_unknown_start(
                    request_id=request_id,
                    user_id=user_id,
                    payload=payload,
                    started_at=str(state.get("started_at") or ""),
                )
                if state.get("status") == "running":
                    self.start_relay(request_id)
            return state

        started_at = datetime.now(UTC).isoformat()
        await self.store.create_state(
            request_id,
            {
                "request_id": request_id,
                "user_id": user_id,
                "chat_id": payload.chat_id,
                "synapse_project_id": None,
                "run_id": None,
                "workflow_id": payload.workflow_id,
                "run_config_id": payload.run_config_id,
                "status": "starting",
                "started_at": started_at,
                "finished_at": None,
                "last_event_id": None,
                "last_stream_id": None,
                "error": None,
            },
        )

        prompt = self.build_prompt(request_id, payload)
        try:
            if payload.chat_id:
                state = await self._continue_project(
                    request_id=request_id,
                    user_id=user_id,
                    prompt=prompt,
                    payload=payload,
                )
            else:
                state = await self._create_project(
                    request_id=request_id,
                    user_id=user_id,
                    prompt=prompt,
                    payload=payload,
                    started_at=started_at,
                )
        except Exception as exc:
            if isinstance(exc, SynapseStartUnknownError):
                await self.store.update_state(
                    request_id,
                    status="start_unknown",
                    error=str(exc),
                )
            else:
                await self.store.update_state(
                    request_id,
                    status="failed",
                    error=self._safe_error(exc),
                    finished_at=datetime.now(UTC).isoformat(),
                )
            raise

        self.start_relay(request_id)
        return state

    async def _create_project(
        self,
        *,
        request_id: str,
        user_id: str,
        prompt: str,
        payload: SynapseRunRequestDTO,
        started_at: str,
    ) -> dict[str, Any]:
        try:
            project = await self.client.create_project(
                prompt,
                workflow_id=payload.workflow_id,
                run_config_id=payload.run_config_id,
            )
        except (SynapseUnavailableError, SynapseResponseError) as create_exc:
            matches = await self._reconcile_project(
                request_id,
                workflow_id=payload.workflow_id,
                run_config_id=payload.run_config_id,
            )
            if len(matches) != 1:
                if isinstance(create_exc, SynapseUnavailableError):
                    raise SynapseStartUnknownError(
                        "Synapse project creation result is unknown"
                    ) from create_exc
                raise create_exc
            project = matches[0]

        return await self._attach_project(
            request_id=request_id,
            user_id=user_id,
            project=project,
            payload=payload,
            started_at=started_at,
        )

    async def _recover_unknown_start(
        self,
        *,
        request_id: str,
        user_id: str,
        payload: SynapseRunRequestDTO,
        started_at: str,
    ) -> dict[str, Any]:
        if payload.chat_id:
            state = await self.store.get_state(request_id)
            assert state is not None
            if state.get("synapse_project_id") and state.get("run_id"):
                await self.store.update_state(
                    request_id,
                    status="running",
                    error=None,
                )
                state = await self.store.get_state(request_id)
                assert state is not None
            return state
        matches = await self._reconcile_project(
            request_id,
            workflow_id=payload.workflow_id,
            run_config_id=payload.run_config_id,
        )
        if len(matches) != 1:
            state = await self.store.get_state(request_id)
            assert state is not None
            return state
        return await self._attach_project(
            request_id=request_id,
            user_id=user_id,
            project=matches[0],
            payload=payload,
            started_at=started_at,
        )

    async def _attach_project(
        self,
        *,
        request_id: str,
        user_id: str,
        project: dict[str, Any],
        payload: SynapseRunRequestDTO,
        started_at: str,
    ) -> dict[str, Any]:

        project_id = str(project.get("project_id") or project.get("id") or "")
        if not project_id:
            raise SynapseGatewayError("Synapse project response has no project_id")
        # Synapse starts the workflow asynchronously. Publish the project/user
        # correlation before polling so an early A2A callback can be authorized.
        await self.store.update_state(
            request_id,
            synapse_project_id=project_id,
        )
        await self.store.bind_project(request_id, project_id)
        project_state = await self._wait_for_run_id(project_id)
        run_id = project_state.get("current_run_id") or project_state.get("run_id")

        title = str(project.get("title") or payload.request.strip()[:120])
        chat = await self.chat_storage.create_chat(
            None,
            title,
            scenario_id=payload.scenario_id,
            project_id=payload.project_id,
            space="synapse",
            user_id=user_id,
            provider="synapse",
            agent_id="synapse",
            synapse_project_id=project_id,
            synapse_workflow_id=payload.workflow_id,
            synapse_run_config_id=payload.run_config_id,
        )
        await self.chat_storage.add_single_message(
            None,
            chat.chat_id,
            RoleEnum.USER,
            payload.request,
            space="synapse",
            user_id=user_id,
            source_event_id=f"request:{request_id}",
            provider="synapse",
            request_id=request_id,
        )
        if not await self.store.claim_chat(chat.chat_id, request_id):
            raise SynapseGatewayConflict("Chat already has an active Synapse run")
        await self.store.bind_project(
            request_id,
            project_id,
            run_id=str(run_id) if run_id else None,
            chat_id=chat.chat_id,
        )
        await self.store.update_state(
            request_id,
            chat_id=chat.chat_id,
            synapse_project_id=project_id,
            run_id=str(run_id) if run_id else None,
            status="running",
            started_at=project_state.get("created_at") or started_at,
        )
        state = await self.store.get_state(request_id)
        assert state is not None
        return state

    async def _continue_project(
        self,
        *,
        request_id: str,
        user_id: str,
        prompt: str,
        payload: SynapseRunRequestDTO,
    ) -> dict[str, Any]:
        assert payload.chat_id is not None
        chat = await self.chat_storage.get_chat(
            None, payload.chat_id, space="synapse", user_id=user_id
        )
        metadata = chat.metadata or {}
        project_id = metadata.get("synapse_project_id")
        if not project_id:
            raise SynapseGatewayConflict("Chat has no Synapse project mapping")
        stored_workflow_id = metadata.get("synapse_workflow_id")
        stored_run_config_id = metadata.get("synapse_run_config_id")
        if (
            payload.workflow_id
            and stored_workflow_id
            and payload.workflow_id != stored_workflow_id
        ):
            raise SynapseGatewayConflict(
                "Workflow cannot be changed for an existing Synapse chat"
            )
        if (
            payload.run_config_id
            and stored_run_config_id
            and payload.run_config_id != stored_run_config_id
        ):
            raise SynapseGatewayConflict(
                "Run configuration cannot be changed for an existing Synapse chat"
            )
        workflow_id = str(stored_workflow_id or payload.workflow_id or "") or None
        run_config_id = str(stored_run_config_id or payload.run_config_id or "") or None
        await self.store.update_state(
            request_id,
            workflow_id=workflow_id,
            run_config_id=run_config_id,
        )
        if not await self.store.claim_chat(payload.chat_id, request_id):
            raise SynapseGatewayConflict("Chat already has an active Synapse run")
        try:
            project_state = await self.client.get_project(str(project_id))
            run_id = project_state.get("current_run_id") or project_state.get("run_id")
            if not run_id:
                raise SynapseGatewayError(
                    "Synapse project did not expose current_run_id"
                )
            # A follow-up may delegate immediately after the messages API returns.
            # Rebind the existing Synapse project before sending the message.
            await self.store.update_state(
                request_id,
                chat_id=payload.chat_id,
                synapse_project_id=str(project_id),
                run_id=str(run_id),
            )
            await self.store.bind_project(
                request_id,
                str(project_id),
                run_id=str(run_id),
                chat_id=payload.chat_id,
            )
            try:
                await self.client.send_message(
                    str(project_id),
                    prompt,
                    run_id=str(run_id),
                    metadata={"request_id": request_id},
                )
            except SynapseUnavailableError as exc:
                raise SynapseStartUnknownError(
                    "Synapse message submission result is unknown"
                ) from exc
            await self.chat_storage.add_single_message(
                None,
                payload.chat_id,
                RoleEnum.USER,
                payload.request,
                space="synapse",
                user_id=user_id,
                source_event_id=f"request:{request_id}",
                provider="synapse",
                request_id=request_id,
            )
        except SynapseStartUnknownError:
            raise
        except Exception:
            await self.store.release_chat(payload.chat_id, request_id)
            raise
        await self.store.bind_project(
            request_id,
            str(project_id),
            run_id=str(run_id) if run_id else None,
            chat_id=payload.chat_id,
        )
        await self.store.update_state(
            request_id,
            chat_id=payload.chat_id,
            synapse_project_id=str(project_id),
            run_id=str(run_id) if run_id else None,
            status="running",
        )
        state = await self.store.get_state(request_id)
        assert state is not None
        return state

    async def _reconcile_project(
        self,
        request_id: str,
        *,
        workflow_id: str | None,
        run_config_id: str | None,
    ) -> list[dict[str, Any]]:
        try:
            return await self.client.find_projects(
                request_id,
                workflow_id=workflow_id,
                run_config_id=run_config_id,
            )
        except Exception:
            return []

    def _with_resolved_configuration(
        self, payload: SynapseRunRequestDTO
    ) -> SynapseRunRequestDTO:
        workflow_id = payload.workflow_id or self.workflow_id
        run_config_id = payload.run_config_id or self.run_config_id
        if not workflow_id or not run_config_id:
            raise SynapseConfigurationRequired(
                "Select a Synapse workflow and run configuration"
            )
        return payload.model_copy(
            update={"workflow_id": workflow_id, "run_config_id": run_config_id}
        )

    async def get_configuration_options(self) -> dict[str, Any]:
        workflows, run_configurations = await asyncio.gather(
            self.client.list_workflows(),
            self.client.list_run_configurations(),
        )

        workflow_options = [self._workflow_option(item) for item in workflows]
        run_config_options = [
            self._configuration_option(item) for item in run_configurations
        ]
        workflow_options = [item for item in workflow_options if item]
        run_config_options = [item for item in run_config_options if item]
        return {
            "workflows": workflow_options,
            "run_configurations": run_config_options,
            "default_workflow_id": self._default_option_id(
                workflow_options, self.workflow_id
            ),
            "default_run_config_id": self._default_option_id(
                run_config_options, self.run_config_id
            ),
        }

    @staticmethod
    def _configuration_option(item: dict[str, Any]) -> dict[str, Any] | None:
        option_id = item.get("id") or item.get("_id")
        if not option_id:
            return None
        return {
            "id": str(option_id),
            "name": str(item.get("name") or option_id),
            "description": str(
                item.get("short_description") or item.get("description") or ""
            ),
            "is_default": bool(item.get("is_default")),
        }

    @classmethod
    def _workflow_option(cls, item: dict[str, Any]) -> dict[str, Any] | None:
        option = cls._configuration_option(item)
        if option is None:
            return None
        option.update(
            display_name=item.get("display_name"),
            execution_mode=item.get("execution_mode"),
        )
        return option

    @staticmethod
    def _default_option_id(
        options: list[dict[str, Any]], configured_id: str | None
    ) -> str | None:
        if configured_id and any(item["id"] == configured_id for item in options):
            return configured_id
        default = next((item for item in options if item["is_default"]), None)
        if default:
            return str(default["id"])
        return str(options[0]["id"]) if options else None

    async def _wait_for_run_id(self, project_id: str) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for attempt in range(6):
            state = await self.client.get_project(project_id)
            if state.get("current_run_id") or state.get("run_id"):
                return state
            await asyncio.sleep(0.1 * 2**attempt)
        raise SynapseGatewayError("Synapse project did not expose current_run_id")

    def start_relay(self, request_id: str) -> None:
        task = self._tasks.get(request_id)
        if task and not task.done():
            return
        self._tasks[request_id] = asyncio.create_task(self._relay(request_id))

    async def recover_active_runs(self) -> None:
        for request_id in await self.store.active_request_ids():
            state = await self.store.get_state(request_id)
            if state and state.get("status") == "running":
                self.start_relay(request_id)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.close()

    async def _relay(self, request_id: str) -> None:
        while not await self.store.acquire_relay(request_id, self._owner):
            state = await self.store.get_state(request_id)
            if not state or state.get("status") != "running":
                return
            await asyncio.sleep(5)
        relay_task = asyncio.current_task()
        assert relay_task is not None
        renew_task = asyncio.create_task(
            self._renew_relay_owner(request_id, relay_task)
        )
        backoff = 0.5
        try:
            while True:
                state = await self.store.get_state(request_id)
                if not state or state.get("status") != "running":
                    return
                project_id = str(state.get("synapse_project_id") or "")
                if not project_id:
                    return
                try:
                    async for incoming in self.client.stream_events(
                        project_id,
                        run_id=state.get("run_id"),
                        last_event_id=state.get("last_event_id"),
                        since=state.get("started_at"),
                    ):
                        await self._process_event(request_id, state, incoming)
                        if incoming.event in self.TERMINAL_EVENTS:
                            return
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.reconnect_max_seconds)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning(
                        "Synapse relay disconnected request_id={} error_type={}",
                        request_id,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(backoff + random.random() * min(backoff, 1.0))
                    backoff = min(backoff * 2, self.reconnect_max_seconds)
        finally:
            renew_task.cancel()
            await asyncio.gather(renew_task, return_exceptions=True)
            await self.store.release_relay(request_id, self._owner)
            self._tasks.pop(request_id, None)

    async def _renew_relay_owner(
        self, request_id: str, relay_task: asyncio.Task
    ) -> None:
        while True:
            await asyncio.sleep(10)
            if not await self.store.renew_relay(request_id, self._owner):
                relay_task.cancel()
                return

    async def _process_event(
        self,
        request_id: str,
        state: dict[str, Any],
        incoming: SynapseSseEvent,
    ) -> None:
        source_event_id = incoming.event_id or str(
            uuid5(
                NAMESPACE_URL,
                json.dumps(
                    [state.get("synapse_project_id"), incoming.event, incoming.data],
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            )
        )
        normalized = {
            "type": "synapse_event",
            "source_type": incoming.event,
            "source_event_id": source_event_id,
            "request_id": request_id,
            "synapse_project_id": state["synapse_project_id"],
            "run_id": state.get("run_id"),
            "timestamp": incoming.data.get("timestamp")
            or datetime.now(UTC).isoformat(),
            "content": incoming.data,
        }
        if self._is_durable(incoming.event):
            await self._persist_event(state, normalized)
        stream_id = await self.store.add_event(
            request_id,
            source_event_id=source_event_id,
            event=normalized,
        )
        if stream_id is None:
            return
        if incoming.event in self.TERMINAL_EVENTS:
            status = self.TERMINAL_EVENTS[incoming.event]
            await self.store.update_state(
                request_id,
                status=status,
                finished_at=datetime.now(UTC).isoformat(),
                error=(
                    self._safe_event_error(incoming.data)
                    if status == "failed"
                    else None
                ),
            )
            if state.get("chat_id"):
                await self.store.release_chat(str(state["chat_id"]), request_id)

    async def _persist_event(
        self, state: dict[str, Any], event: dict[str, Any]
    ) -> None:
        event_type = event["source_type"]
        if event_type in self.TERMINAL_EVENTS:
            status = self.TERMINAL_EVENTS[event_type]
            parts = [
                StatusPartRequest(
                    kind="status", payload=StatusPayload(status=status, text=event_type)
                )
            ]
            if status == "failed":
                parts.append(
                    StructuredPartRequest(kind="failure", payload=event["content"])
                )
        elif event_type in {"message_appended", "a2a_message"}:
            message = event["content"].get("message")
            message = message if isinstance(message, dict) else event["content"]
            if str(message.get("role") or "").lower() == "user":
                return
            text = self._event_text(event["content"])
            parts = [TextPartRequest(kind="text", payload=TextPayload(text=text))]
        else:
            parts = [StructuredPartRequest(kind="data", payload=event["content"])]
        await self.chat_storage.add_parts_message(
            None,
            state["chat_id"],
            (
                RoleEnum.ASSISTANT
                if event_type in {"message_appended", "a2a_message"}
                else RoleEnum.SYSTEM
            ),
            parts,
            space="synapse",
            user_id=state["user_id"],
            source_event_id=event["source_event_id"],
            provider="synapse",
            source_event_type=event_type,
            synapse_project_id=state["synapse_project_id"],
            synapse_run_id=state.get("run_id"),
        )

    async def get_state_for_user(self, request_id: str, user_id: str) -> dict[str, Any]:
        state = await self.store.get_state(request_id)
        if not state or state.get("user_id") != user_id:
            raise SynapseRunNotFound(request_id)
        return state

    async def stop_run(self, request_id: str, user_id: str) -> dict[str, Any]:
        state = await self.get_state_for_user(request_id, user_id)
        project_id = state.get("synapse_project_id")
        if project_id and state.get("status") in {"starting", "running"}:
            await self.client.stop_project(str(project_id))
        await self.store.update_state(
            request_id,
            status="cancelled",
            finished_at=datetime.now(UTC).isoformat(),
        )
        if state.get("chat_id"):
            await self.store.release_chat(str(state["chat_id"]), request_id)
        updated = await self.store.get_state(request_id)
        assert updated is not None
        return updated

    @staticmethod
    def _is_durable(event_type: str) -> bool:
        lowered = event_type.lower()
        return not (lowered.endswith(".delta") or lowered.endswith("_delta"))

    @staticmethod
    def _event_text(content: dict[str, Any]) -> str:
        message = content.get("message")
        if isinstance(message, dict):
            for key in ("text", "content", "message"):
                value = message.get(key)
                if isinstance(value, str) and value:
                    return value
        for key in ("text", "message", "content"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value
        return json.dumps(content, ensure_ascii=False, default=str)

    @staticmethod
    def _safe_event_error(content: dict[str, Any]) -> str:
        for key in ("error", "message", "detail"):
            value = content.get(key)
            if isinstance(value, str) and value:
                return value[:1000]
        return "Synapse project failed"

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        if isinstance(exc, SynapseResponseError):
            return f"Synapse returned HTTP {exc.status_code} during {exc.operation}"
        return str(exc)[:1000] or type(exc).__name__
