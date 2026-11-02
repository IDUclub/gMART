from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.agents.a2a.a2a_format import utc_now_rfc3339
from src.agents.a2a.a2a_types import TaskState

A2ATaskData = dict[str, Any]
A2AArtifactData = dict[str, Any]
A2AMessageData = dict[str, Any]


@dataclass
class _TaskStatus:
    state: TaskState
    timestamp: str
    message: A2AMessageData | None = None

    def to_dict(self) -> A2ATaskData:
        result: A2ATaskData = {"state": TaskState(self.state).value}
        result["timestamp"] = self.timestamp
        if self.message:
            result["message"] = self.message
        return result


@dataclass
class _Task:
    id: str
    context_id: str
    status: _TaskStatus
    message: A2AMessageData | None = None
    history: list[A2AMessageData] = field(default_factory=list)
    artifacts: list[A2AArtifactData] = field(default_factory=list)
    metadata: A2ATaskData = field(default_factory=dict)


class A2ATaskStore:
    """
    In-memory storage for A2A tasks.
    Attributes:
        _tasks (dict[str, _Task]): Internal mapping from task id to the task record.
    """

    def __init__(self) -> None:
        """
        A2ATaskStore initialization function.
        """

        self._tasks: dict[str, _Task] = {}

    def create_task(
        self,
        task_id: str,
        context_id: str,
        user_message: A2AMessageData,
        metadata: A2ATaskData | None = None,
    ) -> A2ATaskData:
        """
        Function creates and stores A2A task.
        Args:
            task_id (str): A2A task id.
            context_id (str): A2A context/session id.
            user_message (A2AMessageData): Sanitized user message.
            metadata (A2ATaskData | None): Task metadata.
        Returns:
            A2ATaskData: Serialized task.
        """

        task = _Task(
            id=task_id,
            context_id=context_id,
            status=_TaskStatus(state=TaskState.submitted, timestamp=utc_now_rfc3339()),
            message=copy.deepcopy(user_message),
            history=[copy.deepcopy(user_message)],
            artifacts=[],
            metadata=metadata or {},
        )
        self._tasks[task_id] = task
        return self._to_dict(task)

    def get_task(self, task_id: str) -> A2ATaskData | None:
        """
        Function returns serialized task by id.
        Args:
            task_id (str): A2A task id.
        Returns:
            A2ATaskData | None: Serialized task or None if task was not found.
        """

        task = self._tasks.get(task_id)
        if task is None:
            return None
        return self._to_dict(task)

    def list_tasks(self, include_artifacts: bool = True) -> list[A2ATaskData]:
        """
        Function returns stored tasks.
        Args:
            include_artifacts (bool): Whether artifacts should be included.
        Returns:
            list[A2ATaskData]: Serialized tasks.
        """

        tasks = [self._to_dict(task) for task in self._tasks.values()]
        if not include_artifacts:
            for task in tasks:
                task.pop("artifacts", None)
        return tasks

    def set_status(
        self,
        task_id: str,
        state: TaskState,
        message: A2AMessageData | None = None,
    ) -> A2ATaskData:
        """
        Function updates A2A task status.
        Args:
            task_id (str): A2A task id.
            state (TaskState): New task state.
            message (A2AMessageData | None): Optional status message.
        Returns:
            A2ATaskData: Serialized task status.
        """

        task = self._tasks[task_id]
        task.status = _TaskStatus(
            state=state,
            message=copy.deepcopy(message),
            timestamp=utc_now_rfc3339(),
        )
        if message is not None:
            task.history.append(copy.deepcopy(message))
        return task.status.to_dict()

    def add_or_append_artifact(
        self,
        task_id: str,
        artifact: A2AArtifactData,
        append: bool = False,
    ) -> A2AArtifactData:
        """
        Function adds or appends artifact to stored task.
        Args:
            task_id (str): A2A task id.
            artifact (A2AArtifactData): Artifact data.
            append (bool): Whether parts should be appended to existing artifact.
        Returns:
            A2AArtifactData: Stored artifact.
        """

        task = self._tasks[task_id]
        artifacts = task.artifacts
        artifact_id = artifact["artifactId"]

        if append:
            for existing in artifacts:
                if existing.get("artifactId") == artifact_id:
                    existing.setdefault("parts", []).extend(
                        copy.deepcopy(artifact.get("parts", []))
                    )
                    return copy.deepcopy(existing)

        artifacts.append(copy.deepcopy(artifact))
        return copy.deepcopy(artifact)

    @staticmethod
    def _to_dict(task: _Task) -> A2ATaskData:
        """
        Function serializes a task to A2A v0.3.0 format.
        Args:
            task (_Task): Stored task.
        Returns:
            A2ATaskData: Serialized task (empty collections are left out).
        """

        result: A2ATaskData = {
            "id": task.id,
            "contextId": task.context_id,
            "status": task.status.to_dict(),
        }
        if task.message:
            result["message"] = task.message
        if task.history:
            result["history"] = task.history
        if task.artifacts:
            result["artifacts"] = task.artifacts
        if task.metadata:
            result["metadata"] = task.metadata
        result = copy.deepcopy(result)
        result["kind"] = "task"
        return result
