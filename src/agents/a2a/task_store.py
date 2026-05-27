from __future__ import annotations

import copy
from typing import Any

from python_a2a.models.task import Task, TaskState, TaskStatus

_WAITING_STATE = TaskState.WAITING.value

A2ATaskData = dict[str, Any]
A2AArtifactData = dict[str, Any]
A2AMessageData = dict[str, Any]


class A2ATaskStore:
    """
    In-memory storage for A2A tasks.
    Attributes:
        _tasks (dict[str, Task]): Internal mapping from task id to python-a2a Task.
    """

    def __init__(self) -> None:
        """
        A2ATaskStore initialization function.
        """

        self._tasks: dict[str, Task] = {}

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

        task = Task(
            id=task_id,
            session_id=context_id,
            status=TaskStatus(state=TaskState.SUBMITTED),
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
        task.status = TaskStatus(state=state, message=copy.deepcopy(message))
        if message is not None:
            task.history.append(copy.deepcopy(message))
        status_dict = task.status.to_dict()
        if status_dict.get("state") == _WAITING_STATE:
            status_dict["state"] = "working"
        return status_dict

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
    def _to_dict(task: Task) -> A2ATaskData:
        """
        Function serializes python-a2a task to A2A v0.3.0 format.
        Args:
            task (Task): A2A task instance.
        Returns:
            A2ATaskData: Serialized task.
        """

        result = copy.deepcopy(task.to_dict())
        result["kind"] = "task"
        # python_a2a uses legacy "sessionId"; A2A v0.3.0 spec requires "contextId"
        if "sessionId" in result:
            result["contextId"] = result.pop("sessionId")
        # python_a2a uses "waiting"; A2A v0.3.0 spec uses "working"
        if result.get("status", {}).get("state") == _WAITING_STATE:
            result["status"]["state"] = "working"
        return result
