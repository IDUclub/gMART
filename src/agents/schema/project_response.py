from typing import Any, Literal

from pydantic import BaseModel


class ProjectResponse(BaseModel):
    """SSE envelope of the project creation from files.

    ``clarification`` carries ``kind`` (reuse_files, entity_mapping, region, profile,
    existing_objects, retry) and numbered ``options``; the answer is the next message
    of the same chat. ``project_created`` / ``scenario_created`` arrive as soon as
    Urban API returns the ids, ``project_summary`` once the objects are saved.
    """

    type: Literal[
        "pipeline_started",
        "service_event",
        "status",
        "chunk",
        "warning",
        "error",
        "clarification",
        "user_files",
        "project_created",
        "scenario_created",
        "project_summary",
    ]
    content: dict[str, Any]
