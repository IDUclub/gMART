"""A2A adapters share the existing gMART task/event wire format."""

from contextlib import aclosing

from python_a2a.models.task import TaskState

from src.agents.__version__ import APP_VERSION
from src.agents.a2a.a2a_format import (
    scenario_context_extension,
    synapse_compatible_agent_card,
)
from src.agents.a2a.scenario_data_executor import ScenarioDataAgentExecutor
from src.agents.a2a.task_store import A2ATaskStore
from src.agents.services.scenario_data.scenario_data_a2a_service import (
    ScenarioDataA2AService,
)


class PlanningAgentCard:
    def __init__(self, profile):
        self.profile = profile

    def get_agent_card(self, base_url):
        p = self.profile
        return synapse_compatible_agent_card(
            {
                "name": p.key + "-agent",
                "description": p.description,
                "url": base_url.rstrip("/") + "/" + p.key + "/a2a",
                "version": APP_VERSION,
                "protocolVersion": "0.3.0",
                "preferredTransport": "JSONRPC",
                "capabilities": {
                    "streaming": True,
                    "pushNotifications": False,
                    "extensions": [scenario_context_extension(required=False)],
                },
                "defaultInputModes": ["text/plain", "application/json"],
                "defaultOutputModes": [
                    "text/plain",
                    "application/geo+json",
                    "application/json",
                ],
                "skills": [
                    {
                        "id": p.key,
                        "name": p.title,
                        "description": p.description,
                        "tags": [p.key, "urban-planning", "geojson"],
                    }
                ],
            }
        )


class PlanningExecutor(ScenarioDataAgentExecutor):
    def __init__(self, service, task_store):
        self.service = service
        self.task_store = task_store

    async def _run_pipeline(self, execution, urban_mcp_client, token):
        tid, cid = execution["task_id"], execution["context_id"]
        state = self.task_store.set_status(tid, TaskState.WAITING)
        yield self._status_update(tid, cid, state, final=False)
        try:
            pipeline = self.service.run(
                token=token,
                user_query=execution["user_query"],
                scenario_id=execution["scenario_id"],
                model=execution["model"],
                temperature=execution["temperature"],
                request_id=tid,
                input_artifacts=execution["metadata"].get("input_artifacts"),
                urban_mcp_client=urban_mcp_client,
            )
            async with aclosing(pipeline):
                async for item in pipeline:
                    if (
                        self.task_store.get_task(tid)["status"]["state"]
                        == TaskState.CANCELED.value
                    ):
                        return
                    if item["type"] in {"clarification", "error"}:
                        content = item["content"]
                        status = (
                            TaskState.INPUT_REQUIRED
                            if item["type"] == "clarification"
                            else TaskState.FAILED
                        )
                        state = self.task_store.set_status(
                            tid,
                            status,
                            self._agent_message(
                                cid,
                                tid,
                                content.get("question") or content.get("message", ""),
                            ),
                        )
                        yield self._status_update(tid, cid, state, final=True)
                        return
                    event = self._pipeline_item_to_event(tid, cid, item)
                    if event:
                        yield event
            state = self.task_store.set_status(tid, TaskState.COMPLETED)
            yield self._status_update(tid, cid, state, final=True)
        except Exception as exc:
            state = self.task_store.set_status(
                tid, TaskState.FAILED, self._agent_message(cid, tid, str(exc))
            )
            yield self._status_update(tid, cid, state, final=True)

    def _text_artifact(self, text, append):
        artifact = super()._text_artifact(text, append)
        artifact.update(
            artifactId=self.service.profile.key + "-text",
            name=self.service.profile.title,
            description=self.service.profile.description,
        )
        return artifact


class PlanningA2AService(ScenarioDataA2AService):
    def __init__(self, service, task_store=None):
        self.agent = PlanningAgentCard(service.profile)
        self.task_store = task_store or A2ATaskStore()
        self.executor = PlanningExecutor(service, self.task_store)
