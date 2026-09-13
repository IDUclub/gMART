"""Synthetic negative probes. True means a reproduced defect, not a passing check.

No network or real user data. Exit 1 when an unsafe acceptance is reproduced.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import fakeredis.aioredis

from src.agents.common.exceptions.base_exceptions import AgentsNotFound
from src.agents.services.orchestrator.analysis_context import AnalysisContext
from src.agents.services.orchestrator.analysis_goal import GoalDecision, GoalState
from src.agents.services.orchestrator.analysis_support import missing_input
from src.agents.services.orchestrator.orchestrator_service import OrchestratorService
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus
from src.agents.services.service_entities.orchestrator_plan import OrchestratorStep
from tests.unit.test_analytical_orchestrator import table
from tests.unit.test_goal_orchestrator import contract, school_artifacts


def goal_probes():
    goal = contract().model_copy(update={"requirements": contract().requirements[1:]})
    context = AnalysisContext()
    state = GoalState(context, goal)
    step = OrchestratorStep(
        agent="provision", task="Calculate", requirement_id="provision"
    )
    for i in range(2):
        state.record(step, str(i), "failed", missing_input("service", "Unavailable"))
        state.resume()
    try:
        state.validate_decision(
            GoalDecision(
                action="blocked", missing=[missing_input("service", "Unavailable")]
            ),
            {"provision"},
        )
        skips_retry = True
    except ValueError:
        skips_retry = False

    context = AnalysisContext()
    state = GoalState(context, goal)
    empty = table()
    empty["content"].update(rows=[], total_rows=0)
    context.add_artifact(empty, 1, "empty")
    context.finish(1, "Calculate", 772, "completed", "", "empty")
    state.record(step, "empty", "completed")
    empty_satisfied = state.progress()[0]["status"] == "satisfied"

    context = AnalysisContext()
    state = GoalState(context, contract())
    t, layer = school_artifacts()
    t["content"]["rows"] = [{"service_id": 101}]
    layer["content"]["feature_collection"]["features"][0]["properties"] = {
        "service_id": 202
    }
    for event in (t, layer):
        context.add_artifact(event, 1, "mismatch")
    context.finish(1, "schools", 772, "completed", "", "mismatch")
    state.record(
        OrchestratorStep(
            agent="scenario_data", task="schools", requirement_id="schools"
        ),
        "mismatch",
        "completed",
    )
    return {
        "resumed_goal_can_block_without_new_attempt_after_two_historical_failures": skips_retry,
        "empty_calculation_table_satisfies_requirement": empty_satisfied,
        "same_count_but_different_service_ids_satisfy_requirement": state.progress()[0][
            "status"
        ]
        == "satisfied",
    }


async def replay_probe():
    db = fakeredis.aioredis.FakeRedis(decode_responses=True)
    service = object.__new__(OrchestratorService)
    service.state_store = PipelineStateStore(db)
    try:
        await service.state_store.create(
            "synthetic-run-a",
            chat_id="synthetic-user-a-chat",
            user_query="Synthetic private query",
            scenario_id=772,
            model="m",
            temperature=0,
        )
        event = {
            "type": "orchestrator_final",
            "content": {"answer": "Synthetic owner A result"},
        }
        await service.state_store.buffer_event("synthetic-run-a", event)
        await service.state_store.set_status("synthetic-run-a", PipelineStatus.DONE)
        result = [
            e
            async for e in service.run_orchestration_pipeline(
                idu_mcp_client=None,
                effects_mcp_client=None,
                dvd_mcp_client=None,
                normgraph_mcp_client=None,
                token="different-user-b",
                model="m",
                temperature=0,
                user_query="replay",
                request_id="synthetic-run-a",
            )
        ]
        return result == [event]
    except AgentsNotFound:
        return False
    finally:
        await db.aclose()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = goal_probes()
    result["different_subject_can_replay_known_request_id"] = await replay_probe()
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return int(any(result.values()))


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
