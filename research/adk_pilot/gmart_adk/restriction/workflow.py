"""The restriction pipeline as an ADK ``Workflow``.

Same steps and the same model calls as the restrictions branch of
``RestrictionParserService._run_restriction_execution_pipline`` (no compliance,
NormGraph lookup, chat persistence or reconnect), but the control flow lives
in the graph instead of in ``if … return`` and hand-written repair calls:

    START → load_catalogs → planner → check_plan ─┬─ semantic_repair → planner_semantic_repair → check_plan
                                                  ├─ names_repair    → planner_names_repair    → check_plan
                                                  ├─ clarify         → clarification
                                                  └─ explain         → announce_explanation → explainer
                    → announce_layers → retrieve_layers → announce_buffers → build_buffers ─┬─ buffers_only → report_buffers
                                                                                            └─ restrictions → announce_restrictions
                                                         → intersect → responder → finish_response

The domain logic (catalog canonicalization, semantic checks, repair prompts,
MCP tool plans, context building) is reused from ``src.agents`` unchanged.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from google.adk import Agent, Context, Event, Workflow
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.workflow import START, FunctionNode, RetryConfig

from research.adk_pilot.gmart_adk.events import chunk, gmart_event
from research.adk_pilot.gmart_adk.harness import (
    USER_QUERY,
    conversation,
    request_params,
)
from research.adk_pilot.gmart_adk.llm import GmartGenerateContentConfig, GmartLlm
from research.adk_pilot.gmart_adk.restriction.prompts import (
    final_response_prompt,
    plan_explanation_prompt,
)
from src.agents.services.restriction.restriction_catalog import (
    RestrictionPlanBuilder,
    normalize_name,
)
from src.agents.services.restriction.restriction_context import (
    RestrictionContextBuilder,
)
from src.agents.services.restriction.restriction_parser_service import (
    RestrictionParserService,
)
from src.agents.services.restriction.restriction_tool_executor import (
    RestrictionToolExecutor,
)
from src.agents.services.service_entities.restriction_plan import (
    RestrictionPlan,
    RestrictionTaskMode,
)

# Pure event and summary helpers of the current service, reused for parity.
_legacy = RestrictionParserService
status = _legacy._status
tool_call = _legacy._tool_call

# Session-state keys of this workflow (besides the harness ones).
SCENARIO_ID = "scenario_id"
NORMGRAPH = "normgraph_restrictions"
SERVICES = "services_catalog"
OBJECTS = "physical_objects_catalog"
RAW_PLAN = "raw_plan"
PLAN_PHASE = "plan_phase"
PLAN_ISSUES = "plan_issues"
UNRESOLVED = "unresolved_names"
PLAN = "plan"
LAYERS = "layers"
BUFFERS = "buffers"
CONTEXT = "restriction_context"
FINAL_TEXT = "final_text"

# Plan-check phases: the order RestrictionPlanBuilder.build_plan repairs in.
INITIAL, SEMANTIC_DONE, NAMES_DONE = "initial", "semantic_done", "names_done"

UNCLEAR_TARGETS_QUESTION = (
    "Не удалось однозначно определить целевые объекты проверки. "
    "Уточните, какие объекты нужно проверить на пересечение с буферной зоной."
)

# Request-scoped clients that cannot live in session state.
current_mcp_client: ContextVar[Any] = ContextVar("current_mcp_client")


@dataclass
class RestrictionWorkflowDeps:
    llm: GmartLlm
    plan_builder: RestrictionPlanBuilder
    tool_executor: RestrictionToolExecutor
    context_builder: RestrictionContextBuilder
    # Retry policy of the MCP steps; None keeps today's behaviour (no retry).
    tool_retry: RetryConfig | None = None


def _plan_config() -> GmartGenerateContentConfig:
    # The options RestrictionPlanBuilder._request_plan sends, as one object.
    return GmartGenerateContentConfig(
        temperature=0,
        max_output_tokens=4096,
        think=False,
        backend_options={"num_ctx": 16384},
        json_repair_attempts=2,
    )


def _plan(state: Any) -> RestrictionPlan:
    return RestrictionPlan.model_validate(state[PLAN])


def build_restriction_workflow(deps: RestrictionWorkflowDeps) -> Workflow:
    builder = deps.plan_builder

    # --- planning -----------------------------------------------------------

    async def load_catalogs(ctx: Context, node_input: Any):
        yield gmart_event(
            status(
                "data_retrievement", "Получаю каталоги сервисов и физических объектов"
            )
        )
        services, objects = await builder.get_entity_catalogs(
            current_mcp_client.get(), ctx.state[SCENARIO_ID]
        )
        ctx.state[SERVICES] = services
        ctx.state[OBJECTS] = objects
        ctx.state[PLAN_PHASE] = INITIAL

    def planner_instruction(ctx: ReadonlyContext) -> str:
        state = ctx.state
        return builder._build_prompt(
            state[SERVICES], state[OBJECTS], state.get(NORMGRAPH) or []
        )

    def semantic_repair_instruction(ctx: ReadonlyContext) -> str:
        state = ctx.state
        return builder._build_semantic_repair_prompt(
            state[USER_QUERY],
            RestrictionPlan.model_validate(state[RAW_PLAN]),
            state[PLAN_ISSUES],
            state[SERVICES],
            state[OBJECTS],
            state.get(NORMGRAPH) or [],
        )

    def names_repair_instruction(ctx: ReadonlyContext) -> str:
        state = ctx.state
        return builder._build_repair_prompt(
            state[USER_QUERY],
            RestrictionPlan.model_validate(state[RAW_PLAN]),
            state[UNRESOLVED],
            state[SERVICES],
            state[OBJECTS],
            state.get(NORMGRAPH) or [],
        )

    def plan_agent(name: str, instruction, sees_conversation: bool) -> Agent:
        return Agent(
            name=name,
            model=deps.llm,
            instruction=instruction,
            output_schema=RestrictionPlan,
            generate_content_config=_plan_config(),
            before_model_callback=[
                request_params(),
                # Repair calls carry their whole context in the system prompt.
                conversation(history=sees_conversation, query=sees_conversation),
            ],
        )

    planner = plan_agent("planner", planner_instruction, sees_conversation=True)
    planner_semantic_repair = plan_agent(
        "planner_semantic_repair", semantic_repair_instruction, sees_conversation=False
    )
    planner_names_repair = plan_agent(
        "planner_names_repair", names_repair_instruction, sees_conversation=False
    )

    def check_plan(ctx: Context, node_input: dict):
        state = ctx.state
        query, services, objects = state[USER_QUERY], state[SERVICES], state[OBJECTS]
        normgraph = state.get(NORMGRAPH) or []
        raw = RestrictionPlan.model_validate(node_input)
        plan = builder.validate_and_canonicalize_plan(raw, query, services, objects)
        plan = builder._ground_normgraph_rules(plan, normgraph)
        state[RAW_PLAN] = raw.model_dump(mode="json")

        phase = state[PLAN_PHASE]
        if phase == INITIAL:
            issues = builder._find_semantic_issues(plan, query, normgraph)
            if issues:
                state[PLAN_PHASE] = SEMANTIC_DONE
                state[PLAN_ISSUES] = issues
                yield Event(route="semantic_repair")
                return
            phase = SEMANTIC_DONE
        if phase == SEMANTIC_DONE:
            unresolved = builder._find_unresolved_names(raw, services, objects)
            if unresolved:
                state[PLAN_PHASE] = NAMES_DONE
                state[UNRESOLVED] = unresolved
                yield Event(route="names_repair")
                return

        if builder._find_semantic_issues(plan, query, normgraph):
            plan = plan.model_copy(
                update={
                    "mode": RestrictionTaskMode.NEEDS_CLARIFICATION,
                    "target_entities": [],
                    "restriction_rules": [],
                    "clarification_question": UNCLEAR_TARGETS_QUESTION,
                }
            )
        if plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION:
            plan = builder._enrich_clarification(plan, services, objects)
        state[PLAN] = plan.model_dump(mode="json")
        clarify = plan.mode == RestrictionTaskMode.NEEDS_CLARIFICATION
        yield Event(route="clarify" if clarify else "explain")

    def clarification(ctx: Context):
        plan = _plan(ctx.state)
        yield gmart_event(
            status("context_preparation", "Нужно уточнение параметров запроса."),
            {
                "type": "clarification",
                "content": {
                    "question": plan.clarification_question
                    or "Уточните параметры запроса."
                },
            },
        )

    # --- explanation --------------------------------------------------------

    def announce_explanation():
        yield gmart_event(
            status("plan_explanation", "Объясняю, почему выбраны эти параметры")
        )

    def explainer_instruction(ctx: ReadonlyContext) -> str:
        return plan_explanation_prompt(_legacy._plan_summary(_plan(ctx.state)))

    explainer = Agent(
        name="explainer",
        model=deps.llm,
        instruction=explainer_instruction,
        generate_content_config=GmartGenerateContentConfig(think=False),
        before_model_callback=[
            request_params(temperature=lambda t: min(t, 0.4)),
            conversation(),
        ],
    )

    # --- geometry -----------------------------------------------------------

    def announce_layers():
        yield gmart_event(
            chunk("\n\n", done=False),
            status(
                "data_retrievement", "Получаю необходимые слои по утверждённому плану"
            ),
        )

    async def retrieve_layers(ctx: Context):
        plan = _plan(ctx.state)
        result = await deps.tool_executor.retrieve_layers_for_plan(
            current_mcp_client.get(), plan, ctx.state[SCENARIO_ID]
        )
        layers = result.tool_result
        ctx.state[LAYERS] = layers
        visible = layers
        if plan.mode == RestrictionTaskMode.RESTRICTIONS:
            # Targets are published only after checking their intersections.
            sources = {normalize_name(entity.name) for entity in plan.source_entities}
            visible = {
                name: layer
                for name, layer in layers.items()
                if normalize_name(name) in sources
            }
        yield gmart_event(
            tool_call("data_retrievement", result.tool_calls, mcp_source="IDU_MCP_URL"),
            *_legacy._feature_collections(visible),
        )

    def announce_buffers():
        yield gmart_event(
            status("buffer_creation", "Начинаю построение буферов зон с ограничениями")
        )

    async def build_buffers(ctx: Context):
        plan = _plan(ctx.state)
        result = await deps.tool_executor.run_buffer_plan(
            current_mcp_client.get(), plan, ctx.state[LAYERS]
        )
        ctx.state[BUFFERS] = result.tool_result
        yield gmart_event(
            tool_call("buffer_creation", result.tool_calls, mcp_source="IDU_MCP_URL"),
            status("buffer_creation", "Построил необходимые буферы с ограничениями."),
        )
        yield Event(route=plan.mode.value)

    def report_buffers(ctx: Context):
        buffers = ctx.state[BUFFERS]
        # A buffer count is not a count of affected objects.
        counts = [
            f"«{name}»: {len(layer.get('features', []))}"
            for name, layer in buffers.items()
        ]
        yield gmart_event(
            *_legacy._feature_collections(buffers),
            chunk(
                "Построены буферы. Количество геометрий по слоям: "
                + "; ".join(counts)
                + ". Пересечения с целевыми объектами в этом расчёте не проверялись.",
                done=True,
            ),
        )

    def announce_restrictions():
        yield gmart_event(
            status(
                "restriction_formation", "Начинаю извлечение нормативных ограничений."
            )
        )

    async def intersect(ctx: Context):
        plan = _plan(ctx.state)
        result = await deps.tool_executor.run_restriction_plan(
            current_mcp_client.get(), plan, ctx.state[LAYERS], ctx.state[BUFFERS]
        )
        restrictions = result.tool_result
        ctx.state[CONTEXT] = await deps.context_builder.generate_restrictions_context(
            restrictions["generators"], restrictions["objects"]
        )
        yield gmart_event(
            tool_call(
                "restriction_formation", result.tool_calls, mcp_source="IDU_MCP_URL"
            ),
            status(
                "restriction_formation", "Извлечение нормативных ограничений завершено."
            ),
            *_legacy._feature_collections(restrictions),
        )

    # --- answer -------------------------------------------------------------

    responder = Agent(
        name="responder",
        model=deps.llm,
        instruction=lambda ctx: final_response_prompt(ctx.state[CONTEXT]),
        output_key=FINAL_TEXT,
        generate_content_config=GmartGenerateContentConfig(think=False),
        before_model_callback=[request_params(temperature=lambda t: t), conversation()],
    )

    def finish_response(ctx: Context):
        items = []
        if not ctx.state.get(FINAL_TEXT):
            items.append(
                chunk(_legacy._fallback_final_response(ctx.state[CONTEXT]), done=False)
            )
        items.append(chunk("", done=True))
        yield gmart_event(*items)

    def tool_node(func) -> FunctionNode:
        return FunctionNode(func=func, retry_config=deps.tool_retry)

    retrieve_layers_node = tool_node(retrieve_layers)
    build_buffers_node = tool_node(build_buffers)
    intersect_node = tool_node(intersect)

    return Workflow(
        name="restriction_workflow",
        edges=[
            (START, load_catalogs, planner, check_plan),
            (
                check_plan,
                {
                    "semantic_repair": planner_semantic_repair,
                    "names_repair": planner_names_repair,
                    "clarify": clarification,
                    "explain": announce_explanation,
                },
            ),
            (planner_semantic_repair, check_plan),
            (planner_names_repair, check_plan),
            (
                announce_explanation,
                explainer,
                announce_layers,
                retrieve_layers_node,
                announce_buffers,
                build_buffers_node,
            ),
            (
                build_buffers_node,
                {
                    RestrictionTaskMode.BUFFERS_ONLY.value: report_buffers,
                    RestrictionTaskMode.RESTRICTIONS.value: announce_restrictions,
                },
            ),
            (announce_restrictions, intersect_node, responder, finish_response),
        ],
    )


STREAMED_AGENTS = ("explainer", "responder")
