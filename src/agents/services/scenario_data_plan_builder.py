from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.agents.mcp_clients.urban_mcp_client import (
    URBAN_MCP_GROUP_DESCRIPTIONS,
    URBAN_MCP_GROUPS,
    UrbanMcpTool,
)
from src.agents.services.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.scenario_data_action import (
    ScenarioDataAction,
    ScenarioDataActionKind,
)
from src.agents.services.service_entities.scenario_data_plan import (
    AcquisitionPlan,
    ExecutionPlanRevision,
    PlanStep,
    PlanStepKind,
)

MAX_SCENARIO_TOOL_CALLS = 6
MAX_PLANNER_RETRIES = 2
#: Answer budget for the planner. The retry budget is deliberately much larger: an empty
#: reply usually means the reasoning trace consumed everything before any content was emitted.
#: How many tools the planner is shown. The window is the binding constraint, not the model's
#: ability to choose: gpt-oss is served with a 16k context, and 41 tools cost 10.3k prompt
#: tokens, leaving too little for the reasoning channel *and* a final message. Measured on the
#: same question at reasoning_effort=medium — 41 tools: no content, whether the budget ran out
#: (finish=length) or the model gave up (finish=stop); 12 tools: answers on a 3k budget.
SHORTLIST_SIZE = 12
#: gpt-oss spends this budget on its reasoning channel before any answer: a measured planner
#: retry exhausted 3000 tokens and produced only a 14723-character reasoning trace. The retry
#: budget therefore leaves enough room for that trace *and* the small JSON answer while still
#: fitting beside the roughly 4k-token prompt in the 16k context window.
PLANNER_NUM_PREDICT = 2500
PLANNER_NUM_PREDICT_RETRY = 5000
#: Effort used on a retry after an empty reply. "low" is exactly the value that produces no
#: content on a Harmony-served gpt-oss, so escalating to "medium" is the fix, not a guess.
PLANNER_RETRY_REASONING_EFFORT = "medium"

WORKSPACE_TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "WorkspaceDescribe": {"required": ["handle"]},
    "WorkspaceUniqueValues": {"required": ["handle", "column"], "optional": ["limit"]},
    "WorkspaceSample": {"required": ["handle"], "optional": ["limit"]},
    "WorkspaceFilter": {"required": ["handle", "conditions"]},
    "WorkspaceSelect": {"required": ["handle", "columns"]},
    "WorkspaceSort": {"required": ["handle", "columns"], "optional": ["ascending"]},
    "WorkspaceDeduplicate": {
        "required": ["handle", "columns"],
        "optional": ["keep"],
    },
    "WorkspaceAggregate": {
        "required": ["handle", "group_by", "aggregations"],
    },
    "WorkspaceJoinMapping": {
        "required": ["handle", "mapping_handle", "left_on", "right_on"],
        "optional": ["how"],
    },
    "WorkspaceSpatialFilter": {
        "required": ["handle", "mask_handle"],
        "optional": ["predicate"],
    },
    "WorkspaceToFeatureCollection": {
        "required": ["handle"],
        "optional": ["limit"],
    },
}

#: Subjects a tool can be *about* that a general data question is not asking for. These tools
#: stay in the catalogue — restriction zones are legitimate Urban API data and must remain
#: answerable — but they must not outrank an on-topic tool merely by sharing generic words.
#: "Получить зоны ограничений объектов на территории" matches "объекты" and "территория" in its
#: title, so on a question about which objects exist it scored level with the plain objects
#: tool, and the model was free to pick either.
_TOPIC_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ограничен", ("ограничен", "буфер", "зон", "buffer", "restrict")),
    ("буфер", ("ограничен", "буфер", "буферн", "buffer")),
    ("показател", ("показател", "индикатор", "indicator", "значени")),
    ("социальн", ("социальн", "soc_group", "ценност")),
    ("норматив", ("норматив", "норм")),
)
#: Enough to drop an off-subject tool below an on-subject one without hiding it outright.
_OFF_TOPIC_PENALTY = 4


def _off_topic_penalty(tool: UrbanMcpTool, context: str) -> int:
    """Penalise a tool whose declared subject the question never mentions."""

    title = tool.title.lower()
    for marker, query_words in _TOPIC_MARKERS:
        if marker in title and not any(word in context for word in query_words):
            return _OFF_TOPIC_PENALTY
    return 0


class ScenarioDataPlanBuilder:
    """Select one grounded Urban MCP action at a time."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def build_acquisition_plan(
        self,
        model: str,
        user_query: str,
        history: list[dict] | None,
        scenario_id: int | None,
        context: dict[str, Any] | None = None,
        project_id: int | None = None,
    ) -> AcquisitionPlan:
        """Decide what must be known before selecting concrete tools."""

        prompt = f"""Ты составляешь логический план получения городских данных.
Сначала определи необходимые факты, маппинги и критерии полноты. Не называй MCP-инструменты
и не придумывай идентификаторы. Если задача неоднозначна настолько, что разные трактовки
дадут разные результаты, заполни clarification одним коротким вопросом на русском языке.
История сообщений перед текущей репликой — обязательная часть задачи. Короткие ответы вроде
«да», «оба», «эти», «первый вариант» разрешай по последнему вопросу и предмету диалога.
Не меняй предметную область на нормативы, ограничения, показатели или другую тему, если
пользователь явно не переключил её в текущей реплике либо в недавней истории.
Пользователь оперирует названиями и не обязан знать внутренние ID. Для каждого названного
типа или объекта сохраняй домен mapping (например service_type или physical_object_type),
направление name_to_id и исходное название. Никогда не превращай service_type_id в
бездоменный type_id и не проси пользователя самостоятельно искать идентификатор.

Сценарий: {scenario_id if scenario_id is not None else "не выбран"}.
Проект выбранного сценария: {project_id if project_id is not None else "не определён"}.
Сценарий является конкретным вариантом проекта. В вопросах о территории, расположении и
параметрах «проекта» используй проект выбранного сценария. При этом ID различны:
scenario_id передавай только в scenario-параметры, project_id — только в project-параметры.
Сжатый подтверждённый контекст чата:
{json.dumps(context or {}, ensure_ascii=False)[:12000]}

Ответ должен соответствовать JSON Schema. requirement_id — короткий латинский идентификатор.
Для mapping_needs укажи domain, direction name_to_id/id_to_name и известные values.
required_output перечисляет ожидаемые таблицы, слои и обязательные поля. Не включай
внутренние рассуждения, только проверяемый план."""
        return await self._structured_plan_call(
            model,
            [
                {"role": "system", "content": prompt},
                *(history or []),
                {"role": "user", "content": user_query},
            ],
            AcquisitionPlan,
            "acquisition plan",
            post_validate=lambda plan: self._validate_acquisition_topic(
                plan, user_query, history or []
            ),
        )

    async def build_execution_plan(
        self,
        model: str,
        user_query: str,
        acquisition: AcquisitionPlan,
        tools: list[UrbanMcpTool],
        mappings: list[dict[str, Any]],
        *,
        scenario_id: int | None,
        project_id: int | None = None,
        revision: int = 1,
        reason: str = "initial plan",
        observations: list[dict[str, Any]] | None = None,
        completed_fingerprints: list[str] | None = None,
        completed_step_ids: list[str] | None = None,
        workspace_enabled: bool = False,
    ) -> ExecutionPlanRevision:
        """Bind a logical plan to exact catalogue tools and ordered arguments."""

        resolved_query = self._resolved_planning_query(user_query, acquisition)
        shortlist = self._shortlist(tools, resolved_query, observations or [])
        catalog = [tool.compact_prompt_entry() for tool in shortlist]
        prompt = f"""Построй целиком исполнимый план read-only Urban MCP до начала выполнения.
Используй только точные group/tool_name и параметры из каталога. Шаги идут строго
последовательно; depends_on может ссылаться только на предыдущие step_id. parallel_group
можно заполнить для независимых шагов на будущее, но порядок списка остаётся безопасным.
Не повторяй уже выполненные fingerprints и ранее завершённые step_id. scenario_id
и project_id подставляются системой, не меняй и не смешивай их. Пользователь не знает
внутренние ID: бери их только из актуальных маппингов ниже. Сохраняй домен пары name/id:
service_type.id можно передавать только как service_type_id/service_type_ids, а
physical_object_type.id — только как physical_object_type_id/physical_object_type_ids.
Если пользователь говорит «проект» при выбранном сценарии, это проект данного сценария;
для scenario-инструмента всё равно используй scenario_id, для project-инструмента project_id.
Для запроса о расположении именованного типа сначала получи записи сценария этого домена,
затем отфильтруй их по подтверждённому ID. Один справочник типов не доказывает расположение.
{self._workspace_prompt(workspace_enabled)}

Текущая реплика: {user_query}
Задача, уже восстановленная по истории: {resolved_query}
Логический план: {acquisition.model_dump_json()}
Актуальные маппинги: {json.dumps(mappings, ensure_ascii=False)[:10000]}
Наблюдения: {json.dumps(observations or [], ensure_ascii=False)[:12000]}
Выполненные fingerprints: {json.dumps(completed_fingerprints or [], ensure_ascii=False)}
Ранее завершённые step_id, на чьи artifact можно ссылаться:
{json.dumps(completed_step_ids or [], ensure_ascii=False)}
Каталог: {json.dumps(catalog, ensure_ascii=False)}
Сценарий: {scenario_id if scenario_id is not None else "не выбран"}
Проект сценария: {project_id if project_id is not None else "не определён"}
Требуемая revision: {revision}; reason: {reason}.

Верни JSON по схеме ExecutionPlanRevision. Не завершай план до получения всех данных и
справочников, необходимых для required_output. Каждый requirement_id логического плана
должен встречаться в satisfies хотя бы одного шага, который реально его закрывает.
Не переопределяй предмет задачи по короткой текущей реплике: логический план выше уже
учёл историю диалога и является источником истины для выбора инструментов."""

        def validate_plan(plan: ExecutionPlanRevision) -> ExecutionPlanRevision:
            plan = plan.model_copy(update={"revision": revision, "reason": reason})
            if acquisition.required_output.layers:
                available_tools = {(tool.group, tool.name) for tool in tools}
                upgraded_steps = []
                for step in plan.steps:
                    geometry_name = f"{step.tool_name}WithGeometry"
                    if (
                        step.kind == PlanStepKind.URBAN_TOOL
                        and step.group is not None
                        and not step.tool_name.endswith("WithGeometry")
                        and (step.group, geometry_name) in available_tools
                    ):
                        step = step.model_copy(update={"tool_name": geometry_name})
                    upgraded_steps.append(step)
                plan = plan.model_copy(update={"steps": upgraded_steps})
            if acquisition.requirements and not plan.steps:
                raise ValueError("execution plan has requirements but no steps")
            urban_count = sum(
                step.kind == PlanStepKind.URBAN_TOOL for step in plan.steps
            )
            workspace_count = sum(
                step.kind == PlanStepKind.WORKSPACE for step in plan.steps
            )
            scenario_scoped = scenario_id is not None and (
                "scenario" in acquisition.objective.lower()
                or "сценар" in acquisition.objective.lower()
                or "project" in acquisition.objective.lower()
                or "проект" in acquisition.objective.lower()
                or "сценар" in user_query.lower()
                or "проект" in user_query.lower()
            )
            if revision == 1 and scenario_scoped and urban_count == 0:
                scenario_steps = self._scenario_seed_steps(
                    acquisition, tools, scenario_id
                )
                if not scenario_steps:
                    raise ValueError(
                        "initial scenario-scoped plan must call an Urban MCP scenario "
                        "tool; a global mapping artifact cannot prove scenario membership"
                    )
                covered = {
                    requirement
                    for step in scenario_steps
                    for requirement in step.satisfies
                }
                remaining = [
                    step
                    for step in plan.steps
                    if not (
                        step.kind == PlanStepKind.WORKSPACE
                        and step.satisfies
                        and set(step.satisfies).issubset(covered)
                    )
                ]
                plan = plan.model_copy(update={"steps": [*scenario_steps, *remaining]})
                urban_count = len(scenario_steps)
                workspace_count = sum(
                    step.kind == PlanStepKind.WORKSPACE for step in remaining
                )
            if urban_count > 10 or workspace_count > 20:
                raise ValueError(
                    "execution plan exceeds call budgets: "
                    f"urban={urban_count}, workspace={workspace_count}"
                )
            return self._canonicalize_plan(
                plan,
                tools,
                workspace_enabled=workspace_enabled,
                completed_step_ids=set(completed_step_ids or []),
            )

        if revision == 1 and scenario_id is not None:
            scenario_steps = self._scenario_seed_steps(acquisition, tools, scenario_id)
            covered = {
                requirement_id
                for step in scenario_steps
                for requirement_id in step.satisfies
            }
            required = {
                requirement.requirement_id for requirement in acquisition.requirements
            }
            if (
                scenario_steps
                and required.issubset(covered)
                and any(
                    step.tool_name.endswith("WithGeometry") for step in scenario_steps
                )
            ):
                logger.info(
                    "Using deterministic scenario-data geometry plan; "
                    "all requirements are covered by scoped Urban tools"
                )
                return validate_plan(
                    ExecutionPlanRevision(
                        revision=revision,
                        reason=reason,
                        objective=acquisition.objective,
                        steps=scenario_steps,
                        required_output=acquisition.required_output,
                    )
                )

        try:
            return await self._structured_plan_call(
                model,
                [{"role": "system", "content": prompt}],
                ExecutionPlanRevision,
                "execution plan",
                post_validate=validate_plan,
                stop_after_first_error=revision == 1,
            )
        except ValueError:
            if revision != 1 or scenario_id is None:
                raise
            scenario_steps = self._scenario_seed_steps(acquisition, tools, scenario_id)
            if not scenario_steps:
                raise
            logger.warning(
                "Using deterministic scenario-data seed plan after invalid LLM plan"
            )
            return validate_plan(
                ExecutionPlanRevision(
                    revision=revision,
                    reason=reason,
                    objective=acquisition.objective,
                    steps=scenario_steps,
                    required_output=acquisition.required_output,
                )
            )

    @staticmethod
    def _resolved_planning_query(user_query: str, acquisition: AcquisitionPlan) -> str:
        requirements = " ".join(
            " ".join(
                [requirement.description]
                + [need.domain for need in requirement.mapping_needs]
                + list(requirement.completion_criteria)
            )
            for requirement in acquisition.requirements
        )
        output = " ".join(
            [
                *acquisition.required_output.tables,
                *acquisition.required_output.layers,
                *acquisition.required_output.fields,
            ]
        )
        return " ".join(
            part
            for part in (user_query, acquisition.objective, requirements, output)
            if part
        )

    @staticmethod
    def _validate_acquisition_topic(
        plan: AcquisitionPlan, user_query: str, history: list[dict]
    ) -> AcquisitionPlan:
        user_evidence = " ".join(
            [
                *(
                    str(message.get("content") or "")
                    for message in history[-10:]
                    if message.get("role") == "user"
                ),
                user_query,
            ]
        ).lower()
        planned = plan.model_dump_json().lower()
        normative_markers = (
            "нормативн",
            "зон ограничен",
            "буферн",
            "restriction zone",
            "default buffer",
        )
        conversation_requests_norms = any(
            marker in user_evidence for marker in normative_markers
        )
        plan_switched_to_norms = any(marker in planned for marker in normative_markers)
        if plan_switched_to_norms and not conversation_requests_norms:
            raise ValueError(
                "acquisition plan switched to normative restrictions without support "
                "in the current dialogue"
            )
        return plan

    @staticmethod
    def _scenario_seed_steps(
        acquisition: AcquisitionPlan,
        tools: list[UrbanMcpTool],
        scenario_id: int,
    ) -> list[PlanStep]:
        """Repair an initial plan that mistakes a global mapping for scenario data."""

        available = {tool.name: tool for tool in tools}
        steps: list[PlanStep] = []
        used: set[str] = set()
        requires_entities = bool(acquisition.required_output.layers) or any(
            marker in acquisition.objective.lower()
            for marker in (
                "располож",
                "территор",
                "объект",
                "сервис",
                "геосло",
                "location",
                "within",
            )
        )
        requires_geometry = bool(acquisition.required_output.layers) or any(
            marker in acquisition.objective.lower()
            for marker in (
                "располож",
                "геометр",
                "геосло",
                "карт",
                "где",
                "location",
                "geometry",
                "map",
            )
        )
        for requirement in acquisition.requirements:
            domain = " ".join(
                [acquisition.objective, requirement.description]
                + [need.domain for need in requirement.mapping_needs]
            ).lower()
            if requires_entities and (
                "physical_object_type" in domain
                or {"physical", "object", "type"}.issubset(set(domain.split()))
            ):
                tool_name = (
                    "GetScenarioPhysicalObjectsWithGeometry"
                    if requires_geometry
                    and "GetScenarioPhysicalObjectsWithGeometry" in available
                    else "GetScenarioPhysicalObjects"
                )
            elif requires_entities and (
                "service_type" in domain
                or {"service", "type"}.issubset(set(domain.split()))
            ):
                tool_name = (
                    "GetScenarioServicesWithGeometry"
                    if requires_geometry
                    and "GetScenarioServicesWithGeometry" in available
                    else "GetScenarioServices"
                )
            elif "physical_object_type" in domain or (
                "physical" in domain and "object" in domain and "type" in domain
            ):
                tool_name = "GetScenarioPhysicalObjectTypes"
            elif "service_type" in domain or ("service" in domain and "type" in domain):
                tool_name = "GetScenarioServiceTypes"
            elif "physical_object" in domain or (
                "physical" in domain and "object" in domain
            ):
                tool_name = (
                    "GetScenarioPhysicalObjectsWithGeometry"
                    if requires_geometry
                    and "GetScenarioPhysicalObjectsWithGeometry" in available
                    else "GetScenarioPhysicalObjects"
                )
            elif "service" in domain or "сервис" in domain:
                tool_name = (
                    "GetScenarioServicesWithGeometry"
                    if requires_geometry
                    and "GetScenarioServicesWithGeometry" in available
                    else "GetScenarioServices"
                )
            else:
                continue
            tool = available.get(tool_name)
            if tool is None or tool_name in used:
                continue
            arguments = (
                {"scenario_id": scenario_id}
                if "scenario_id" in ((tool.input_schema or {}).get("properties") or {})
                else {}
            )
            steps.append(
                PlanStep(
                    step_id=f"scenario_data_{len(steps) + 1}",
                    purpose=requirement.description,
                    group=tool.group,
                    tool_name=tool.name,
                    arguments=arguments,
                    satisfies=[requirement.requirement_id],
                    expected_output="scenario records",
                )
            )
            used.add(tool_name)
        return steps

    @staticmethod
    def _workspace_prompt(enabled: bool) -> str:
        if not enabled:
            return "Workspace-шаги не создавай: workspace отключён."
        return f"""Для фильтрации, дедупликации, агрегации, join и получения ограниченной
выборки разрешены только workspace-инструменты из каталога ниже. Результат каждого Urban
шага автоматически становится artifact. В аргументе handle/mapping_handle/mask_handle
ссылайся на него строкой $artifact:<step_id>. Не передавай chat_id — его добавит система.
Workspace-шаг обязан depends_on на шаг, чей artifact он использует. Нельзя использовать
query/eval/apply/Python/SQL, URL или файловые пути.
Workspace-каталог: {json.dumps(WORKSPACE_TOOL_CATALOG, ensure_ascii=False)}"""

    async def _structured_plan_call(
        self,
        model: str,
        messages: list[dict],
        schema,
        label: str,
        post_validate: Callable[[Any], Any] | None = None,
        stop_after_first_error: bool = False,
    ):
        error = ""
        for attempt in range(MAX_PLANNER_RETRIES + 1):
            call: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "think": False,
                "options": {
                    "temperature": 0,
                    "num_predict": (
                        PLANNER_NUM_PREDICT
                        if attempt == 0
                        else PLANNER_NUM_PREDICT_RETRY
                    ),
                },
            }
            if attempt:
                call["reasoning_effort"] = PLANNER_RETRY_REASONING_EFFORT
                call["messages"] = messages + [
                    {
                        "role": "user",
                        "content": f"Исправь JSON: {error}. Верни только JSON.",
                    }
                ]
            if attempt < MAX_PLANNER_RETRIES:
                call["format"] = schema.model_json_schema()
            response = await self.llm_client.chat(**call)
            raw = (response.get("message") or {}).get("content") or ""
            if (
                schema is ExecutionPlanRevision
                and not raw.strip()
                and response.get("done_reason") == "length"
            ):
                error = "empty execution plan after reasoning exhausted max_tokens"
                logger.warning(
                    "Scenario-data execution planner exhausted max_tokens; "
                    "using deterministic fallback without redundant retries"
                )
                break
            try:
                payload = json.loads(strip_json_fence(raw))
                if schema is ExecutionPlanRevision:
                    payload = self._normalize_execution_plan_payload(payload)
                parsed = schema.model_validate(payload)
                return post_validate(parsed) if post_validate else parsed
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
                logger.warning(
                    f"Invalid scenario-data {label}, attempt {attempt + 1}: {error}"
                )
                if stop_after_first_error:
                    break
        raise ValueError(f"invalid scenario-data {label} after retries: {error}")

    @staticmethod
    def _normalize_execution_plan_payload(payload: Any) -> Any:
        """Repair harmless source-kind artefacts commonly emitted by gpt-oss.

        ``group`` selects one of the remote Urban MCP endpoints and has no meaning for
        local workspace operations.  Some otherwise valid structured replies copy the
        preceding Urban step's group into every step, or label a grouped remote Get*
        call as a workspace step.  The source fields make both repairs unambiguous and
        cannot broaden tool access; all tool names and arguments are still checked by
        :meth:`_canonicalize_plan`.
        """

        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        steps = normalized.get("steps")
        if not isinstance(steps, list):
            return normalized
        normalized_steps = []
        for step in steps:
            if not isinstance(step, dict):
                normalized_steps.append(step)
                continue
            repaired = dict(step)
            tool_name = repaired.get("tool_name")
            if isinstance(tool_name, str) and "." in tool_name:
                prefix, bare_name = tool_name.split(".", 1)
                if prefix in URBAN_MCP_GROUPS and bare_name:
                    repaired["group"] = prefix
                    repaired["tool_name"] = bare_name
                    tool_name = bare_name
            if repaired.get("kind") != "workspace":
                normalized_steps.append(repaired)
            elif tool_name in WORKSPACE_TOOL_CATALOG:
                normalized_steps.append({**repaired, "group": None})
            elif repaired.get("group"):
                # The model occasionally labels a remote Get* call as workspace even
                # though it preserved its exact Urban group.  Restore the only kind
                # compatible with that declared source; canonical catalogue validation
                # below still rejects unknown group/tool pairs and arguments.
                normalized_steps.append({**repaired, "kind": "urban_tool"})
            else:
                normalized_steps.append(repaired)
        normalized["steps"] = normalized_steps
        return normalized

    @staticmethod
    def _canonicalize_plan(
        plan: ExecutionPlanRevision,
        tools: list[UrbanMcpTool],
        *,
        workspace_enabled: bool = False,
        completed_step_ids: set[str] | None = None,
    ) -> ExecutionPlanRevision:
        available = {(tool.group, tool.name): tool for tool in tools}
        prior_completed = set(completed_step_ids or set())
        known = set(prior_completed)
        canonical_steps = []
        for step in plan.steps:
            if step.step_id in prior_completed:
                raise ValueError(f"replan reuses completed step_id: {step.step_id}")
            missing_dependencies = set(step.depends_on) - known
            if missing_dependencies:
                raise ValueError(
                    f"unknown dependencies for {step.step_id}: "
                    f"{sorted(missing_dependencies)}"
                )
            if step.kind == PlanStepKind.WORKSPACE:
                if not workspace_enabled:
                    raise ValueError("workspace step is disabled")
                spec = WORKSPACE_TOOL_CATALOG.get(step.tool_name)
                if spec is None:
                    raise ValueError(f"unknown workspace tool: {step.tool_name}")
                arguments = dict(step.arguments)
                if (
                    "handle" in spec.get("required", [])
                    and "handle" not in arguments
                    and len(step.depends_on) == 1
                ):
                    arguments["handle"] = f"$artifact:{step.depends_on[0]}"
                if (
                    step.tool_name == "WorkspaceSelect"
                    and "columns" not in arguments
                    and plan.required_output.fields
                ):
                    arguments["columns"] = list(plan.required_output.fields)
                if arguments != step.arguments:
                    step = step.model_copy(update={"arguments": arguments})
                allowed = set(spec.get("required", [])) | set(spec.get("optional", []))
                unknown = set(step.arguments) - allowed
                missing = set(spec.get("required", [])) - set(step.arguments)
                if unknown or missing:
                    raise ValueError(
                        f"invalid arguments for {step.tool_name}: "
                        f"unknown={sorted(unknown)}, missing={sorted(missing)}"
                    )
                references = ScenarioDataPlanBuilder._artifact_references(
                    step.arguments
                )
                undeclared = references - set(step.depends_on)
                if undeclared:
                    raise ValueError(
                        f"workspace step {step.step_id} must depend_on artifact sources: "
                        f"{sorted(undeclared)}"
                    )
                known.add(step.step_id)
                canonical_steps.append(step)
                continue
            if step.kind != PlanStepKind.URBAN_TOOL:
                continue
            tool = available.get((step.group, step.tool_name))
            if tool is None:
                matches = [
                    candidate for candidate in tools if candidate.name == step.tool_name
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"unknown plan tool: {step.group}.{step.tool_name}"
                    )
                tool = matches[0]
                step = step.model_copy(update={"group": tool.group})
            properties = tool.input_schema.get("properties") or {}
            unknown = set(step.arguments) - set(properties)
            if unknown:
                raise ValueError(
                    f"unknown arguments for {step.tool_name}: {sorted(unknown)}"
                )
            known.add(step.step_id)
            canonical_steps.append(step)
        return plan.model_copy(update={"steps": canonical_steps})

    @staticmethod
    def _artifact_references(value: Any) -> set[str]:
        if isinstance(value, str) and value.startswith("$artifact:"):
            return {value.removeprefix("$artifact:")}
        if isinstance(value, list):
            return set().union(
                *(ScenarioDataPlanBuilder._artifact_references(item) for item in value)
            )
        if isinstance(value, dict):
            return set().union(
                *(
                    ScenarioDataPlanBuilder._artifact_references(item)
                    for item in value.values()
                )
            )
        return set()

    async def choose_action(
        self,
        model: str,
        user_query: str,
        tools: list[UrbanMcpTool],
        observations: list[dict[str, Any]],
        history: list[dict] | None = None,
        scenario_id: int | None = None,
    ) -> ScenarioDataAction:
        shortlist = self._shortlist(tools, user_query, observations)
        prompt = self._build_prompt(shortlist, observations, scenario_id)
        messages = [
            {"role": "system", "content": prompt},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        error = ""
        for attempt in range(MAX_PLANNER_RETRIES + 1):
            if error:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Предыдущий JSON не прошёл проверку: "
                            f"{error}. Верни исправленный JSON строго по схеме. "
                            'Поле action должно быть ровно "call_tool" или '
                            '"final_answer"; не объединяй варианты через символ |.'
                        ),
                    }
                )
            # Escalate on each retry. Repeating an identical call is pointless when the
            # server answered with an empty string, and on a Harmony-served gpt-oss the
            # lever that actually matters is the reasoning effort, not the budget:
            # measured on the same prompt, reasoning_effort="low" returns *no content at
            # all* — the model finishes its analysis channel and stops without emitting a
            # final message — while "medium", "high" and omitting the field all answer.
            # Prompt size is not the factor: six tools (2k tokens) fail on "low" just as
            # forty-one (11.5k) do. So a retry raises the effort explicitly, which wins
            # over the configured default because _apply_think uses setdefault.
            budget = PLANNER_NUM_PREDICT if attempt == 0 else PLANNER_NUM_PREDICT_RETRY
            call: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "think": False,
                "options": {"temperature": 0, "num_predict": budget},
            }
            if attempt > 0:
                call["reasoning_effort"] = PLANNER_RETRY_REASONING_EFFORT
            if attempt < MAX_PLANNER_RETRIES:
                call["format"] = ScenarioDataAction.model_json_schema()
            else:
                # Last chance: no structured-output constraint at all, JSON asked for in
                # words. A model that returns nothing under the schema usually answers here.
                messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Ответь ТОЛЬКО JSON-объектом по описанной схеме, без "
                            "пояснений и без markdown-ограждения."
                        ),
                    }
                ]
                call["messages"] = messages
            response = await self.llm_client.chat(**call)
            raw = response["message"]["content"]
            if not (raw or "").strip():
                # Name the real cause. "Empty answer" on its own sent a reader looking for a
                # vague prompt, when the model had in fact reasoned to a conclusion and
                # simply never emitted it.
                trace = (response["message"].get("thinking") or "").strip()
                error = (
                    "модель не выдала финальный ответ"
                    + (
                        f" (сгенерирован только след рассуждений: {trace[:160]}…)"
                        if trace
                        else " (пустой ответ без следа рассуждений)"
                    )
                    + "; на gpt-oss через Harmony это даёт reasoning_effort=low — "
                    "поднимите OPENAI_THINK_EFFORT до medium"
                )
                logger.warning(
                    f"Empty scenario-data action, attempt {attempt + 1} "
                    f"(num_predict={budget}, format={'format' in call}, "
                    f"reasoning_effort={call.get('reasoning_effort', 'configured')}, "
                    f"reasoning_trace_len={len(trace)})"
                )
                continue
            try:
                payload = json.loads(strip_json_fence(raw))
                payload = self._repair_ambiguous_action(payload, shortlist)
                action = ScenarioDataAction.model_validate(payload)
                return self._canonicalize(action, shortlist)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
                logger.warning(
                    f"Invalid scenario-data action, attempt {attempt + 1}: {error}"
                )
        raise ValueError(f"invalid scenario-data action after retries: {error}")

    @staticmethod
    def _repair_ambiguous_action(payload: Any, tools: list[UrbanMcpTool]) -> Any:
        """Repair only the exact pseudo-enum emitted by the old planner prompt.

        The choice is unambiguous when the response names a real shortlisted tool or omits
        both tool coordinates. All other malformed values remain invalid and go through the
        normal retry path instead of being guessed.
        """

        if not isinstance(payload, dict) or payload.get("action") != (
            "call_tool | final_answer"
        ):
            return payload

        group = payload.get("group")
        tool_name = payload.get("tool_name")
        if any(tool.group == group and tool.name == tool_name for tool in tools):
            return {**payload, "action": ScenarioDataActionKind.CALL_TOOL.value}
        if group is None and tool_name is None:
            return {**payload, "action": ScenarioDataActionKind.FINAL_ANSWER.value}
        return payload

    @staticmethod
    def _canonicalize(
        action: ScenarioDataAction, tools: list[UrbanMcpTool]
    ) -> ScenarioDataAction:
        if action.action == ScenarioDataActionKind.FINAL_ANSWER:
            return action.model_copy(
                update={
                    "group": None,
                    "tool_name": None,
                    "arguments": {},
                    "layer_name": None,
                }
            )
        if action.group not in URBAN_MCP_GROUPS:
            raise ValueError(f"unknown Urban MCP group: {action.group}")
        selected = next(
            (
                tool
                for tool in tools
                if tool.group == action.group and tool.name == action.tool_name
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                f"tool {action.group}.{action.tool_name} is not in the supplied catalog"
            )
        properties = selected.input_schema.get("properties") or {}
        unknown = set(action.arguments) - set(properties)
        if unknown:
            raise ValueError(
                f"unknown arguments for {selected.name}: {sorted(unknown)}"
            )
        return action

    @classmethod
    def _shortlist(
        cls,
        tools: list[UrbanMcpTool],
        user_query: str,
        observations: list[dict[str, Any]],
    ) -> list[UrbanMcpTool]:
        context = (
            user_query
            + " "
            + " ".join(str(item.get("summary") or "") for item in observations[-3:])
        )
        query_tokens = cls._tokens(context)

        def score(tool: UrbanMcpTool) -> int:
            haystack = " ".join(
                (
                    tool.name,
                    tool.title,
                    tool.description[:600],
                    URBAN_MCP_GROUP_DESCRIPTIONS[tool.group],
                    " ".join(tool.tags),
                )
            ).lower()
            base = sum(
                3 if token in tool.title.lower() else 1
                for token in query_tokens
                if token in haystack
            )
            return base - _off_topic_penalty(tool, context.lower())

        ranked = sorted(tools, key=lambda item: (-score(item), item.name))
        chosen: dict[tuple[str, str], UrbanMcpTool] = {}
        # Best matches first, regardless of group. Reserving slots per group is what pushed
        # the catalogue to 41 entries and 10.3k prompt tokens — see SHORTLIST_SIZE.
        for tool in ranked[:SHORTLIST_SIZE]:
            chosen[(tool.group, tool.name)] = tool
        # One dictionary tool is kept even when it did not score: resolving an id to a name is
        # the second half of nearly every question, and the planner cannot call what it
        # cannot see.
        if not any(group == "dictionaries" for group, _ in chosen):
            for tool in ranked:
                if tool.group == "dictionaries":
                    chosen[(tool.group, tool.name)] = tool
                    break
        return list(chosen.values())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-zа-яё0-9_]+", text.lower())
            if len(token) >= 3
        }

    @staticmethod
    def _build_prompt(
        tools: list[UrbanMcpTool],
        observations: list[dict[str, Any]],
        scenario_id: int | None,
    ) -> str:
        catalog = [tool.compact_prompt_entry() for tool in tools]
        final_answer_example = {
            "action": "final_answer",
            "group": None,
            "tool_name": None,
            "arguments": {},
            "layer_name": None,
            "reason": "данных достаточно для ответа",
        }
        return f"""Ты — управляющий агент данных городского сценария. На каждом шаге выбери
ровно одно действие: вызвать read-only Urban MCP инструмент или завершить сбор данных.

Доступные инструменты этого шага:
{json.dumps(catalog, ensure_ascii=False)}

Результаты уже выполненных шагов (геометрия сокращена, полные слои уже отправлены клиенту):
{json.dumps(observations, ensure_ascii=False)}

Верни только JSON-объект со следующими полями:
- action: ровно одна из двух строк — "call_tool" или "final_answer". Никогда не записывай
  сразу оба варианта и не используй символ | в значении.
- group: точное имя MCP-группы из каталога для call_tool, иначе null.
- tool_name: точное имя инструмента из каталога для call_tool, иначе null.
- arguments: JSON-объект аргументов инструмента, иначе пустой объект.
- layer_name: понятное название ожидаемого географического слоя или null.
- reason: краткая причина выбора.

Пример корректного завершения сбора данных:
{json.dumps(final_answer_example, ensure_ascii=False)}

Правила:
- Используй только точные group и tool_name из каталога.
- Не повторяй уже выполненный вызов с теми же аргументами.
- Контекст сценария: {scenario_id if scenario_id is not None else "не выбран"}.
- Если scenario_id выбран, он подставляется системой: не угадывай и не меняй его.
- Если scenario_id не выбран и вопрос требует данных конкретного сценария, заверши сбор
  данных через final_answer: в итоговом ответе нужно попросить пользователя выбрать сценарий.
- Сначала используй справочники, если для основного запроса нужно узнать ID по названию.
- ОБРАТНОЕ НАПРАВЛЕНИЕ ВАЖНЕЕ: если в наблюдении есть unresolved_references, названия по
  этим идентификаторам ещё не получены. Вызови справочник, который вернёт их названия
  (например, типы сервисов или типы объектов), и не выбирай final_answer, пока это не
  сделано, — иначе ответ будет про номера, а не про то, что человек спросил.
- Один вызов почти никогда не отвечает на вопрос целиком. Типичная последовательность:
  получить записи -> получить справочник типов -> сопоставить -> завершить.
- Для запроса слоя выбирай инструмент, возвращающий GeoJSON/геометрию.
- Выбирай инструмент, ПРЕДМЕТ которого совпадает с вопросом. Инструменты про зоны
  ограничений, буферы, показатели и социальные группы доступны и уместны, только если
  спросили именно о них; на вопрос «какие объекты/сервисы есть и сколько» бери инструменты
  по объектам и сервисам, а не по зонам ограничений.
- Выбирай final_answer, только если по наблюдениям можно назвать конкретные сущности и
  числа, а не только их идентификаторы и общее количество.
- Не вызывай инструменты для создания, изменения или удаления данных.
- layer_name заполняй только если ожидается географический слой."""
