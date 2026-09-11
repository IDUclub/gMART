"""Authenticated acquisition and publication of scenario analytics."""

from __future__ import annotations

import re

from loguru import logger

from src.agents.api_clients.chat_storage_client.request_models import (
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.services.scenario_data.scenario_data_indicators import (
    IndicatorRequest,
    base_comparison_requested,
    calculation_request,
    comparison_entities,
    complete_records,
    explanation_requested,
    indicator_comparison,
    indicator_query,
    literal_indicator_request,
    lower_first,
    names_indicator,
    normalize_indicators,
    render_indicators,
    scenario_labels,
    scenario_scope,
    selection_messages,
    signed,
    validate_request,
)
from src.agents.services.scenario_data.scenario_data_type_mapper import UrbanTypeMapper


def _scenario_name(info: dict) -> str | None:
    name = info.get("name")
    return name.strip() if isinstance(name, str) and name.strip() else None


class ScenarioAnalytics:
    def __init__(self, service):
        self.service = service

    async def run(
        self,
        *,
        request_id,
        client,
        token_ref,
        model,
        query,
        selected,
        tools,
        parts,
        chat_id,
        persist_history,
        indicators_route=False,
    ):
        host = self.service
        named = {(t.group, t.name): t for t in tools}
        metadata, data = {}, {}

        async def execute(group, name, sid, box, project_id=None):
            tool = named.get((group, name))
            if tool is None:
                raise ValueError(f"Инструмент {name} недоступен.")
            # Each target comes from the explicit user scope, not model arguments.
            arguments = host._prepare_arguments(tool, {}, sid, project_id)
            source = f"URBAN_MCP/{group}"
            yield await host._buf(
                request_id,
                host._tool_call_event(
                    {"group": group, "tool_name": name, "arguments": arguments}, source
                ),
            )
            parts.append(
                ToolCallPartRequest(
                    kind="tool_call",
                    mcp_source=source,
                    payload=ToolCallPayload(
                        execution_mode="sequential",
                        calls=[
                            ToolCall(
                                step=len(parts) + 1, tool_name=name, arguments=arguments
                            )
                        ],
                    ),
                )
            )
            async for event in host._retryable_operation(
                request_id,
                client,
                token_ref,
                lambda: client.execute_tool(
                    group,
                    name,
                    arguments,
                    meta=(
                        {"scenario_id": sid}
                        if sid is not None
                        else {"project_id": project_id}
                    ),
                ),
                box,
                retry_transient=True,
            ):
                yield await host._buf(request_id, event)

        async def read_metadata(sid):
            box = []
            async for event in execute("projects", "GetScenarioById", sid, box):
                yield event
            info = host._unwrap_result(box[0])
            if not isinstance(info, dict) or info.get("scenario_id") != sid:
                raise ValueError(
                    "Не удалось подтвердить доступ и принадлежность сценария."
                )
            metadata[sid] = info

        async def read_base_scenario(target, box):
            """Resolve the project base scenario, or say why the comparison is absent."""
            info = metadata[target]
            own_base = "Этот сценарий — базовый сценарий проекта, сравнивать не с чем."
            if info.get("is_based"):
                box.append((target, own_base))
                return
            project_id = (info.get("project") or {}).get("project_id")
            if project_id is None:
                box.append(
                    (
                        None,
                        "Проект сценария не определён, поэтому показана только сводка без сравнения.",
                    )
                )
                return
            project = []
            try:
                async for event in execute(
                    "projects", "GetProjectById", None, project, project_id=project_id
                ):
                    yield event
                base = (host._unwrap_result(project[0]) or {}).get(
                    "base_scenario"
                ) or {}
            except Exception as exc:
                # The base scenario is optional context. Losing it degrades the answer to a
                # single-scenario summary instead of failing an otherwise complete request.
                logger.warning(
                    "Base scenario lookup failed: {}: {}", type(exc).__name__, exc
                )
                box.append(
                    (
                        None,
                        "Базовый сценарий недоступен, поэтому показана только сводка без сравнения.",
                    )
                )
                return
            base_id = base.get("id")
            if not isinstance(base_id, int) or isinstance(base_id, bool):
                box.append(
                    (
                        None,
                        "У проекта не задан базовый сценарий, поэтому показана только сводка без сравнения.",
                    )
                )
                return
            if base_id == target:
                box.append((target, own_base))
                return
            box.append((base_id, None))

        rows = []
        base_note = None
        base_id = None
        column_labels = None
        try:
            ids = scenario_scope(query, selected)
            entity = comparison_entities(query)
            if not indicators_route and not indicator_query(query) and entity is None:
                raise ValueError(
                    "Для сравнения укажите показатели или общее количество сервисов/физических объектов."
                )
            if entity and re.search(
                r"типа|[«\"]|этаж|радиус|вместим|по адресу", query, re.I
            ):
                raise ValueError(
                    "Уточните сравнение: здесь поддерживается общее количество сущностей без дополнительных фильтров."
                )
            # Read metadata with the caller's token for every target. A denied target
            # aborts acquisition; partial results are never rendered as a comparison.
            if base_comparison_requested(query, ids, default=indicators_route):
                target = ids[0]
                async for event in read_metadata(target):
                    yield event
                resolved = []
                async for event in read_base_scenario(target, resolved):
                    yield event
                base_id, base_note = resolved[0]
                if base_id not in (None, target):
                    ids = [base_id, *ids]
            for sid in ids:
                if sid not in metadata:
                    async for event in read_metadata(sid):
                        yield event
                box = []
                group, name = (
                    (
                        "projects",
                        (
                            "GetScenarioServices"
                            if entity == "service"
                            else "GetScenarioPhysicalObjects"
                        ),
                    )
                    if entity
                    else ("indicators", "GetScenarioIndicatorsValues")
                )
                async for event in execute(group, name, sid, box):
                    yield event
                data[sid] = host._unwrap_result(box[0])
            names = {sid: _scenario_name(metadata[sid]) for sid in ids}
            labels = scenario_labels(names, selected=selected, base_id=base_id)
            if entity:
                counts = {}
                for sid, result in data.items():
                    records = complete_records(result)
                    identifiers = [r.get(entity + "_id") for r in records]
                    if any(
                        isinstance(v, bool) or not isinstance(v, int)
                        for v in identifiers
                    ):
                        raise ValueError(
                            "Полнота сущностей не подтверждена: отсутствуют ID."
                        )
                    counts[sid] = len(set(identifiers))
                    rows.append(
                        {
                            "scenario": labels[sid],
                            "scenario_id": sid,
                            "count": counts[sid],
                        }
                    )
                noun = "сервисов" if entity == "service" else "физических объектов"
                lines = [
                    f"{labels[sid]}: всего {noun} — {count}."
                    for sid, count in counts.items()
                ]
                first = ids[0]
                lines += [
                    f"Разница: {lower_first(labels[sid])} − {lower_first(labels[first])} = {signed(counts[sid] - counts[first])}."
                    for sid in ids[1:]
                ]
                answer = "\n\n".join(lines)
            else:
                scenarios = {
                    sid: normalize_indicators(value, sid) for sid, value in data.items()
                }
                facts = [fact for values in scenarios.values() for fact in values]
                request = (
                    calculation_request(query)
                    or (
                        IndicatorRequest(operation="all", names=[], missing=[])
                        if indicators_route and not names_indicator(query)
                        else None
                    )
                    or literal_indicator_request(query, facts)
                    or await UrbanTypeMapper(host.llm_client)._request_json(
                        model,
                        selection_messages(query, facts),
                        IndicatorRequest,
                        "indicator selection",
                        post_validate=lambda value: validate_request(
                            value, facts, query
                        ),
                    )
                )
                if (
                    indicators_route
                    and request.operation in {"values", "all"}
                    and not explanation_requested(query)
                ):
                    answer, rows, column_labels = indicator_comparison(
                        request,
                        scenarios,
                        query=query,
                        names=names,
                        selected=selected,
                        base_id=base_id,
                    )
                else:
                    answer, rows = render_indicators(
                        request, scenarios, query=query, labels=labels
                    )
            projects = {
                (m.get("project") or {}).get("project_id") for m in metadata.values()
            }
            if len(projects - {None}) > 1:
                answer += "\n\nСценарии относятся к разным проектам; сравниваются их сохранённые значения."
            if base_note:
                answer = base_note + "\n\n" + answer
        except ValueError as exc:
            logger.warning(
                "Scenario analytics validation failed: {}", type(exc).__name__
            )
            rows = []
            # Validation failures are explicit; no partially acquired facts are published.
            answer = "Не удалось подтвердить данные для ответа. Уточните сценарии, названия показателей и условия запроса."
        if rows:
            table = host._table_from_result(
                rows,
                name="scenario_analytics",
                title="Показатели и сравнение сценариев",
                labels=column_labels,
            )
            if table:
                yield await host._buf(request_id, {"type": "table", "content": table})
                parts.append(host._table_part(table))
        for event in host._answer_events(answer):
            yield await host._buf(request_id, event)
        parts.append(TextPartRequest(kind="text", payload=TextPayload(text=answer)))
        await host._complete_pipeline(
            request_id,
            token_ref[0],
            chat_id,
            parts,
            scenario_id=selected,
            persist_history=persist_history,
            context_model=model if host.linear_workflow_enabled else None,
        )
