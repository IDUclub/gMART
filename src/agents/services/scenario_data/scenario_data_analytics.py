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
    calculation_request,
    comparison_entities,
    complete_records,
    indicator_query,
    literal_indicator_request,
    normalize_indicators,
    render_indicators,
    scenario_scope,
    selection_messages,
    validate_request,
)
from src.agents.services.scenario_data.scenario_data_type_mapper import UrbanTypeMapper


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
    ):
        host = self.service
        named = {(t.group, t.name): t for t in tools}

        async def execute(group, name, sid, box):
            tool = named.get((group, name))
            if tool is None:
                raise ValueError(f"Инструмент {name} недоступен.")
            # Each target comes from the explicit user scope, not model arguments.
            arguments = host._prepare_arguments(tool, {}, sid)
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
                    group, name, arguments, meta={"scenario_id": sid}
                ),
                box,
                retry_transient=True,
            ):
                yield await host._buf(request_id, event)

        rows = []
        try:
            ids = scenario_scope(query, selected)
            entity = comparison_entities(query)
            if not indicator_query(query) and entity is None:
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
            metadata, data = {}, {}
            for sid in ids:
                box = []
                async for event in execute("projects", "GetScenarioById", sid, box):
                    yield event
                info = host._unwrap_result(box[0])
                if not isinstance(info, dict) or info.get("scenario_id") != sid:
                    raise ValueError(
                        "Не удалось подтвердить доступ и принадлежность сценария."
                    )
                metadata[sid] = info
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
                    rows.append({"scenario_id": sid, "count": counts[sid]})
                noun = "сервисов" if entity == "service" else "физических объектов"
                lines = [
                    f"Сценарий {sid}: всего {noun} — {count}."
                    for sid, count in counts.items()
                ]
                first = ids[0]
                lines += [
                    f"Разница {sid} − {first}: {counts[sid] - counts[first]}."
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
                answer, rows = render_indicators(request, scenarios, query=query)
            projects = {
                (m.get("project") or {}).get("project_id") for m in metadata.values()
            }
            if len(projects - {None}) > 1:
                answer += "\n\nСценарии относятся к разным проектам; сравниваются их сохранённые значения."
        except ValueError as exc:
            logger.warning(
                "Scenario analytics validation failed: {}", type(exc).__name__
            )
            rows = []
            # Validation failures are explicit; no partially acquired facts are published.
            answer = "Не удалось подтвердить данные для ответа. Уточните ID сценариев, названия показателей и условия запроса."
        if rows:
            table = host._table_from_result(
                rows,
                name="scenario_analytics",
                title="Показатели и сравнение сценариев",
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
