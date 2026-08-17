"""Safe fixed-operation DataFrame/GeoDataFrame MCP workspace."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import geopandas as gpd
import pandas as pd
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from src.idu_mcp.common.auth.token_verifier import AnyTokenVerifier
from src.idu_mcp.dependencies.auth_dependencies import extract_workspace_owner
from src.idu_mcp.dependencies.dependencies import get_workspace_store
from src.idu_mcp.tools_services.workspace_store import (
    WorkspaceStore,
    frame_from_payload,
)

workspace_mcp = FastMCP("SCENARIO DATA WORKSPACE", auth=AnyTokenVerifier())
tags = {"workspace", "scenario_data"}
annotations = {"readOnlyHint": False, "destructiveHint": False, "openWorldHint": False}


class FilterOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    CONTAINS = "contains"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class FilterCondition(BaseModel):
    column: str = Field(min_length=1)
    operator: FilterOperator
    value: Any = None


class AggregateFunction(StrEnum):
    COUNT = "count"
    NUNIQUE = "nunique"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"


class AggregateSpec(BaseModel):
    column: str = Field(min_length=1)
    function: AggregateFunction
    output_column: str = Field(min_length=1)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ToolError(f"Колонки не найдены: {sorted(missing)}")


@workspace_mcp.tool(
    name="WorkspaceCreate",
    title="Создать временный набор данных",
    description="Создать из записей или GeoJSON неизменяемый workspace-набор для scenario-data.",
    tags=tags,
    annotations=annotations,
)
async def workspace_create(
    chat_id: str,
    records: list[dict[str, Any]] | None = None,
    feature_collection: dict[str, Any] | None = None,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame = frame_from_payload(records, feature_collection)
    return await store.create(
        frame,
        owner_id=owner_id,
        chat_id=chat_id,
        lineage={"operation": "create"},
    )


@workspace_mcp.tool(
    name="WorkspaceDescribe",
    title="Описать набор",
    description="Вернуть схему, размеры и ограниченный профиль workspace-набора.",
    tags=tags,
    annotations=annotations,
)
async def workspace_describe(
    handle: str,
    chat_id: str,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    return await store.describe(handle, owner_id=owner_id, chat_id=chat_id)


@workspace_mcp.tool(
    name="WorkspaceUniqueValues",
    title="Получить уникальные значения",
    description="Вернуть до 50 уникальных значений колонки и их точные количества.",
    tags=tags,
    annotations=annotations,
)
async def workspace_unique_values(
    handle: str,
    chat_id: str,
    column: str,
    limit: int = 50,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    if not 1 <= limit <= 50:
        raise ToolError("limit должен быть от 1 до 50")
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, [column])
    series = frame[column]
    counts = series.value_counts(dropna=False)
    values = [
        {"value": None if pd.isna(value) else value, "count": int(count)}
        for value, count in counts.head(limit).items()
    ]
    return {
        "handle": handle,
        "column": column,
        "cardinality": int(series.nunique(dropna=False)),
        "values": values,
        "truncated": len(counts) > limit,
        "null_count": int(series.isna().sum()),
    }


@workspace_mcp.tool(
    name="WorkspaceSample",
    title="Получить выборку строк",
    description="Вернуть ограниченную выборку первых строк без геометрии.",
    tags=tags,
    annotations=annotations,
)
async def workspace_sample(
    handle: str,
    chat_id: str,
    limit: int = 20,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    if not 1 <= limit <= 20:
        raise ToolError("limit должен быть от 1 до 20")
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    sample = frame.drop(columns=["geometry"], errors="ignore").head(limit)
    return {
        "handle": handle,
        "rows": sample.where(pd.notna(sample), None).to_dict(orient="records"),
        "total_rows": int(len(frame)),
        "truncated": len(frame) > limit,
    }


@workspace_mcp.tool(
    name="WorkspaceFilter",
    title="Отфильтровать строки",
    description="Создать новый набор по списку безопасных структурированных условий.",
    tags=tags,
    annotations=annotations,
)
async def workspace_filter(
    handle: str,
    chat_id: str,
    conditions: list[FilterCondition],
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, [item.column for item in conditions])
    mask = pd.Series(True, index=frame.index)
    for item in conditions:
        series = frame[item.column]
        if item.operator == FilterOperator.EQ:
            current = series == item.value
        elif item.operator == FilterOperator.NE:
            current = series != item.value
        elif item.operator == FilterOperator.GT:
            current = series > item.value
        elif item.operator == FilterOperator.GTE:
            current = series >= item.value
        elif item.operator == FilterOperator.LT:
            current = series < item.value
        elif item.operator == FilterOperator.LTE:
            current = series <= item.value
        elif item.operator == FilterOperator.IN:
            if not isinstance(item.value, list):
                raise ToolError("Оператор in требует список value")
            current = series.isin(item.value)
        elif item.operator == FilterOperator.CONTAINS:
            current = series.astype("string").str.contains(
                str(item.value), regex=False, na=False
            )
        elif item.operator == FilterOperator.IS_NULL:
            current = series.isna()
        else:
            current = series.notna()
        mask &= current.fillna(False)
    result = frame.loc[mask].copy()
    return await store.derive(
        result,
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="filter",
        arguments={"conditions": [item.model_dump(mode="json") for item in conditions]},
    )


@workspace_mcp.tool(
    name="WorkspaceSelect",
    title="Выбрать колонки",
    description="Создать новый набор только с перечисленными колонками.",
    tags=tags,
    annotations=annotations,
)
async def workspace_select(
    handle: str,
    chat_id: str,
    columns: list[str],
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, columns)
    return await store.derive(
        frame.loc[:, columns].copy(),
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="select",
        arguments={"columns": columns},
    )


@workspace_mcp.tool(
    name="WorkspaceSort",
    title="Отсортировать строки",
    description="Создать новый набор со стабильной сортировкой по колонкам.",
    tags=tags,
    annotations=annotations,
)
async def workspace_sort(
    handle: str,
    chat_id: str,
    columns: list[str],
    ascending: bool = True,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, columns)
    result = frame.sort_values(columns, ascending=ascending, kind="stable")
    return await store.derive(
        result,
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="sort",
        arguments={"columns": columns, "ascending": ascending},
    )


@workspace_mcp.tool(
    name="WorkspaceDeduplicate",
    title="Удалить дубликаты",
    description="Создать новый набор с уникальными строками по выбранным колонкам.",
    tags=tags,
    annotations=annotations,
)
async def workspace_deduplicate(
    handle: str,
    chat_id: str,
    columns: list[str],
    keep: Literal["first", "last"] = "first",
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, columns)
    result = frame.drop_duplicates(subset=columns, keep=keep)
    return await store.derive(
        result,
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="deduplicate",
        arguments={"columns": columns, "keep": keep},
    )


@workspace_mcp.tool(
    name="WorkspaceAggregate",
    title="Сгруппировать и агрегировать",
    description="Создать агрегат с функциями count/nunique/sum/min/max/mean.",
    tags=tags,
    annotations=annotations,
)
async def workspace_aggregate(
    handle: str,
    chat_id: str,
    group_by: list[str],
    aggregations: list[AggregateSpec],
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, group_by + [item.column for item in aggregations])
    named = {
        item.output_column: pd.NamedAgg(column=item.column, aggfunc=item.function.value)
        for item in aggregations
    }
    result = frame.groupby(group_by, dropna=False).agg(**named).reset_index()
    return await store.derive(
        result,
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="aggregate",
        arguments={
            "group_by": group_by,
            "aggregations": [item.model_dump(mode="json") for item in aggregations],
        },
    )


@workspace_mcp.tool(
    name="WorkspaceJoinMapping",
    title="Присоединить справочник",
    description="Безопасно присоединить справочник many-to-one по двум колонкам.",
    tags=tags,
    annotations=annotations,
)
async def workspace_join_mapping(
    handle: str,
    mapping_handle: str,
    chat_id: str,
    left_on: str,
    right_on: str,
    how: Literal["left", "inner"] = "left",
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    mapping, _ = await store.load(mapping_handle, owner_id=owner_id, chat_id=chat_id)
    _require_columns(frame, [left_on])
    _require_columns(mapping, [right_on])
    result = frame.merge(
        mapping.drop(columns=["geometry"], errors="ignore"),
        how=how,
        left_on=left_on,
        right_on=right_on,
        validate="many_to_one",
        suffixes=("", "_mapping"),
    )
    return await store.derive(
        result,
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="join_mapping",
        arguments={
            "mapping_handle": mapping_handle,
            "left_on": left_on,
            "right_on": right_on,
            "how": how,
        },
    )


@workspace_mcp.tool(
    name="WorkspaceSpatialFilter",
    title="Пространственно отфильтровать",
    description="Создать геонабор по предикату intersects, within или contains.",
    tags=tags,
    annotations=annotations,
)
async def workspace_spatial_filter(
    handle: str,
    mask_handle: str,
    chat_id: str,
    predicate: Literal["intersects", "within", "contains"] = "intersects",
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    mask, _ = await store.load(mask_handle, owner_id=owner_id, chat_id=chat_id)
    if not isinstance(frame, gpd.GeoDataFrame) or not isinstance(
        mask, gpd.GeoDataFrame
    ):
        raise ToolError("SpatialFilter требует два геопространственных набора")
    if frame.crs != mask.crs:
        mask = mask.to_crs(frame.crs)
    geometry = mask.geometry.union_all()
    result = frame.loc[getattr(frame.geometry, predicate)(geometry)].copy()
    return await store.derive(
        result,
        parent_handle=handle,
        owner_id=owner_id,
        chat_id=chat_id,
        operation="spatial_filter",
        arguments={"mask_handle": mask_handle, "predicate": predicate},
    )


@workspace_mcp.tool(
    name="WorkspaceToFeatureCollection",
    title="Получить GeoJSON-слой",
    description="Вернуть ограниченный GeoJSON FeatureCollection в EPSG:4326.",
    tags=tags,
    annotations=annotations,
)
async def workspace_to_feature_collection(
    handle: str,
    chat_id: str,
    limit: int = 10000,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    if not 1 <= limit <= 10000:
        raise ToolError("limit должен быть от 1 до 10000")
    frame, _ = await store.load(handle, owner_id=owner_id, chat_id=chat_id)
    if not isinstance(frame, gpd.GeoDataFrame):
        raise ToolError("Набор не содержит геометрию")
    output = frame.to_crs("EPSG:4326") if frame.crs else frame.set_crs("EPSG:4326")
    collection = output.head(limit).__geo_interface__
    return {
        "handle": handle,
        "feature_collection": collection,
        "total_features": int(len(frame)),
        "truncated": len(frame) > limit,
    }


@workspace_mcp.tool(
    name="WorkspaceRelease",
    title="Освободить набор",
    description="Удалить принадлежащий пользователю и чату временный набор.",
    tags=tags,
    annotations=annotations,
)
async def workspace_release(
    handle: str,
    chat_id: str,
    owner_id: str = Depends(extract_workspace_owner),
    store: WorkspaceStore = Depends(get_workspace_store),
) -> dict[str, Any]:
    return {
        "handle": handle,
        "released": await store.release(handle, owner_id=owner_id, chat_id=chat_id),
    }
