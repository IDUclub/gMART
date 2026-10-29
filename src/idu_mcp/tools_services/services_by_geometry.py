"""Lossless union queries to Urban API, with bounded component/quadrant splits."""

import asyncio
import hashlib
import math
import os

import geopandas as gpd
import pandas as pd
from fastmcp.exceptions import ToolError
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from src.idu_mcp.common.api_handlers.json_api_handler import GeometryPayloadTooLarge
from src.idu_mcp.tools_services.geometry_tools import GeometryTools
from src.idu_mcp.tools_services.spatial_predicates import exact_spatial_predicate


def positive_env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def search_area(zones_layers: list[str], layers: dict, buffer_m: float):
    if not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("buffer_m должен быть конечным неотрицательным числом")
    if not zones_layers:
        raise ValueError("zones_layers не должен быть пустым")
    geometries = []
    for name in dict.fromkeys(zones_layers):
        if name not in layers:
            raise ValueError(f"Слой {name!r} отсутствует")
        for feature in layers[name].get("features", []):
            if not feature.get("geometry"):
                raise ValueError(f"В слое {name!r} отсутствует геометрия")
            geometry = shape(feature["geometry"])
            if not geometry.is_valid:
                raise ValueError(f"В слое {name!r} некорректная геометрия")
            geometries.append(geometry)
    area = unary_union(geometries)
    if buffer_m and not area.is_empty:
        series = gpd.GeoSeries([area], crs=4326)
        crs = series.estimate_utm_crs()
        if crs is None:
            raise ValueError("Не удалось определить метрическую CRS")
        area = series.to_crs(crs).buffer(buffer_m).to_crs(4326).iloc[0]
    if not area.is_empty and area.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Область поиска должна быть Polygon или MultiPolygon")
    return area


def _polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type not in {"MultiPolygon", "GeometryCollection"}:
        return []
    return [part for child in geometry.geoms for part in _polygon_parts(child)]


class ServicesByGeometry:
    def __init__(self, client):
        self.client = client
        self.concurrency = positive_env_int("COMPLIANCE_BY_GEOMETRY_CONCURRENCY", 8)
        self.max_split_depth = positive_env_int(
            "COMPLIANCE_BY_GEOMETRY_MAX_SPLIT_DEPTH", 6
        )
        # Shared by calls on this tool instance, including concurrent MCP requests.
        self.semaphore = asyncio.Semaphore(self.concurrency)

    async def query(
        self,
        *,
        scenario_id: int,
        services_names: list[str],
        zones_layers: list[str],
        layer_name: str,
        layers: dict,
        token: str,
        buffer_m: float = 0,
        intersection_type: str = "intersects",
        physical_object_type_ids=None,
    ) -> dict:
        if intersection_type not in {"intersects", "within"}:
            raise ValueError("intersection_type: ожидается intersects или within")
        if not services_names or not layer_name.strip():
            raise ValueError("Нужны services_names и непустое layer_name")
        # TODO: Urban API by_geometry has no physical_object_type_ids filter yet.
        area = await asyncio.to_thread(search_area, zones_layers, layers, buffer_m)
        names = list(
            dict.fromkeys(name.strip().capitalize() for name in services_names)
        )
        resolved = await self.client.get_service_name_id(names, token)
        missing = [name for name in names if name not in resolved]
        if missing:
            raise ToolError(f"Типы сервисов не найдены: {', '.join(missing)}")
        types = list(dict.fromkeys(resolved.values()))
        components = _polygon_parts(area)
        stats = {
            "calls_made": 0,
            "mode": "single",
            "component_count": len(components),
            "part_count": 0,
            "intersection_type": intersection_type,
        }

        def count_attempt():
            stats["calls_made"] += 1

        async def request(type_id, geometry, *, predicate, depth=0):
            try:
                async with self.semaphore:
                    response = await self.client.get_services_by_geometry(
                        scenario_id,
                        type_id,
                        mapping(geometry),
                        token,
                        intersection_type=predicate,
                        on_attempt=count_attempt,
                    )
                if (
                    not isinstance(response, dict)
                    or response.get("type") != "FeatureCollection"
                    or not isinstance(response.get("features"), list)
                ):
                    raise ToolError(
                        "Urban API вернул некорректный FeatureCollection сервисов"
                    )
                stats["part_count"] += 1
                return response["features"]
            except GeometryPayloadTooLarge as exc:
                stats["mode"] = "split"
                parts = _polygon_parts(geometry)
                if len(parts) == 1:
                    if depth >= self.max_split_depth:
                        raise ToolError(
                            "Urban API отклоняет размер части геометрии после "
                            f"{self.max_split_depth} уровней деления; "
                            "геометрия не упрощалась."
                        ) from exc
                    x0, y0, x1, y1 = geometry.bounds
                    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
                    parts = [
                        part
                        for bounds in (
                            (x0, y0, xm, ym),
                            (xm, y0, x1, ym),
                            (x0, ym, xm, y1),
                            (xm, ym, x1, y1),
                        )
                        for part in _polygon_parts(geometry.intersection(box(*bounds)))
                    ]
                    depth += 1
                # Sequential parts avoid an exponentially large task tree; types
                # still run concurrently under the shared semaphore.
                features = []
                for part in parts:
                    features.extend(
                        await request(
                            type_id, part, predicate="intersects", depth=depth
                        )
                    )
                return features

        async def one_type(type_id):
            if area.is_empty:
                return []
            return await request(type_id, area, predicate=intersection_type)

        tasks = [asyncio.create_task(one_type(type_id)) for type_id in types]
        try:
            responses = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        unique = {}
        per_type_counts = {}
        for type_id, features in zip(types, responses):
            accepted = set()
            for feature in features:
                properties = dict(feature.get("properties") or {})
                service_id = properties.get("service_id")
                if service_id is None or not feature.get("geometry"):
                    raise ToolError(
                        "Urban API вернул сервис без service_id или геометрии"
                    )
                geometry = shape(feature["geometry"])
                if geometry.geom_type == "GeometryCollection":
                    raise ToolError(
                        "Urban API вернул неподдерживаемую GeometryCollection"
                    )
                if not exact_spatial_predicate(geometry, area, intersection_type):
                    continue
                properties.setdefault("service_type_id", type_id)
                properties.setdefault(
                    "service_type_name",
                    next(name for name, value in resolved.items() if value == type_id),
                )
                properties.setdefault(
                    "service_type",
                    {
                        "id": properties["service_type_id"],
                        "name": properties["service_type_name"],
                    },
                )
                properties["object_ref"] = GeometryTools._object_ref(
                    pd.Series(properties), layer_name, len(unique)
                )
                unique.setdefault(
                    str(service_id),
                    {
                        **feature,
                        "properties": properties,
                    },
                )
                accepted.add(str(service_id))
            per_type_counts[str(type_id)] = len(accepted)
        revision = hashlib.sha256(area.wkb).hexdigest()
        return {
            layer_name: {
                "type": "FeatureCollection",
                "features": list(unique.values()),
                "meta": {
                    **stats,
                    "per_type_counts": per_type_counts,
                    "complete": True,
                    "truncated": False,
                    "revision": f"scenario:{scenario_id}:services_by_geometry:{revision}:"
                    f"{','.join(map(str, types))}:{intersection_type}",
                },
            }
        }
