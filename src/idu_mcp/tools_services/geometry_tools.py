from __future__ import annotations

import asyncio
import json
from typing import Any

import geopandas as gpd
import pandas as pd

from src.idu_mcp.tools_services.entites.buffer_type_enum import BufferTypeEnum


class GeometryTools:
    """Geometry operations used by the restriction MCP tools."""

    @staticmethod
    def _plain(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if pd.isna(value):
            return None
        return str(value)

    @classmethod
    def _object_ref(
        cls, row: pd.Series, layer_name: str, row_number: int
    ) -> dict[str, Any]:
        if cls._plain(row.get("service_id")) is not None:
            namespace = "service"
            entity_id = cls._plain(row.get("service_id"))
        elif cls._plain(row.get("physical_object_id")) is not None:
            namespace = "physical_object"
            entity_id = cls._plain(row.get("physical_object_id"))
        else:
            namespace = "feature"
            entity_id = cls._plain(row.get("_feature_id")) or str(row_number)

        geometry_id = cls._plain(row.get("object_geometry_id"))
        uid = f"{namespace}/{entity_id}"
        if geometry_id is not None:
            uid += f"/geometry/{geometry_id}"

        raw_name = cls._plain(row.get("name"))
        raw_address = cls._plain(row.get("address"))
        display_name = raw_name or raw_address or f"{layer_name} #{entity_id}"
        return {
            "id": uid,
            "namespace": namespace,
            "entity_id": entity_id,
            "geometry_id": geometry_id,
            "layer": layer_name,
            "name": display_name,
        }

    @classmethod
    def _feature_collection_to_gdf(
        cls, layer_name: str, feature_collection: dict
    ) -> gpd.GeoDataFrame:
        layer = gpd.GeoDataFrame.from_features(feature_collection, crs=4326)
        feature_ids = [
            feature.get("id") for feature in feature_collection.get("features", [])
        ]
        if len(feature_ids) == len(layer):
            layer["_feature_id"] = feature_ids
        elif "_feature_id" not in layer:
            layer["_feature_id"] = list(range(len(layer)))
        layer["source_layer"] = layer_name
        layer["_row_id"] = [
            f"{layer_name}:{position}" for position in range(len(layer))
        ]
        layer["object_ref"] = [
            cls._object_ref(row, layer_name, position)
            for position, (_, row) in enumerate(layer.iterrows())
        ]
        return layer

    @classmethod
    def create_buffer(
        cls,
        layer: gpd.GeoDataFrame,
        buffer_size: float,
        buffer_type: BufferTypeEnum,
        title: str,
        layer_name: str = "objects",
        metadata: dict[str, Any] | None = None,
    ) -> gpd.GeoDataFrame:
        """Create metric buffers while preserving every source feature property."""

        result = layer.copy()
        if result.crs is None:
            result = result.set_crs(4326)
        else:
            result = result.to_crs(4326)
        if "source_layer" not in result:
            result["source_layer"] = layer_name
        if "object_ref" not in result:
            result["object_ref"] = [
                cls._object_ref(row, layer_name, position)
                for position, (_, row) in enumerate(result.iterrows())
            ]
        result["buffer_size"] = buffer_size
        result["restriction_title"] = title
        for key, value in (metadata or {}).items():
            if key not in {"buffer_size", "buffer_type", "title"}:
                result[key] = [value] * len(result)
        if result.empty:
            return result
        metric_crs = result.estimate_utm_crs()
        if metric_crs is None:
            raise ValueError(f"Cannot determine metric CRS for layer {layer_name!r}")
        metric = result.to_crs(metric_crs)
        metric["geometry"] = metric.buffer(buffer_size, cap_style=buffer_type)
        return metric.to_crs(4326)

    def generate_geometry_buffers(
        self,
        buffer_info: dict[str, dict[str, Any]],
        objects_geoms: dict[str, dict],
    ) -> dict[str, dict]:
        object_lookup = {
            key.casefold(): (key, self._feature_collection_to_gdf(key, value))
            for key, value in objects_geoms.items()
        }
        result_layers: dict[str, gpd.GeoDataFrame] = {}
        for requested_name, info in buffer_info.items():
            matched = object_lookup.get(requested_name.casefold())
            if matched is None:
                continue
            layer_name, layer = matched
            result_layers[requested_name] = self.create_buffer(
                layer,
                info["buffer_size"],
                info["buffer_type"],
                info["title"],
                layer_name=layer_name,
                metadata=info,
            )
        return {
            name: json.loads(layer.to_json()) for name, layer in result_layers.items()
        }

    async def async_generate_geometry_buffers(
        self,
        buffer_info: dict[str, dict[str, Any]],
        objects_geoms: dict[str, dict],
    ) -> dict[str, dict]:
        return await asyncio.to_thread(
            self.generate_geometry_buffers, buffer_info, objects_geoms
        )

    @classmethod
    def form_layers(
        cls,
        layers: dict[str, dict],
        restriction_generators: list[str],
        restricted_objects: list[str],
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Return generator features and one joined row per geometric intersection."""

        generator_names = {name.casefold() for name in restriction_generators}
        object_names = {name.casefold() for name in restricted_objects}
        generators: list[gpd.GeoDataFrame] = []
        objects: list[gpd.GeoDataFrame] = []
        for name, feature_collection in layers.items():
            current = cls._feature_collection_to_gdf(name, feature_collection)
            if name.casefold() in generator_names:
                generators.append(current)
            if name.casefold() in object_names:
                objects.append(current)

        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
        if not generators:
            return empty, empty.copy()
        generators_gdf = gpd.GeoDataFrame(
            pd.concat(generators, ignore_index=True), geometry="geometry", crs=4326
        )
        if not objects:
            return generators_gdf, empty.copy()
        objects_gdf = gpd.GeoDataFrame(
            pd.concat(objects, ignore_index=True), geometry="geometry", crs=4326
        )

        generator_columns = generators_gdf.copy().rename(
            columns={
                "_row_id": "_generator_row_id",
                "source_layer": "_generator_layer",
                "object_ref": "_generator_ref",
                "buffer_size": "_generator_buffer_size",
                "restriction_title": "_generator_restriction_title",
            }
        )
        keep = [
            column
            for column in (
                "geometry",
                "_generator_row_id",
                "_generator_layer",
                "_generator_ref",
                "_generator_buffer_size",
                "_generator_restriction_title",
            )
            if column in generator_columns
        ]
        joined = gpd.sjoin(
            objects_gdf,
            generator_columns[keep],
            how="inner",
            predicate="intersects",
        ).drop(columns=["index_right"], errors="ignore")
        return generators_gdf, joined

    @staticmethod
    def _matches_rule(row: pd.Series, generator: str, rule: dict[str, Any]) -> bool:
        return str(
            row.get("_generator_layer", "")
        ).casefold() == generator.casefold() and str(
            row.get("source_layer", "")
        ).casefold() in {
            str(name).casefold() for name in rule.get("to", [])
        }

    @classmethod
    def _evidence(
        cls, row: pd.Series, generator: str, rule: dict[str, Any]
    ) -> dict[str, Any]:
        distance = cls._plain(row.get("_generator_buffer_size"))
        return {
            "reason_code": "BUFFER_INTERSECTION",
            "reason": (
                f"Геометрия объекта пересекает включительную буферную зону "
                f"слоя «{generator}»"
            ),
            "operation": "intersects",
            "boundary_policy": "inclusive",
            "source_layer": generator,
            "target_layer": cls._plain(row.get("source_layer")),
            "generator_ref": row.get("_generator_ref"),
            "distance_m": distance,
            "title": rule.get("title"),
            "description": rule.get("description"),
            "origin": rule.get("origin", "user"),
            "restriction_id": rule.get("restriction_id"),
            "provenance": rule.get("provenance"),
        }

    def create_restrictions(
        self,
        layers: dict[str, dict],
        generators: list[str],
        objects: list[str],
        restrictions: dict[str, dict[str, Any]],
    ) -> dict[str, dict]:
        generators_gdf, intersections = self.form_layers(layers, generators, objects)
        matched: dict[str, dict[str, Any]] = {}
        for _, row in intersections.iterrows():
            evidence = [
                self._evidence(row, generator, rule)
                for generator, rule in restrictions.items()
                if self._matches_rule(row, generator, rule)
            ]
            if not evidence:
                continue
            row_id = str(row["_row_id"])
            if row_id not in matched:
                matched[row_id] = {"row": row, "evidence": []}
            existing = matched[row_id]["evidence"]
            known = {
                (
                    item.get("restriction_id"),
                    item.get("source_layer"),
                    str(item.get("generator_ref")),
                )
                for item in existing
            }
            for item in evidence:
                identity = (
                    item.get("restriction_id"),
                    item.get("source_layer"),
                    str(item.get("generator_ref")),
                )
                if identity not in known:
                    existing.append(item)
                    known.add(identity)

        object_rows: list[pd.Series] = []
        for item in matched.values():
            row = item["row"].copy()
            evidence = item["evidence"]
            row["restriction_name"] = "&".join(
                dict.fromkeys(
                    str(reason.get("title") or "Ограничение") for reason in evidence
                )
            )
            row["restriction_description"] = "&".join(
                dict.fromkeys(
                    str(reason.get("description") or "") for reason in evidence
                )
            )
            row["restriction_evidence"] = evidence
            object_rows.append(row)

        if object_rows:
            objects_gdf = gpd.GeoDataFrame(object_rows, geometry="geometry", crs=4326)
            objects_gdf = objects_gdf.drop(
                columns=[
                    column
                    for column in objects_gdf.columns
                    if column.startswith("_generator_")
                    or column in {"_row_id", "_feature_id"}
                ],
                errors="ignore",
            )
        else:
            objects_gdf = gpd.GeoDataFrame(
                {"geometry": []}, geometry="geometry", crs=4326
            )

        generators_gdf = generators_gdf.drop(
            columns=["_row_id", "_feature_id"], errors="ignore"
        )
        return {
            "objects": json.loads(objects_gdf.to_crs(4326).to_json()),
            "generators": json.loads(generators_gdf.to_crs(4326).to_json()),
        }

    async def async_create_restrictions(
        self,
        layers: dict[str, dict],
        generators: list[str],
        objects: list[str],
        restrictions: dict[str, dict[str, Any]],
    ) -> dict[str, dict]:
        return await asyncio.to_thread(
            self.create_restrictions, layers, generators, objects, restrictions
        )
