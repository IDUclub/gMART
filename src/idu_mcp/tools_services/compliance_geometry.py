"""Deterministic geospatial primitives for executable normative checks."""

from __future__ import annotations

import json
import operator as operator_module
from typing import Any, Callable

import geopandas as gpd
import pandas as pd

from src.idu_mcp.tools_services.entites.buffer_type_enum import BufferTypeEnum
from src.idu_mcp.tools_services.geometry_tools import GeometryTools

_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "<": operator_module.lt,
    "<=": operator_module.le,
    ">": operator_module.gt,
    ">=": operator_module.ge,
    "==": operator_module.eq,
}


class ComplianceGeometryTools:
    """Bounded operations returning explicit passed, violated and unchecked baskets."""

    def __init__(self, max_features: int = 50_000) -> None:
        self.max_features = max_features

    def _layer(self, name: str, layers: dict[str, dict]) -> gpd.GeoDataFrame:
        if name not in layers:
            raise ValueError(f"Layer {name!r} is missing")
        features = layers[name].get("features") or []
        frame = (
            GeometryTools._feature_collection_to_gdf(name, layers[name])
            if features
            else gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
        )
        flattened = pd.json_normalize(
            [feature.get("properties") or {} for feature in features],
            sep=".",
        )
        for column in flattened.columns:
            if column not in frame.columns:
                frame[column] = flattened[column].values
        if len(frame) > self.max_features:
            raise ValueError(
                f"Layer {name!r} exceeds the {self.max_features} feature limit"
            )
        if frame.crs is None:
            frame = frame.set_crs(4326)
        return frame.to_crs(4326)

    @staticmethod
    def _combine(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        present = [frame for frame in frames if not frame.empty]
        if not present:
            return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
        return gpd.GeoDataFrame(
            pd.concat(present, ignore_index=True), geometry="geometry", crs=4326
        )

    @staticmethod
    def _to_fc(frame: gpd.GeoDataFrame) -> dict[str, Any]:
        clean = frame.drop(
            columns=[name for name in frame.columns if name.startswith("_")],
            errors="ignore",
        )
        return json.loads(clean.to_crs(4326).to_json(drop_id=True))

    @staticmethod
    def _metric_crs(*frames: gpd.GeoDataFrame):
        combined = ComplianceGeometryTools._combine(list(frames))
        if combined.empty:
            return "EPSG:3857"
        crs = combined.estimate_utm_crs()
        if crs is None:
            raise ValueError("Cannot determine a local metric CRS")
        return crs

    @staticmethod
    def _prepare_right(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        right = frame[["geometry", "object_ref", "source_layer"]].copy()
        return right.rename(
            columns={
                "object_ref": "_join_generator_ref",
                "source_layer": "_join_generator_layer",
            }
        )

    @classmethod
    def _matched_refs(
        cls,
        objects: gpd.GeoDataFrame,
        generators: gpd.GeoDataFrame,
        predicate: str,
    ) -> dict[str, list[dict[str, Any]]]:
        if objects.empty or generators.empty:
            return {}
        right = cls._prepare_right(generators)
        joined = gpd.sjoin(objects, right, how="left", predicate=predicate)
        matches: dict[str, list[dict[str, Any]]] = {}
        for _, row in joined.iterrows():
            row_id = str(row["_row_id"])
            ref = row.get("_join_generator_ref")
            if not isinstance(ref, dict):
                continue
            bucket = matches.setdefault(row_id, [])
            if ref not in bucket:
                bucket.append(ref)
        return matches

    @staticmethod
    def _annotate(
        objects: gpd.GeoDataFrame,
        matches: dict[str, list[dict[str, Any]]],
        *,
        violation_when: str,
        operation: str,
        restriction_id: str,
        template: str,
        template_version: int,
        threshold: float | None = None,
        unit: str | None = None,
        provenance: dict[str, Any] | None = None,
        input_revision: str | None = None,
    ) -> tuple[gpd.GeoDataFrame, list[dict[str, Any]]]:
        result = objects.copy()
        evidence: list[dict[str, Any]] = []
        violated_values: list[bool] = []
        evidence_values: list[list[dict[str, Any]]] = []
        for _, row in result.iterrows():
            refs = matches.get(str(row["_row_id"]), [])
            matched = bool(refs)
            violated = matched if violation_when == "matched" else not matched
            item = {
                "restriction_id": restriction_id,
                "template": template,
                "template_version": template_version,
                "object_ref": row["object_ref"],
                "generator_ref": refs[0] if len(refs) == 1 else None,
                "generator_refs": refs,
                "operation": operation,
                "measured_value": len(refs),
                "unit": unit,
                "threshold": threshold,
                "operator": "matched" if violation_when == "matched" else "not_matched",
                "violated": violated,
                "used_fields": [],
                "provenance": provenance or {},
                "warnings": [],
                "input_revision": input_revision,
            }
            evidence.append(item)
            violated_values.append(violated)
            evidence_values.append([item])
        result["verification_status"] = "complete"
        result["compliance_status"] = [
            "violated" if value else "passed" for value in violated_values
        ]
        result["restriction_id"] = restriction_id
        result["compliance_evidence"] = evidence_values
        result["_violated"] = violated_values
        return result, evidence

    def distance_from_source(
        self,
        *,
        source_layer: str,
        targets: list[str],
        geometry_mode: str,
        predicate: str,
        violation_when: str,
        result_mode: str,
        layers: dict[str, dict],
        restriction_id: str,
        template_version: int = 1,
        distance_m: float | None = None,
        provenance: dict[str, Any] | None = None,
        input_revision: str | None = None,
    ) -> dict[str, Any]:
        source = self._layer(source_layer, layers)
        objects = self._combine([self._layer(name, layers) for name in targets])
        generators = source
        operation = predicate
        if geometry_mode == "buffered":
            if distance_m is None or distance_m <= 0:
                raise ValueError("distance_m must be positive for buffered geometry")
            generators = GeometryTools.create_buffer(
                source,
                distance_m,
                BufferTypeEnum.ROUND,
                "Executable norm buffer",
                layer_name=source_layer,
            )
            operation = f"buffer({distance_m}m)+{predicate}"
        elif distance_m is not None:
            raise ValueError("distance_m is forbidden for source_geometry")
        matches = self._matched_refs(objects, generators, predicate)
        annotated, evidence = self._annotate(
            objects,
            matches,
            violation_when=violation_when,
            operation=operation,
            restriction_id=restriction_id,
            template="distance_from_source",
            template_version=template_version,
            threshold=distance_m,
            unit="m" if distance_m is not None else None,
            provenance=provenance,
            input_revision=input_revision,
        )
        return self._result(annotated, evidence, result_mode)

    def distance_table(
        self,
        *,
        source_layer: str,
        targets: list[str],
        attribute_field: str,
        bands: list[dict[str, Any]],
        predicate: str,
        violation_when: str,
        result_mode: str,
        layers: dict[str, dict],
        restriction_id: str,
        template_version: int = 1,
        provenance: dict[str, Any] | None = None,
        input_revision: str | None = None,
    ) -> dict[str, Any]:
        source = self._layer(source_layer, layers)
        objects = self._combine([self._layer(name, layers) for name in targets])
        if not source.empty and attribute_field not in source.columns:
            raise ValueError(f"Attribute {attribute_field!r} is missing")
        metric_crs = self._metric_crs(source, objects)
        metric = source.to_crs(metric_crs).copy()
        distances: list[float | None] = []
        values = (
            pd.to_numeric(metric[attribute_field], errors="coerce")
            if attribute_field in metric.columns
            else []
        )
        for value in values:
            selected = next(
                (
                    float(band["distance_m"])
                    for band in bands
                    if not pd.isna(value)
                    and value >= band["min"]
                    and (band.get("max") is None or value <= band["max"])
                ),
                None,
            )
            distances.append(selected)
        valid_mask = pd.Series(
            [item is not None for item in distances], index=metric.index
        )
        valid = metric.loc[valid_mask].copy()
        valid["_distance_m"] = [item for item in distances if item is not None]
        if not valid.empty:
            valid["geometry"] = [
                geom.buffer(distance, cap_style=BufferTypeEnum.ROUND)
                for geom, distance in zip(valid.geometry, valid["_distance_m"])
            ]
        generators = valid.to_crs(4326)
        matches = self._matched_refs(objects, generators, predicate)
        annotated, evidence = self._annotate(
            objects,
            matches,
            violation_when=violation_when,
            operation=f"variable_buffer({attribute_field})+{predicate}",
            restriction_id=restriction_id,
            template="distance_table",
            template_version=template_version,
            unit="m",
            provenance=provenance,
            input_revision=input_revision,
        )
        unchecked_source_count = int((~valid_mask).sum())
        if unchecked_source_count:
            stable_ids = set(matches)
            uncertain = ~annotated["_row_id"].astype(str).isin(stable_ids)
            annotated["_violated"] = annotated["_violated"].astype(object)
            annotated.loc[uncertain, "_violated"] = None
            annotated.loc[uncertain, "verification_status"] = "unverifiable"
            annotated.loc[uncertain, "compliance_status"] = "unknown"
            annotated.loc[uncertain, "compliance_evidence"] = annotated.loc[
                uncertain, "compliance_evidence"
            ].map(lambda _: [])
            checked_refs = annotated.loc[~uncertain, "object_ref"].tolist()
            evidence = [item for item in evidence if item["object_ref"] in checked_refs]
        distance_by_ref = {
            json.dumps(row["object_ref"], ensure_ascii=False, sort_keys=True): {
                "field": attribute_field,
                "quality": "direct",
                "value": row[attribute_field],
                "distance_m": row["_distance_m"],
            }
            for _, row in valid.iterrows()
        }
        for item in evidence:
            item["used_fields"] = [
                distance_by_ref[key]
                for ref in item["generator_refs"]
                if (key := json.dumps(ref, ensure_ascii=False, sort_keys=True))
                in distance_by_ref
            ]
            distances_used = [value["distance_m"] for value in item["used_fields"]]
            item["threshold"] = max(distances_used) if distances_used else None
        result = self._result(annotated, evidence, result_mode)
        result["source_coverage"] = {
            "applicable_objects": len(source),
            "checked_objects": int(valid_mask.sum()),
            "unchecked_objects": int((~valid_mask).sum()),
            "fill_rate": float(valid_mask.mean()) if len(valid_mask) else 1.0,
        }
        return result

    def presence_within(
        self,
        *,
        objects_layer: str,
        required_neighbor_layers: list[str],
        distance_m: float,
        minimum_neighbors: int,
        result_mode: str,
        layers: dict[str, dict],
        restriction_id: str,
        template_version: int = 1,
        provenance: dict[str, Any] | None = None,
        input_revision: str | None = None,
    ) -> dict[str, Any]:
        objects = self._layer(objects_layer, layers)
        neighbor_frames = {
            name: self._layer(name, layers) for name in required_neighbor_layers
        }
        neighbors = self._combine(list(neighbor_frames.values()))
        metric_crs = self._metric_crs(objects, neighbors)
        buffers = objects.to_crs(metric_crs).copy()
        buffers["geometry"] = buffers.geometry.buffer(
            distance_m, cap_style=BufferTypeEnum.ROUND
        )
        wgs84_buffers = buffers.to_crs(4326)
        matches_by_layer = {
            name: self._matched_refs(wgs84_buffers, frame, "intersects")
            for name, frame in neighbor_frames.items()
        }
        annotated = objects.copy()
        evidence: list[dict[str, Any]] = []
        violated_values: list[bool] = []
        evidence_values: list[list[dict[str, Any]]] = []
        for _, row in annotated.iterrows():
            row_id = str(row["_row_id"])
            refs_by_layer = {
                name: matches.get(row_id, [])
                for name, matches in matches_by_layer.items()
            }
            neighbor_counts = {name: len(refs) for name, refs in refs_by_layer.items()}
            refs = [ref for layer_refs in refs_by_layer.values() for ref in layer_refs]
            count = min(neighbor_counts.values(), default=0)
            violated = any(
                layer_count < minimum_neighbors
                for layer_count in neighbor_counts.values()
            )
            item = {
                "restriction_id": restriction_id,
                "template": "presence_within",
                "template_version": template_version,
                "object_ref": row["object_ref"],
                "generator_ref": refs[0] if len(refs) == 1 else None,
                "generator_refs": refs,
                "operation": "neighbors_within",
                "measured_value": count,
                "neighbor_count": count,
                "neighbor_counts": neighbor_counts,
                "unit": "count",
                "threshold": minimum_neighbors,
                "operator": ">=",
                "violated": violated,
                "used_fields": [],
                "provenance": provenance or {},
                "warnings": [],
                "radius_m": distance_m,
                "neighbor_layers": required_neighbor_layers,
                "input_revision": input_revision,
            }
            evidence.append(item)
            violated_values.append(violated)
            evidence_values.append([item])
        annotated["verification_status"] = "complete"
        annotated["compliance_status"] = [
            "violated" if value else "passed" for value in violated_values
        ]
        annotated["restriction_id"] = restriction_id
        annotated["compliance_evidence"] = evidence_values
        annotated["_violated"] = violated_values
        return self._result(annotated, evidence, result_mode)

    def zonal_attribute_threshold(
        self,
        *,
        objects_layer: str,
        zones_layer: str,
        object_attribute: str,
        operator: str,
        constant_threshold: float | None,
        zone_threshold_attribute: str | None,
        join_predicate: str,
        result_mode: str,
        layers: dict[str, dict],
        restriction_id: str,
        template_version: int = 1,
        provenance: dict[str, Any] | None = None,
        input_revision: str | None = None,
    ) -> dict[str, Any]:
        objects = self._layer(objects_layer, layers)
        zones = self._layer(zones_layer, layers)
        if object_attribute not in objects.columns:
            raise ValueError(f"Object attribute {object_attribute!r} is missing")
        if constant_threshold is None and (
            not zone_threshold_attribute
            or zone_threshold_attribute not in zones.columns
        ):
            raise ValueError("A constant or resolvable zone threshold is required")
        zone_columns = ["geometry", "object_ref"]
        if zone_threshold_attribute:
            zone_columns.append(zone_threshold_attribute)
        right = zones[zone_columns].rename(
            columns=(
                {
                    "object_ref": "_join_zone_ref",
                    zone_threshold_attribute: "_join_threshold",
                }
                if zone_threshold_attribute
                else {"object_ref": "_join_zone_ref"}
            )
        )
        joined = gpd.sjoin(objects, right, how="left", predicate=join_predicate)
        grouped = {key: value for key, value in joined.groupby("_row_id", sort=False)}
        annotated = objects.copy()
        violation_values: list[Any] = []
        status_values: list[str] = []
        evidence_values: list[list[dict[str, Any]]] = []
        evidence: list[dict[str, Any]] = []
        compare = _OPERATORS[operator]
        for _, row in annotated.iterrows():
            candidates = grouped.get(row["_row_id"])
            object_value = pd.to_numeric(
                pd.Series([row[object_attribute]]), errors="coerce"
            ).iloc[0]
            zone_rows = (
                []
                if candidates is None
                else [
                    item
                    for _, item in candidates.iterrows()
                    if isinstance(item.get("_join_zone_ref"), dict)
                ]
            )
            threshold_values = (
                [float(constant_threshold)]
                if constant_threshold is not None and zone_rows
                else []
            )
            if constant_threshold is None:
                threshold_values = [
                    float(value)
                    for item in zone_rows
                    if not pd.isna(
                        value := pd.to_numeric(
                            pd.Series([item.get("_join_threshold")]), errors="coerce"
                        ).iloc[0]
                    )
                ]
            if pd.isna(object_value) or not threshold_values:
                violation_values.append(None)
                status_values.append("unknown")
                evidence_values.append([])
                continue
            if operator == "==" and len(set(threshold_values)) > 1:
                violation_values.append(None)
                status_values.append("unknown")
                evidence_values.append([])
                continue
            threshold = (
                min(threshold_values)
                if operator in {"<", "<="}
                else max(threshold_values)
            )
            compliant = bool(compare(float(object_value), threshold))
            violated = not compliant
            zone_refs = [item["_join_zone_ref"] for item in zone_rows]
            item = {
                "restriction_id": restriction_id,
                "template": "zonal_attribute_threshold",
                "template_version": template_version,
                "object_ref": row["object_ref"],
                "zone_ref": zone_refs[0] if len(zone_refs) == 1 else None,
                "zone_refs": zone_refs,
                "operation": f"{join_predicate}+attribute_compare",
                "measured_value": float(object_value),
                "threshold": threshold,
                "operator": operator,
                "violated": violated,
                "used_fields": [{"field": object_attribute, "quality": "direct"}],
                "provenance": provenance or {},
                "warnings": [],
                "input_revision": input_revision,
            }
            evidence.append(item)
            evidence_values.append([item])
            violation_values.append(violated)
            status_values.append("violated" if violated else "passed")
        annotated["verification_status"] = [
            "complete" if value is not None else "unverifiable"
            for value in violation_values
        ]
        annotated["compliance_status"] = status_values
        annotated["restriction_id"] = restriction_id
        annotated["compliance_evidence"] = evidence_values
        annotated["_violated"] = violation_values
        return self._result(annotated, evidence, result_mode)

    def zonal_ratio(
        self,
        *,
        zones_layer: str,
        numerator_layer: str,
        operator: str,
        threshold: float,
        result_mode: str,
        invalid_geometry_policy: str,
        layers: dict[str, dict],
        restriction_id: str,
        template_version: int = 1,
        provenance: dict[str, Any] | None = None,
        input_revision: str | None = None,
    ) -> dict[str, Any]:
        zones = self._layer(zones_layer, layers)
        numerator = self._layer(numerator_layer, layers)
        invalid_zones = ~zones.geometry.is_valid
        invalid_numerator = ~numerator.geometry.is_valid
        warnings: list[str] = []
        if invalid_zones.any() or invalid_numerator.any():
            if invalid_geometry_policy == "reject":
                raise ValueError("Invalid geometry found")
            zones.loc[invalid_zones, "geometry"] = zones.loc[
                invalid_zones
            ].geometry.make_valid()
            numerator.loc[invalid_numerator, "geometry"] = numerator.loc[
                invalid_numerator
            ].geometry.make_valid()
            warnings.append("invalid_geometry_repaired")
        metric_crs = self._metric_crs(zones, numerator)
        zones_m = zones.to_crs(metric_crs)
        numerator_m = numerator.to_crs(metric_crs)
        compare = _OPERATORS[operator]
        evidence: list[dict[str, Any]] = []
        violation_values: list[Any] = []
        status_values: list[str] = []
        evidence_values: list[list[dict[str, Any]]] = []
        for _, zone in zones_m.iterrows():
            denominator_area = float(zone.geometry.area)
            if denominator_area <= 0:
                violation_values.append(None)
                status_values.append("unknown")
                evidence_values.append([])
                continue
            clipped = [
                geom.intersection(zone.geometry)
                for geom in numerator_m.geometry
                if geom is not None and geom.intersects(zone.geometry)
            ]
            numerator_area = (
                float(gpd.GeoSeries(clipped, crs=metric_crs).union_all().area)
                if clipped
                else 0.0
            )
            ratio = numerator_area / denominator_area * 100
            violated = not bool(compare(ratio, threshold))
            item = {
                "restriction_id": restriction_id,
                "template": "zonal_ratio",
                "template_version": template_version,
                "object_ref": zone["object_ref"],
                "zone_ref": zone["object_ref"],
                "operation": "clip+union+area_ratio",
                "measured_value": ratio,
                "unit": "%",
                "threshold": threshold,
                "operator": operator,
                "violated": violated,
                "used_fields": [],
                "provenance": provenance or {},
                "warnings": warnings,
                "numerator_area_m2": numerator_area,
                "denominator_area_m2": denominator_area,
                "input_revision": input_revision,
            }
            evidence.append(item)
            evidence_values.append([item])
            violation_values.append(violated)
            status_values.append("violated" if violated else "passed")
        zones["verification_status"] = [
            "complete" if value is not None else "unverifiable"
            for value in violation_values
        ]
        zones["compliance_status"] = status_values
        zones["restriction_id"] = restriction_id
        zones["compliance_evidence"] = evidence_values
        zones["_violated"] = violation_values
        return self._result(zones, evidence, result_mode)

    def _result(
        self,
        annotated: gpd.GeoDataFrame,
        evidence: list[dict[str, Any]],
        result_mode: str,
    ) -> dict[str, Any]:
        verified = annotated["_violated"].notna()
        violated_mask = annotated["_violated"] == True  # noqa: E712
        passed_mask = annotated["_violated"] == False  # noqa: E712
        unchecked_mask = ~verified
        applicable = len(annotated)
        checked = int(verified.sum())
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=4326)
        violated = (
            annotated.loc[violated_mask].copy()
            if result_mode in {"violated", "both"}
            else empty
        )
        passed = (
            annotated.loc[passed_mask].copy()
            if result_mode in {"passed", "both"}
            else empty
        )
        unchecked = annotated.loc[unchecked_mask].copy()
        return {
            "coverage": {
                "applicable_objects": applicable,
                "checked_objects": checked,
                "unchecked_objects": applicable - checked,
                "fill_rate": checked / applicable if applicable else 1.0,
            },
            "summary": {
                "violated_objects": int(violated_mask.sum()),
                "passed_objects": int(passed_mask.sum()),
            },
            "violated_objects": self._to_fc(violated),
            "passed_objects": self._to_fc(passed),
            "unchecked_objects": self._to_fc(unchecked),
            "evidence": evidence,
        }
