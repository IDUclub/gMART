import json

import geopandas as gpd
import pandas as pd

MAX_AFFECTED_OBJECT_DETAILS = 10


class RestrictionContextBuilder:
    """
    Static class for building restriction context.
    """

    async def generate_buffers_context(self, buffers: dict) -> str:
        buffer_layers = [
            self._feature_collection_to_gdf(name, buffer)
            for name, buffer in buffers.items()
        ]
        buffers_gdf = pd.concat(buffer_layers)
        buffers_summary = await self.generate_generators_summary(
            buffers_gdf.to_crs(buffers_gdf.estimate_utm_crs())
        )
        return f"""Сводная информация по сгенерированным буферам ограничений:
        \n{buffers_summary}
        """

    async def generate_restrictions_context_chunks(
        self, generators: dict, objects: dict, budget_chars: int
    ) -> list[str]:
        """The same context, split into parts that each fit ``budget_chars``.

        A scenario with thousands of distinct restrictions produces a summary
        table the model cannot be handed in one call. The table is what gets
        divided — every part repeats the generators block and the totals, so a
        part is self-contained and can be summarised on its own. One part means
        one call and a context byte-identical to the unsplit one, so ordinary
        scenarios are unaffected.
        """

        context = await self.generate_restrictions_context(generators, objects)
        if budget_chars <= 0 or len(context) <= budget_chars:
            return [context]

        rows = self._restriction_rows(objects)
        if len(rows) <= 1:
            # Nothing left to divide: one enormous row, or none at all. The
            # caller still has the 400 to fall back on.
            return [context]

        header = await self._generators_block(generators)
        overhead = len(self._render_part(header, [], 1, 1, len(rows)))
        per_row = max(len(json.dumps(row, ensure_ascii=False)) + 2 for row in rows)
        rows_per_part = max(1, (budget_chars - overhead) // per_row)
        groups = [
            rows[i : i + rows_per_part] for i in range(0, len(rows), rows_per_part)
        ]
        return [
            self._render_part(header, group, i, len(groups), len(rows))
            for i, group in enumerate(groups, start=1)
        ]

    @staticmethod
    def _render_part(
        header: str, rows: list[dict], part: int, parts: int, total_rows: int
    ) -> str:
        return f"""Сводная информация по сформированным ограничениям (часть {part} из {parts}):\n
        Генераторы ограничений:
        \n{header}
        \nОбъекты, подверженные ограничениям — {len(rows)} из {total_rows} видов ограничений:
        \n{json.dumps(rows, ensure_ascii=False)}"""

    @staticmethod
    def _restriction_rows(objects: dict) -> list[dict]:
        """The aggregate rows, which is what grows without bound."""

        if not objects.get("features"):
            return []
        objects_gdf = gpd.GeoDataFrame.from_features(objects, crs=4326)
        objects_gdf = objects_gdf.to_crs(objects_gdf.estimate_utm_crs())
        objects_gdf["area"] = objects_gdf.area
        objects_gdf["num"] = 1
        return (
            objects_gdf.groupby("restriction_name", as_index=False)
            .agg({"restriction_description": "first", "area": "sum", "num": "sum"})
            .rename(
                columns={
                    "restriction_name": "Наименование ограничения",
                    "restriction_description": "Описание ограничения",
                    "area": "Площадь объектов кв.м",
                    "num": "Количество объектов",
                }
            )
            .to_dict(orient="records")
        )

    async def _generators_block(self, generators: dict) -> str:
        if not generators.get("features"):
            return ""
        generators_gdf = gpd.GeoDataFrame.from_features(generators, crs=4326)
        return await self.generate_generators_summary(
            generators_gdf.to_crs(generators_gdf.estimate_utm_crs())
        )

    async def generate_restrictions_context(
        self, generators: dict, objects: dict
    ) -> str:
        target_crs = None
        if generators["features"]:
            generators_gdf = gpd.GeoDataFrame.from_features(generators, crs=4326)
            target_crs = generators_gdf.estimate_utm_crs()
            generators_summary = await self.generate_generators_summary(
                generators_gdf.to_crs(target_crs)
            )
        else:
            generators_summary = ""

        if objects["features"]:
            objects_gdf = gpd.GeoDataFrame.from_features(objects, crs=4326)
            target_crs = target_crs or objects_gdf.estimate_utm_crs()
            objects_summary = await self.generate_objects_summary(
                objects_gdf.to_crs(target_crs)
            )
        else:
            objects_summary = ""

        return f"""Сводная информация по сформированным ограничениям:\n
        Генераторы ограничений:
        \n{generators_summary}
        \nОбъекты, подверженные ограничениям:
        \n{objects_summary}"""

    @staticmethod
    async def generate_generators_summary(generators: gpd.GeoDataFrame) -> str:
        generators["area"] = generators.area
        generators["num"] = 1
        grouping_column = "source_layer" if "source_layer" in generators else "name"
        return json.dumps(
            generators.groupby(grouping_column, as_index=False)
            .agg({grouping_column: "first", "area": "sum", "num": "sum"})
            .rename(
                columns={
                    grouping_column: "Название",
                    "area": "Площадь кв.м",
                    "num": "Количество",
                }
            )
            .to_dict(orient="records")
        )

    @staticmethod
    async def generate_objects_summary(objects: gpd.GeoDataFrame) -> str:
        objects["area"] = objects.area
        objects["num"] = 1
        aggregate = (
            objects.groupby("restriction_name", as_index=False)
            .agg(
                {
                    "restriction_description": "first",
                    "area": "sum",
                    "num": "sum",
                }
            )
            .rename(
                columns={
                    "restriction_name": "Наименование ограничения",
                    "restriction_description": "Описание ограничения",
                    "area": "Площадь объектов кв.м",
                    "num": "Количество объектов",
                }
            )
            .to_dict(orient="records")
        )
        details = []
        # A single object may contain evidence for several intersecting
        # generators. Sending one hundred such records can overflow the local
        # model context and produce an empty answer. The complete object list
        # remains in the returned GeoJSON; this is only the textual preview.
        for _, row in objects.head(MAX_AFFECTED_OBJECT_DETAILS).iterrows():
            object_ref = row.get("object_ref") or {}
            details.append(
                {
                    "object_id": object_ref.get("id"),
                    "object_name": object_ref.get("name"),
                    "layer": row.get("source_layer"),
                    "reasons": row.get("restriction_evidence") or [],
                }
            )
        return json.dumps(
            {
                "summary": aggregate,
                "affected_count": len(objects),
                "affected_objects": details,
                "details_truncated": len(objects) > len(details),
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _feature_collection_to_gdf(
        name: str, feature_collection: dict
    ) -> gpd.GeoDataFrame:
        """
        Function loads FeatureCollection dictionaries to
        :param name:
        :param feature_collection:
        :return:
        """

        gdf = gpd.GeoDataFrame.from_features(feature_collection, crs=4326)
        gdf["name"] = name
        return gdf
