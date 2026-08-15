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

        # Both halves grow without bound and either can be the large one: some
        # scenarios have thousands of distinct restrictions, others a single one
        # whose affected objects each carry kilobytes of evidence. Dividing by
        # restrictions alone left the second kind whole — and those were exactly
        # the rows that kept coming back 400.
        items = [
            json.dumps(row, ensure_ascii=False)
            for row in self._restriction_rows(objects) + self._affected_rows(objects)
        ]
        if not items:
            return [context[:budget_chars]]

        header = await self._generators_block(generators)
        room = budget_chars - len(self._render_part(header, [], 1, 1, len(items)))
        if room < 200:
            return [context[:budget_chars]]

        groups: list[list[str]] = [[]]
        size = 0
        for item in items:
            # A single object's evidence can be larger than the whole budget;
            # truncating it keeps the part valid instead of losing the row.
            item = item if len(item) <= room else item[: room - 20] + '…"'
            if groups[-1] and size + len(item) + 2 > room:
                groups.append([])
                size = 0
            groups[-1].append(item)
            size += len(item) + 2
        return [
            self._render_part(header, group, i, len(groups), len(items))
            for i, group in enumerate(groups, start=1)
        ]

    @staticmethod
    def _affected_rows(objects: dict) -> list[dict]:
        """The per-object detail entries, which carry the evidence."""

        if not objects.get("features"):
            return []
        rows = []
        for feature in objects["features"][:MAX_AFFECTED_OBJECT_DETAILS]:
            properties = feature.get("properties") or {}
            object_ref = properties.get("object_ref") or {}
            rows.append(
                {
                    "object_id": object_ref.get("id"),
                    "object_name": object_ref.get("name"),
                    "layer": properties.get("source_layer"),
                    "reasons": properties.get("restriction_evidence") or [],
                }
            )
        return rows

    @staticmethod
    def _render_part(
        header: str, items: list[str], part: int, parts: int, total_items: int
    ) -> str:
        body = "[" + ", ".join(items) + "]"
        return f"""Сводная информация по сформированным ограничениям (часть {part} из {parts}):\n
        Генераторы ограничений:
        \n{header}
        \nОбъекты, подверженные ограничениям — {len(items)} записей из {total_items}:
        \n{body}"""

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
