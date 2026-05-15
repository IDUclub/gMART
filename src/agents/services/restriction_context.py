import json

import geopandas as gpd
import pandas as pd


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
        return json.dumps(
            generators.groupby("name", as_index=False)
            .agg({"name": "first", "area": "sum", "num": "sum"})
            .rename(
                columns={
                    "name": "Название",
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
        return json.dumps(
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
