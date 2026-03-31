import asyncio
import json

import geopandas as gpd
import pandas as pd

from src.idu_mcp.tools_services.entites import BufferTypeEnum


class GeometryTools:

    def __init__(self):
        pass

    #TODO add restrict style
    @staticmethod
    def create_buffer(
        layer: gpd.GeoDataFrame, buffer_size: int, buffer_type: BufferTypeEnum, title: str
    ) -> gpd.GeoDataFrame:
        """
        Function generates buffer for provided layer.
        Args:
            layer (gpd.GeoDataFrame): GeoDataFrame of objects layer.
            buffer_size (int): Buffer size in meters for layer objects.
            buffer_type (BufferTypeEnum): Buffer type to generate. Possible values: "round", "square", "flat".
            title (str): Generating restriction name, close to description.
        Returns:
            gpd.GeoDataFrame: GeoDataFrame of buffered objects.
        """

        original_crs = layer.crs
        if not original_crs:
            layer.set_crs(4326, inplace=True)
        if layer.crs != 4326:
            layer.to_crs(4326, inplace=True)
        layer.to_crs(layer.estimate_utm_crs(), inplace=True)
        layer["buffer_size"] = buffer_size
        layer["restriction_title"] = title
        layer["geometry"] = layer.buffer(buffer_size, cap_style=buffer_type)
        return layer[["geometry", "name", "buffer_size", "restriction_title"]].to_crs(4326)

    def generate_geometry_buffers(
        self,
        buffer_info: dict[str, BufferTypeEnum | int],
        objects_geoms: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Function generates buffer for provided names asynchronously.
        Args:
            buffer_info (dict[str, BufferTypeEnum | int]): dict with buffers information.
            objects_geoms (dict[str, dict]): dict with objects geometry information.
        Returns:
            dict[str, dict]: dict with buffers information if object key was in names.
        """

        objects_geoms = {
            k.lower(): gpd.GeoDataFrame.from_features(v) for k, v in objects_geoms.items()
        }
        buffer_info = {k.lower(): v for k, v in buffer_info.items()}
        result_layers = {
            k: self.create_buffer(
                objects_geoms[k],
                buffer_info[k]["buffer_size"],
                buffer_info[k]["buffer_type"],
                buffer_info[k]["title"],
            )
            for k in buffer_info
        }
        return {k: json.loads(v.to_json()) for k, v in result_layers.items()}

    async def async_generate_geometry_buffers(
        self,
        buffer_info: dict[str, BufferTypeEnum | int],
        objects_geoms: dict[str, dict],
    ) -> dict[str, dict]:
        """
        Function generates buffer for provided names asynchronously.
        Args:
            buffer_info (dict[str, BufferTypeEnum | int]): dict with buffers information.
            objects_geoms (dict[str, dict]): dict with objects geometry information.
        Returns:
            dict[str, dict]: dict with buffers information if object key was in names.
        """

        return await asyncio.to_thread(
            self.generate_geometry_buffers, buffer_info, objects_geoms
        )

    @staticmethod
    def form_layers(
        layers: dict[str, dict],
        restriction_generators: list[str],
        restricted_objects: list[str],
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Function generates layers for provided objects:
        Args:
            layers (dict[str, dict]): dict with object type name as key and FeatureCollection dict as value.
            restriction_generators (list[str]): list of objects generating restrictions.
            restricted_objects (list[str]): list of objects which can be restricted.
        Returns:
            tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]: GeoDataFrame of objects layers.
            First layer is generators, second is restricted objects.
        """

        generators = []
        objects = []
        for k, v in layers.items():
            current_layer = gpd.GeoDataFrame.from_features(v, crs=4326)
            current_layer["name"] = k
            if k in restriction_generators:
                generators.append(current_layer)
            elif k in restricted_objects:
                objects.append(current_layer)
            else:
                continue

        generators, objects = pd.concat(generators), pd.concat(objects)
        joined = (
            objects.sjoin(generators)
            .reset_index(drop=False)
            .dissolve(
                "index", aggfunc={"name_left": "first", "name_right": lambda x: set(x)}
            )
        )
        generators = generators.to_crs(generators.estimate_utm_crs())
        if len(joined) > 0:
            joined.to_crs(joined.estimate_utm_crs())
        return generators, joined
    def create_restrictions(
        self,
        layers: dict[str, dict],
        generators: list[str],
        objects: list[str],
        restrictions: dict[str, dict[str, str | list[str]]],
    ) -> dict[str, dict]:
        """
        Function generates restrictions for provided objects.
        Args:
            layers (dict[str, dict]): dict with object type name as key and FeatureCollection dict as value.
            generators (list[str]): list of objects which produces restrictions.
            objects (list[str]): list of objects which can be restricted.
            restrictions (dict[str, dict[str, str | list[str]]: dict with restrictions object o service type name
            as key and dict with fields  "title" and "description" containing restriction info and field "to" which
            contains list of str names of objects effected by restriction.
        Returns:
            dict[str, dict]: two layers with restricted objects and generated restrictions as dict.
        """

        def apply_strip(string_to_strip: str, symbol: str) -> str:
            return string_to_strip.lstrip(symbol)

        generators, objects = self.form_layers(layers, generators, objects)
        objects[["restriction_name", "restriction_description"]] = "", ""
        for k, v in restrictions.items():
            common = objects["name_left"].isin(v["to"]) & objects["name_right"].apply(
                lambda x: k in x
            )
            objects.loc[common, "restriction_name"] += f"&{v['title']}"
            objects.loc[common, "restriction_description"] += f"&{v['description']}"
        objects["restriction_name"] = objects[
            "restriction_name"
        ].apply(apply_strip, symbol="&")
        objects["restriction_description"] = objects["restriction_description"].apply(
            apply_strip, symbol="&"
        )
        objects.drop(columns=["name_left", "name_right"], inplace=True)
        return {
            "objects": json.loads(objects.to_crs(4326).to_json()),
            "generators": json.loads(generators.to_crs(4326).to_json())
        }

    async def async_create_restrictions(
        self,
        layers: dict[str, dict],
        generators: list[str],
        objects: list[str],
        restrictions: dict[str, dict[str, str]],
    ) -> dict[str | dict]:
        """
        Function generates restrictions for provided objects.
        Args:
            layers (dict[str, dict]): dict with object type name as key and FeatureCollection dict as value.
            generators (list[str]): list of objects which produces restrictions.
            objects (list[str]): list of objects which can be restricted.
            restrictions (dict[str, dict[str, str | list[str]]: dict with restrictions object o service type name
            as key and dict with fields  "title" and "description" containing restriction info and field "to" which
            contains list of str names of objects effected by restriction.
        Returns:
            tuple[dict, dict]: two layers with restricted objects and generated restrictions.
        """

        return await asyncio.to_thread(
            self.create_restrictions, layers, generators, objects, restrictions
        )
