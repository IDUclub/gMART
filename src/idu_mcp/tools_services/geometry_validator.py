from fastmcp.exceptions import ToolError

from src.idu_mcp.tools_descriptions import geometry_validation_messages as messages


class GeometryToolValidator:
    """Validates geometry tool inputs before any geopandas operation runs.

    Raises :class:`ToolError` with an actionable, user-facing message so that
    bad input and empty (geometry-less) layers are reported clearly instead of
    surfacing as opaque geopandas tracebacks.
    """

    _BUFFER_FIELDS = ("buffer_size", "buffer_type", "title")
    _RESTRICTION_FIELDS = ("title", "description", "to")

    @staticmethod
    def _is_empty_layer(feature_collection) -> bool:
        """
        Returns True if a layer carries no usable geometry: missing, not a
        FeatureCollection, with an empty ``features`` list, or with every feature
        lacking a geometry. Such layers crash geopandas (``estimate_utm_crs`` /
        ``sjoin`` / ``concat``) downstream, so they are caught before that point.
        """

        if not isinstance(feature_collection, dict):
            return True
        features = feature_collection.get("features")
        if not features:
            return True
        return all(
            not isinstance(feature, dict) or not feature.get("geometry")
            for feature in features
        )

    @classmethod
    def validate_buffers(cls, buffer_info: dict, objects: dict) -> None:
        """Validate ``CreateBuffers`` inputs.

        Args:
            buffer_info (dict): Layer name -> buffer parameters.
            objects (dict): Layer name -> FeatureCollection to buffer.
        Raises:
            ToolError: if a requested layer is missing, a buffer parameter is
                absent, or a requested layer has no geometry.
        """

        objects_by_key = {key.lower(): value for key, value in objects.items()}

        missing_layers = [
            name for name in buffer_info if name.lower() not in objects_by_key
        ]
        if missing_layers:
            raise ToolError(
                messages.MISSING_BUFFER_LAYERS.format(
                    missing=missing_layers,
                    objects=list(objects),
                    requested=list(buffer_info),
                )
            )

        for name, info in buffer_info.items():
            missing_fields = [
                field
                for field in cls._BUFFER_FIELDS
                if not isinstance(info, dict) or field not in info
            ]
            if missing_fields:
                raise ToolError(
                    messages.MISSING_BUFFER_FIELDS.format(
                        layer=name, fields=missing_fields
                    )
                )

        empty_layers = [
            name
            for name in buffer_info
            if cls._is_empty_layer(objects_by_key[name.lower()])
        ]
        if empty_layers:
            raise ToolError(messages.EMPTY_BUFFER_LAYERS.format(empty=empty_layers))

    @classmethod
    def validate_restrictions(
        cls,
        generators: list[str],
        objects: list[str],
        restrictions: dict[str, dict[str, str | list[str]]],
        layers: dict,
    ) -> None:
        """Validate ``CreateRestrictions`` inputs.

        Args:
            generators (list[str]): Names of restriction-generating layers.
            objects (list[str]): Names of restricted (target) layers.
            restrictions (dict): Restriction name -> metadata (title, description, to).
            layers (dict): Layer name -> FeatureCollection.
        Raises:
            ToolError: if no generator/target layer is present, a restriction
                field is absent, or a referenced layer has no geometry.
        """

        if not any(name in layers for name in generators) or not any(
            name in layers for name in objects
        ):
            raise ToolError(
                messages.MISSING_RESTRICTION_LAYERS.format(
                    layers=list(layers),
                    generators=generators,
                    objects=objects,
                )
            )

        for name, info in restrictions.items():
            missing_fields = [
                field
                for field in cls._RESTRICTION_FIELDS
                if not isinstance(info, dict) or field not in info
            ]
            if missing_fields:
                raise ToolError(
                    messages.MISSING_RESTRICTION_FIELDS.format(
                        name=name, fields=missing_fields
                    )
                )

        empty_layers = [
            name
            for name in (*generators, *objects)
            if name in layers and cls._is_empty_layer(layers[name])
        ]
        if empty_layers:
            raise ToolError(
                messages.EMPTY_RESTRICTION_LAYERS.format(empty=empty_layers)
            )
