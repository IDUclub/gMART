"""Independent geometric oracle: Steiner area formula, not buffer regeneration."""

import math

from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform, unary_union


def is_fifty_metre_buffer(layer, source):
    project = Transformer.from_crs(4326, 32636, always_xy=True).transform
    try:
        source_geom = transform(project, shape(source["features"][0]["geometry"]))
        actual = unary_union(
            [transform(project, shape(f["geometry"])) for f in layer["features"]]
        )
        expected_area = source_geom.area + source_geom.length * 50 + math.pi * 50**2
        return (
            actual.is_valid
            and actual.covers(source_geom)
            and abs(actual.area / expected_area - 1) < 0.01
            and all(
                abs(
                    actual.boundary.interpolate(i / 128, normalized=True).distance(
                        source_geom
                    )
                    - 50
                )
                < 0.5
                for i in range(128)
            )
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
