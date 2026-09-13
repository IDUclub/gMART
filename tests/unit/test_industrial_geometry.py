import math

from pyproj import Transformer

from tests.integration.industrial.geometry import is_fifty_metre_buffer


def test_equal_area_square_does_not_count_as_fifty_metre_buffer():
    project = Transformer.from_crs(32636, 4326, always_xy=True).transform
    center = (500000, 6651411)
    point = list(project(*center))
    source = {"features": [{"geometry": {"type": "Point", "coordinates": point}}]}

    def polygon(offsets):
        points = [list(project(center[0] + x, center[1] + y)) for x, y in offsets]
        return {
            "features": [
                {"geometry": {"type": "Polygon", "coordinates": [points + [points[0]]]}}
            ]
        }

    circle = polygon(
        [
            (50 * math.cos(i * math.tau / 64), 50 * math.sin(i * math.tau / 64))
            for i in range(64)
        ]
    )
    half = math.sqrt(math.pi * 2500) / 2
    square = polygon([(-half, -half), (half, -half), (half, half), (-half, half)])
    assert is_fifty_metre_buffer(circle, source)
    assert not is_fifty_metre_buffer(square, source)
