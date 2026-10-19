"""Recognising МСК-47 zones of user files that carry no usable CRS."""

import pytest
from pyproj import Transformer

from src.agents.services.compilance.local_crs import (
    MSK47_ZONES,
    guess_zone,
    zone_by_coordinates,
    zone_by_name,
)

ZONE2_BOUNDS = (2043570.0, 373937.0, 2151738.0, 444348.0)


@pytest.mark.parametrize(
    ("name", "zone"),
    [
        ("МСК 47 зона 2", 2),
        ("мск-47_зона_3", 3),
        ("Pulkovo 1942 / MSK47 zone 1", 1),
        ("msk47z2", 2),
        ("МСК-47 2", 2),
        ("МСК 47, 2024", None),
        ("МСК-50 зона 2", None),
        ("houses", None),
        (None, None),
    ],
)
def test_zone_by_name(name, zone):
    found = zone_by_name(name)
    assert (found and found.key) == (zone and f"msk47_zone{zone}")


@pytest.mark.parametrize(
    ("bounds", "zone"),
    [
        (ZONE2_BOUNDS, 2),
        ((1100000.0, 300000.0, 1300000.0, 500000.0), 1),
        ((3200000.0, 300000.0, 3300000.0, 500000.0), 3),
        ((4174000.0, 7495000.0, 4174000.0, 7495000.0), None),  # Web Mercator
        ((2100000.0, 6600000.0, 2100000.0, 6600000.0), None),  # northing as in СК-42
        ((1700000.0, 400000.0, 1800000.0, 400000.0), None),  # between zones
    ],
)
def test_zone_by_coordinates(bounds, zone):
    found = zone_by_coordinates(bounds)
    assert (found and found.key) == (zone and f"msk47_zone{zone}")


def test_named_zone_must_fit_the_coordinates():
    assert guess_zone(["МСК 47 зона 2"], ZONE2_BOUNDS) is MSK47_ZONES[2]
    assert guess_zone(["МСК 47 зона 1"], ZONE2_BOUNDS) is None
    assert guess_zone(["houses"], ZONE2_BOUNDS) is MSK47_ZONES[2]


def test_zone2_origin_at_point_one_degree_matches_the_other_zones_convention():
    """0.1° N with FN −6 200 000 equals 0° N with FN −6 211 057.628 to millimetres."""
    published = MSK47_ZONES[2]
    shifted = type(published)(
        "x", "x", 0, published.central_meridian, published.false_easting, -6211057.628
    )
    a = Transformer.from_crs(published.crs(), "EPSG:4284", always_xy=True)
    b = Transformer.from_crs(shifted.crs(), "EPSG:4284", always_xy=True)
    lon_a, lat_a = a.transform(2077804.26, 380959.28)
    lon_b, lat_b = b.transform(2077804.26, 380959.28)
    assert abs(lon_a - lon_b) < 1e-7 and abs(lat_a - lat_b) < 1e-7


@pytest.mark.parametrize("zone", [1, 2, 3])
def test_central_meridian_carries_the_pulkovo_datum_shift(zone):
    """A ballpark transform would land exactly on the meridian; a datum shift does not."""
    crs = MSK47_ZONES[zone]
    to_wgs84 = Transformer.from_crs(crs.crs(), "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(crs.false_easting, 400000)
    assert 59.5 < lat < 59.7
    shift_m = (crs.central_meridian - lon) * 111_320 * 0.506
    assert 80 < shift_m < 200
