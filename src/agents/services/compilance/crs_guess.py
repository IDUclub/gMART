"""Coordinate systems assumed for user files that declare none.

A file whose coordinates are not lon/lat and that names no CRS PROJ understands
is placed only in a system recognised with confidence; any other file is sent back
to be uploaded with its CRS. Two systems are recognised:

- МСК-47 — designs of the Leningrad oblast: Transverse Mercator on the Pulkovo 1942
  datum in three zones, none of them in the EPSG registry. A GeoJSON often names the
  zone only in prose (``"МСК 47 зона 2"``); without the name the coordinates tell it:
  the millions digit of the easting is the zone number and the northing is counted
  from about 56° N.
- Web Mercator (EPSG:3857) — exports of web maps. Recognised by its name or by
  coordinates that land within Russia. Gauss–Krüger zones of СК-42 / ГСК-2011
  share that range, so such a file without a CRS is read as Web Mercator too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from typing import Iterable

_MSK47_NAME = re.compile(
    r"(?:мск|msk)[\s_-]*47\D{0,12}?(?:зона|zone|z)?[\s_-]*([123])(?!\d)",
    re.IGNORECASE,
)
_WEB_MERCATOR_NAME = re.compile(
    r"(?<!\d)(?:3857|900913|3785)(?!\d)|pseudo[\s_-]*mercator|web[\s_-]*mercator",
    re.IGNORECASE,
)
# Farther than this from the central meridian a zone is not used.
_MAX_EASTING_OFFSET = 400_000
# The oblast lies within 58.4–61.3° N: northings 0.25–0.6 Mm, with a margin.
_NORTHING_RANGE = (150_000, 800_000)
# Russia, Kaliningrad to Chukotka west of the antimeridian; lon/lat degrees.
_RUSSIA = (19.0, 41.0, 180.0, 82.0)
_WEB_MERCATOR_LIMIT = 20_037_508.342789244

Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class LocalZone:
    """One zone of a local coordinate system."""

    key: str
    title: str
    latitude_of_origin: float
    central_meridian: float
    false_easting: float
    false_northing: float

    def wkt(self) -> str:
        return (
            f'PROJCRS["Pulkovo 1942 / {self.title}",'
            'BASEGEOGCRS["Pulkovo 1942",DATUM["Pulkovo 1942",'
            'ELLIPSOID["Krassowsky 1940",6378245,298.3,LENGTHUNIT["metre",1]]],'
            'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
            'ID["EPSG",4284]],'
            f'CONVERSION["{self.title}",'
            'METHOD["Transverse Mercator",ID["EPSG",9807]],'
            f'PARAMETER["Latitude of natural origin",{self.latitude_of_origin},'
            'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],'
            f'PARAMETER["Longitude of natural origin",{self.central_meridian},'
            'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],'
            'PARAMETER["Scale factor at natural origin",1,'
            'SCALEUNIT["unity",1],ID["EPSG",8805]],'
            f'PARAMETER["False easting",{self.false_easting},'
            'LENGTHUNIT["metre",1],ID["EPSG",8806]],'
            f'PARAMETER["False northing",{self.false_northing},'
            'LENGTHUNIT["metre",1],ID["EPSG",8807]]],'
            "CS[Cartesian,2],"
            'AXIS["northing (X)",north,ORDER[1],LENGTHUNIT["metre",1]],'
            'AXIS["easting (Y)",east,ORDER[2],LENGTHUNIT["metre",1]]]'
        )

    def crs(self):
        return _crs(self.wkt())

    def holds(self, bounds: Bounds) -> bool:
        """Whether coordinates in ``bounds`` (easting, northing) fit this zone."""
        min_x, min_y, max_x, max_y = bounds
        low, high = _NORTHING_RANGE
        return (
            abs(min_x - self.false_easting) <= _MAX_EASTING_OFFSET
            and abs(max_x - self.false_easting) <= _MAX_EASTING_OFFSET
            and low <= min_y <= max_y <= high
        )


@dataclass(frozen=True)
class WebMercator:
    """EPSG:3857, recognised only where it puts the coordinates within Russia."""

    key: str = "epsg3857"
    title: str = "WGS 84 / Pseudo-Mercator (EPSG:3857)"

    def crs(self):
        return _crs("EPSG:3857")

    def holds(self, bounds: Bounds) -> bool:
        if any(abs(value) > _WEB_MERCATOR_LIMIT for value in bounds):
            return False
        from pyproj import Transformer

        to_lon_lat = Transformer.from_crs(self.crs(), "EPSG:4326", always_xy=True)
        # Longitude grows with x and latitude with y: the corners bound the layer.
        min_lon, min_lat = to_lon_lat.transform(bounds[0], bounds[1])
        max_lon, max_lat = to_lon_lat.transform(bounds[2], bounds[3])
        west, south, east, north = _RUSSIA
        return (
            west <= min_lon <= max_lon <= east and south <= min_lat <= max_lat <= north
        )


# Zone 2 is published with its origin at 0.1° N; 0.1° of the Krassowsky meridian is
# the 11 057.628 m the other zones add to the false northing, so all three agree.
MSK47_ZONES: dict[int, LocalZone] = {
    1: LocalZone("msk47_zone1", "MSK47 zone 1", 0, 27.95, 1_250_000, -6_211_057.628),
    2: LocalZone("msk47_zone2", "MSK47 zone 2", 0.1, 30.95, 2_250_000, -6_200_000),
    3: LocalZone("msk47_zone3", "MSK47 zone 3", 0, 33.95, 3_250_000, -6_211_057.628),
}
WEB_MERCATOR = WebMercator()

GuessedCrs = LocalZone | WebMercator


@cache
def _crs(definition: str):
    from pyproj import CRS

    return CRS.from_user_input(definition)


def zone_by_name(*names: str | None) -> LocalZone | None:
    """The zone the first of ``names`` mentioning one names, e.g. «МСК-47 зона 2»."""
    for name in names:
        match = _MSK47_NAME.search(str(name or ""))
        if match:
            return MSK47_ZONES[int(match.group(1))]
    return None


def zone_by_coordinates(bounds: Bounds) -> LocalZone | None:
    """The only zone whose range holds ``bounds``; zones' ranges do not overlap."""
    return next((zone for zone in MSK47_ZONES.values() if zone.holds(bounds)), None)


def _named(names: list[str | None]) -> GuessedCrs | None:
    zone = zone_by_name(*names)
    if zone is not None:
        return zone
    if any(_WEB_MERCATOR_NAME.search(str(name or "")) for name in names):
        return WEB_MERCATOR
    return None


def guess_crs(names: Iterable[str | None], bounds: Bounds) -> GuessedCrs | None:
    """A CRS for coordinates without one: by name, else by the coordinates.

    A named system is trusted only when the coordinates fit it — a layer called
    «МСК-47 зона 1» holding zone-2 eastings is a mistake to report, not to place
    some 200 km off. МСК-47 northings (under 1 Mm) and Web Mercator ones within
    Russia (over 5 Mm) never meet, so the coordinates name at most one system.
    """
    named = _named(list(names))
    if named is not None:
        return named if named.holds(bounds) else None
    return zone_by_coordinates(bounds) or (
        WEB_MERCATOR if WEB_MERCATOR.holds(bounds) else None
    )
