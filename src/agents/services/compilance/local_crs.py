"""Local coordinate systems of user files that carry no usable CRS.

Designs of the Leningrad oblast are drawn in МСК-47: Transverse Mercator on the
Pulkovo 1942 datum in three zones, none of them in the EPSG registry. Such files
reach us without a CRS PROJ understands — a GeoJSON names the zone only in prose
(``"МСК 47 зона 2"``), or names nothing at all. A zone is recognised by that name or,
failing it, by the coordinates themselves: the millions digit of the easting is the
zone number and the northing is counted from about 56° N.
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
# Farther than this from the central meridian a zone is not used.
_MAX_EASTING_OFFSET = 400_000
# The oblast lies within 58.4–61.3° N: northings 0.25–0.6 Mm, with a margin.
_NORTHING_RANGE = (150_000, 800_000)


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

    def holds(self, bounds: tuple[float, float, float, float]) -> bool:
        """Whether coordinates in ``bounds`` (easting, northing) fit this zone."""
        min_x, min_y, max_x, max_y = bounds
        low, high = _NORTHING_RANGE
        return (
            abs(min_x - self.false_easting) <= _MAX_EASTING_OFFSET
            and abs(max_x - self.false_easting) <= _MAX_EASTING_OFFSET
            and low <= min_y <= max_y <= high
        )


# Zone 2 is published with its origin at 0.1° N; 0.1° of the Krassowsky meridian is
# the 11 057.628 m the other zones add to the false northing, so all three agree.
MSK47_ZONES: dict[int, LocalZone] = {
    1: LocalZone("msk47_zone1", "MSK47 zone 1", 0, 27.95, 1_250_000, -6_211_057.628),
    2: LocalZone("msk47_zone2", "MSK47 zone 2", 0.1, 30.95, 2_250_000, -6_200_000),
    3: LocalZone("msk47_zone3", "MSK47 zone 3", 0, 33.95, 3_250_000, -6_211_057.628),
}


@cache
def _crs(wkt: str):
    from pyproj import CRS

    return CRS.from_wkt(wkt)


def zone_by_name(*names: str | None) -> LocalZone | None:
    """The zone the first of ``names`` mentioning one names, e.g. «МСК-47 зона 2»."""
    for name in names:
        match = _MSK47_NAME.search(str(name or ""))
        if match:
            return MSK47_ZONES[int(match.group(1))]
    return None


def zone_by_coordinates(
    bounds: tuple[float, float, float, float],
) -> LocalZone | None:
    """The only zone whose range holds ``bounds``; zones' ranges do not overlap."""
    return next((zone for zone in MSK47_ZONES.values() if zone.holds(bounds)), None)


def guess_zone(
    names: Iterable[str | None], bounds: tuple[float, float, float, float]
) -> LocalZone | None:
    """A zone for coordinates without a CRS: by name, else by the coordinates.

    A named zone is trusted only when the coordinates fit it — a layer called
    «МСК-47 зона 1» holding zone-2 eastings is a mistake to report, not to place
    some 200 km off.
    """
    named = zone_by_name(*names)
    if named is not None:
        return named if named.holds(bounds) else None
    return zone_by_coordinates(bounds)
