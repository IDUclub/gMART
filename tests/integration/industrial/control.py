"""Small, immutable Urban inputs. No computed provision/compliance responses."""

from copy import deepcopy

VERSION = "industrial-control-2026-09-v1"
# id: name, population, school capacity, kindergarten capacity, school longitude
SCENARIOS = {
    91000: ("Базовый", 1000, 100, 50, 30.004),
    91001: ("Промзона", 12000, 800, 400, 30.004),
    91002: ("Предынвестиционный", 4500, 300, 150, 30.004),
    91003: ("Социальная инфраструктура до", 25000, 1800, 900, 30.004),
    91004: ("Социальная инфраструктура после", 25000, 2500, 1250, 30.006),
    91005: ("До замечаний", 12000, 800, 400, 30.004),
    91006: ("После замечаний", 12000, 1300, 650, 30.006),
    91007: ("А — жилой", 8000, 600, 300, 30.004),
    91008: ("Б — сбалансированный", 8000, 800, 400, 30.006),
    91009: ("В — деловой", 8000, 500, 250, 30.004),
}
SERVICE_TYPES = [
    {"service_type_id": 22, "name": "Школа", "capacity_modeled": 100},
    {"service_type_id": 21, "name": "Детский сад", "capacity_modeled": 50},
]
PHYSICAL_TYPES = [
    {"physical_object_type_id": i, "name": name}
    for i, name in [
        (4, "Жилой дом"),
        (5, "Здание школы"),
        (6, "Здание детского сада"),
        (7, "Открытая автомобильная стоянка"),
        (8, "Парк"),
    ]
]


def rectangle(x, y, width=0.0002, height=0.0001):
    return {
        "type": "Polygon",
        "coordinates": [
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height], [x, y]]
        ],
    }


def feature(geometry, properties, identifier=None):
    return {
        "type": "Feature",
        "id": identifier,
        "geometry": geometry,
        "properties": properties,
    }


def collection(features, sid):
    return {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "complete": True,
            "truncated": False,
            "revision": f"{VERSION}:{sid}",
            "scenario_id": sid,
        },
    }


def scenario(sid):
    name, population, *_ = SCENARIOS[sid]
    return {
        "scenario_id": sid,
        "name": name,
        "is_based": sid == 91000,
        "project": {"project_id": 910, "name": "Контрольный проект"},
        "properties": {"source_version": VERSION, "population": population},
        "updated_at": "2026-09-01T00:00:00Z",
    }


def project():
    return {
        "project_id": 910,
        "name": "Контрольный проект",
        "territory": {"id": 910},
        "properties": {"context": [911]},
        "base_scenario": {"id": 91000, "scenario_id": 91000},
    }


def entities(sid, domain, type_id=None, context=False):
    _, _, school, kindergarten, school_x = SCENARIOS[sid]
    x, y = (30.05, 60.05) if context else (30.001, 60.001)
    scope = {"scenario_id": sid, "source_version": f"{VERSION}:{sid}"}
    if domain == "service":
        positions = (
            [(x + 0.001, y), (x + 0.002, y)]
            if context
            else [(school_x, y), (30.007, y)]
        )
        capacities = (100, 50) if context else (school, kindergarten)
        result = []
        for i, (t, capacity, pos) in enumerate(
            zip(SERVICE_TYPES, capacities, positions)
        ):
            identifier = 9900 + i if context else sid * 10 + i
            result.append(
                feature(
                    rectangle(*pos),
                    {
                        **scope,
                        "service_id": identifier,
                        "service_type": {"id": t["service_type_id"], **t},
                        "name": t["name"],
                        "capacity": capacity,
                    },
                    identifier,
                )
            )
    else:
        positions = [(x, y), (school_x, y), (30.007, y), (30.0044, y), (30.009, y)]
        result = []
        for i, (t, pos) in enumerate(zip(PHYSICAL_TYPES, positions)):
            if context and i:
                break
            identifier = 9800 + i if context else sid * 10 + 100 + i
            result.append(
                feature(
                    rectangle(*pos),
                    {
                        **scope,
                        "physical_object_id": identifier,
                        "physical_object_type": {
                            "id": t["physical_object_type_id"],
                            **t,
                        },
                        "name": t["name"],
                        "building": {"floors": 5},
                        "properties": {"preserve": i in (0, 4)},
                    },
                    identifier,
                )
            )
    if type_id is not None:
        result = [
            f
            for f in result
            if f["properties"][f"{domain}_type"][f"{domain}_type_id"] == int(type_id)
        ]
    return collection(result, sid)


def nested_geometries(sid, params, context=False):
    domain = "service" if params.get("service_type_id") else "physical_object"
    data = entities(sid, domain, params.get(f"{domain}_type_id"), context)
    for f in data["features"]:
        record = deepcopy(f["properties"])
        f["properties"] = {
            "object_geometry_id": f["id"],
            "territory": {"id": 911 if context else 910},
            "address": "Синтетический контроль",
            "osm_id": None,
            "physical_objects": [record] if domain == "physical_object" else [],
            "services": [record] if domain == "service" else [],
        }
    return data


def indicators(sid):
    # Living floor area is a supplied planning indicator, not a computed result.
    values = [
        (1, "Численность населения", SCENARIOS[sid][1], "человек"),
        (2, "Площадь жилых помещений", 240000, "м2"),
    ]
    return [
        {
            "scenario": {"id": sid, "name": SCENARIOS[sid][0]},
            "indicator": {
                "indicator_id": i,
                "name_full": name,
                "measurement_unit": {"name": unit},
            },
            "value": value,
            "source": VERSION,
            "information_source": VERSION,
            "year": 2026,
        }
        for i, name, value, unit in values
    ]


def normatives():
    return [
        {
            "normative_id": 9100 + i,
            "service_type": {"id": t["service_type_id"], "name": t["name"]},
            "year": 2026,
            "services_capacity_per_1000_normative": rate,
            "services_per_1000_normative": None,
            "radius_availability_meters": 1000,
            "time_availability_minutes": None,
            "source": VERSION,
        }
        for i, (t, rate) in enumerate(zip(SERVICE_TYPES, (100, 50)))
    ]
