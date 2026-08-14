from src.idu_mcp.tools_services.geometry_tools import GeometryTools


def _feature(properties, coordinates, feature_id=None):
    feature = {
        "type": "Feature",
        "properties": properties,
        "geometry": {"type": "Point", "coordinates": coordinates},
    }
    if feature_id is not None:
        feature["id"] = feature_id
    return feature


def test_restrictions_preserve_object_identity_and_explain_each_match():
    roads = {
        "type": "FeatureCollection",
        "features": [
            _feature(
                {
                    "physical_object_id": 10,
                    "object_geometry_id": 11,
                    "name": "Дорога А",
                },
                [37.62, 55.75],
                "road-feature",
            )
        ],
    }
    houses = {
        "type": "FeatureCollection",
        "features": [
            _feature(
                {
                    "physical_object_id": 20,
                    "object_geometry_id": 21,
                    "name": "Дом А",
                    "metadata": {"floors": 5},
                },
                [37.6205, 55.75],
            ),
            _feature(
                {
                    "physical_object_id": 30,
                    "object_geometry_id": 31,
                    "name": "Дом Б",
                },
                [38.0, 56.0],
            ),
        ],
    }
    tools = GeometryTools()
    buffers = tools.generate_geometry_buffers(
        {
            "Дороги": {
                "buffer_size": 100,
                "buffer_type": "round",
                "title": "Минимальное расстояние 100 м",
                "origin": "normgraph",
                "restriction_id": "restriction-1",
                "provenance": {"document_name": "СП test"},
            }
        },
        {"Дороги": roads},
    )

    result = tools.create_restrictions(
        {"Дороги": buffers["Дороги"], "Жилые дома": houses},
        generators=["Дороги"],
        objects=["Жилые дома"],
        restrictions={
            "Дороги": {
                "title": "Минимальное расстояние 100 м",
                "description": "Жильё должно находиться вне буфера.",
                "to": ["Жилые дома"],
                "origin": "normgraph",
                "restriction_id": "restriction-1",
                "provenance": {"document_name": "СП test"},
            }
        },
    )

    assert len(result["objects"]["features"]) == 1
    properties = result["objects"]["features"][0]["properties"]
    assert properties["physical_object_id"] == 20
    assert properties["object_geometry_id"] == 21
    assert properties["name"] == "Дом А"
    assert properties["metadata"] == {"floors": 5}
    assert properties["object_ref"] == {
        "id": "physical_object/20/geometry/21",
        "namespace": "physical_object",
        "entity_id": 20,
        "geometry_id": 21,
        "layer": "Жилые дома",
        "name": "Дом А",
    }
    evidence = properties["restriction_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["reason_code"] == "BUFFER_INTERSECTION"
    assert evidence[0]["boundary_policy"] == "inclusive"
    assert evidence[0]["distance_m"] == 100
    assert evidence[0]["restriction_id"] == "restriction-1"
    assert evidence[0]["generator_ref"]["name"] == "Дорога А"
    assert evidence[0]["provenance"]["document_name"] == "СП test"
