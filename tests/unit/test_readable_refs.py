"""Agent answers name documents, objects and steps; system identifiers stay out."""

from __future__ import annotations

import json
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from src.agents.services.compilance.compliance_report import build_compliance_report
from src.agents.services.compilance.compliance_result_harness import (
    ComplianceResultHarness,
)
from src.agents.services.dvd.dvd_context import DvdContextBuilder
from src.agents.services.readable_refs import object_label
from src.agents.services.restriction.restriction_context import (
    RestrictionContextBuilder,
)
from src.agents.services.scenario_data.scenario_data_linear import (
    ScenarioDataLinearWorkflow,
)
from src.idu_mcp.tools_services.geometry_tools import GeometryTools

UNNAMED = {
    "id": "physical_object/1722382/geometry/1722379",
    "namespace": "physical_object",
    "entity_id": 1722382,
    "geometry_id": 1722379,
    "layer": "Жилой дом",
    "name": "(Безымянный физический объект)",
}
SCHOOL = {
    "id": "service/603427/geometry/600184",
    "entity_id": 603427,
    "layer": "Детский сад",
    "name": "Детский сад № 2",
}


@pytest.mark.parametrize(
    ("ref", "label"),
    [
        (SCHOOL, "Детский сад № 2"),
        (UNNAMED, "Жилой дом (без названия)"),
        ({**UNNAMED, "name": "Жилой дом #1722382"}, "Жилой дом (без названия)"),
        ({**UNNAMED, "name": UNNAMED["id"]}, "Жилой дом (без названия)"),
        ({"id": "feature/3"}, "Объект без названия"),
        (None, "Объект без названия"),
    ],
)
def test_object_label_never_falls_back_to_an_identifier(ref, label):
    assert object_label(ref) == label


def _violated_result():
    source = {
        "restriction_id": "0a9f99117acef747e9c3addad35dc4ba5fd323ae",
        "document_name": "СП 42.13330.2016",
        "clause_number": "7.1",
        "extraction_text": "Расстояние не менее 15 м",
    }
    return {
        "restriction_id": source["restriction_id"],
        "template": "distance_from_source",
        "template_version": 1,
        "verification_status": "complete",
        "compliance_status": "violated",
        "coverage": {"applicable_objects": 1, "checked_objects": 1},
        "summary": {"violated_objects": 1, "passed_objects": 0},
        "source": {**source, "equivalent_sources": [source]},
        "evidence": [
            {
                "object_ref": UNNAMED,
                "generator_refs": [SCHOOL],
                "measured_value": 1,
                "unit": "count",
                "violated": True,
            }
        ],
    }


def test_compliance_report_names_objects_without_their_keys():
    report = build_compliance_report(
        {"results": [_violated_result()], "skipped_without_plan": 0}
    )

    assert "| 1 | Жилой дом (без названия) |" in report
    assert "Детский сад № 2" in report
    assert "physical_object/" not in report and "service/" not in report
    assert "0a9f99117" not in report


def test_compliance_follow_up_context_carries_names_not_keys():
    summary = {
        "request_id": "d2c2c0c7-8737-4ad2-8252-c5830923194b",
        "total_norms": 1,
        "violated_norms": 1,
        "results": [_violated_result()],
    }
    compact = ComplianceResultHarness._compact_summary(summary)
    text = json.dumps(compact, ensure_ascii=False)

    assert compact["results"][0]["evidence"][0]["object_ref"] == (
        "Жилой дом (без названия)"
    )
    assert "physical_object/" not in text and "d2c2c0c7" not in text
    assert "0a9f99117" not in text


@pytest.mark.asyncio
async def test_restriction_answer_context_has_no_object_or_restriction_ids():
    objects = gpd.GeoDataFrame(
        {
            "restriction_name": ["Зона 15 м"],
            "restriction_description": ["Пересечение"],
            "object_ref": [UNNAMED],
            "source_layer": ["Жилой дом"],
            "restriction_evidence": [
                [
                    {
                        "reason": "Геометрия объекта пересекает буфер",
                        "title": "Зона 15 м",
                        "distance_m": 15.0,
                        "source_layer": "Детский сад",
                        "generator_ref": SCHOOL,
                        "restriction_id": "0a9f99117acef747e9c3addad35dc4ba5fd323ae",
                        "provenance": {
                            "document_id": "doc-uuid",
                            "document_name": "СП 42.13330.2016",
                            "clause_id": "clause-uuid",
                            "clause_number": "7.1",
                        },
                    }
                ]
            ],
        },
        geometry=[Point(0, 0)],
        crs=3857,
    )

    context = await RestrictionContextBuilder.generate_objects_summary(objects)
    detail = json.loads(context)["affected_objects"][0]

    assert detail["object_name"] == "Жилой дом (без названия)"
    assert detail["reasons"][0]["источник"] == "Детский сад № 2"
    assert detail["reasons"][0]["документ"] == "СП 42.13330.2016, п. 7.1"
    for key in ("physical_object/", "service/", "0a9f99117", "uuid"):
        assert key not in context


def test_dvd_context_does_not_show_fragment_ids():
    context = DvdContextBuilder().build_context(
        [
            {
                "id": "c5e06b82-a763-456c-9339-d5f82f871172",
                "doc_id": "c68ed292-457f-4b0e-bd4f-6c13edc4df88",
                "name": "Градостроительный кодекс",
                "numbering": "12",
                "text": "Норма.",
            }
        ]
    )

    assert "c5e06b82" not in context and "node_id" not in context
    assert "c68ed292" not in context


def test_idu_mcp_names_unnamed_objects_by_address_or_layer():
    rows = [
        {"physical_object_id": 7, "name": "(Безымянный физический объект)"},
        {
            "physical_object_id": 8,
            "name": "(Безымянный физический объект)",
            "address": "ул. Лесная, 5",
        },
    ]
    names = [
        GeometryTools._object_ref(pd.Series(row), "Жилой дом", index)["name"]
        for index, row in enumerate(rows)
    ]

    assert names == ["Жилой дом (без названия)", "ул. Лесная, 5"]


def test_scenario_data_reasons_name_steps_and_requirements_by_purpose():
    step = SimpleNamespace(step_id="clinics_map", purpose="Получить слой поликлиник")
    requirement = SimpleNamespace(
        requirement_id="clinics", description="Поликлиники сценария на карте"
    )
    reasons = ScenarioDataLinearWorkflow._plan_completion_reasons(
        SimpleNamespace(requirements=[requirement]),
        SimpleNamespace(steps=[step]),
        SimpleNamespace(completed_step_ids=set()),
        set(),
    )

    assert reasons == [
        "не завершены шаги плана: «Получить слой поликлиник»",
        "не закрыты требования: Поликлиники сценария на карте",
    ]
