"""Layer loading in geometry_eval — both harnesses' record shapes.

The HTTP harness inlined every feature collection into the record, which is what
made a full results.jsonl run to gigabytes. The in-process runner writes each
layer to its own GeoJSON and records the path. Geometry scoring has to read both,
because the old runs are still the operational screening for six of the models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "harness"))

import geometry_eval as ge  # noqa: E402


def _feature(x: float = 30.0) -> dict:
    return {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Point", "coordinates": [x, 60.0]},
    }


def _write_layer(path: Path, features: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def test_inline_layers_are_read_as_before():
    record = {
        "layers": [
            {
                "name": "objects",
                "feature_collection": {
                    "type": "FeatureCollection",
                    "features": [_feature(), _feature(31.0)],
                },
            }
        ]
    }

    assert len(ge.layers_by_name(record)["objects"]) == 2


def test_layer_files_are_read_from_disk(tmp_path):
    _write_layer(tmp_path / "layers" / "00003" / "objects.geojson", [_feature()])
    record = {"layer_files": {"objects": "layers/00003/objects.geojson"}}

    produced = ge.layers_by_name(record, results_root=tmp_path)

    assert len(produced["objects"]) == 1


def test_relative_layer_paths_resolve_against_the_results_file(tmp_path):
    """A results directory has to stay movable between machines."""

    run_dir = tmp_path / "gemma3_12b" / "base--local"
    _write_layer(run_dir / "layers" / "00000" / "generators.geojson", [_feature()])
    record = {"layer_files": {"generators": "layers/00000/generators.geojson"}}

    assert ge.layers_by_name(record, results_root=run_dir)["generators"]


def test_absolute_layer_paths_are_used_as_given(tmp_path):
    path = tmp_path / "elsewhere" / "objects.geojson"
    _write_layer(path, [_feature(), _feature(31.0)])
    record = {"layer_files": {"objects": str(path)}}

    assert len(ge.layers_by_name(record, results_root=tmp_path / "unrelated")) == 1


def test_layer_names_are_normalised_like_the_inline_path(tmp_path):
    """geometry_eval matches produced layers by normalised name."""

    _write_layer(tmp_path / "obj.geojson", [_feature()])
    record = {"layer_files": {"Объекты В Зоне Ограничений": "obj.geojson"}}

    produced = ge.layers_by_name(record, results_root=tmp_path)

    assert "объекты в зоне ограничений" in produced


def test_an_unreadable_layer_is_skipped_not_fatal(tmp_path, capsys):
    """One missing file must not take down a whole scoring run."""

    _write_layer(tmp_path / "good.geojson", [_feature()])
    record = {
        "layer_files": {
            "objects": "good.geojson",
            "generators": "gone.geojson",
        }
    }

    produced = ge.layers_by_name(record, results_root=tmp_path)

    assert "objects" in produced
    assert "generators" not in produced
    assert "cannot read layer" in capsys.readouterr().out


def test_inline_layers_win_over_a_path_for_the_same_name(tmp_path):
    """A record carrying both is the HTTP shape; reading the file would be wasted."""

    _write_layer(tmp_path / "objects.geojson", [_feature(), _feature(31.0)])
    record = {
        "layers": [
            {
                "name": "objects",
                "feature_collection": {
                    "type": "FeatureCollection",
                    "features": [_feature()],
                },
            }
        ],
        "layer_files": {"objects": "objects.geojson"},
    }

    assert len(ge.layers_by_name(record, results_root=tmp_path)["objects"]) == 1


def test_a_record_with_neither_yields_nothing():
    assert ge.layers_by_name({}) == {}
