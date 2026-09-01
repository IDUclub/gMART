#!/usr/bin/env python3
"""Object-selection and geometry scoring vs the expert reference GeoJSON.

This is the authoritative correctness layer (the NL-answer count check in
semantic_eval.py is only a weak cross-check). For every gold record it compares
the layers a model produced against the expert reference layers and reports:

  * object-selection precision / recall / F1 — did the model return the *right*
    objects (matched by geometry: centroid within TOL metres);
  * geometry IoU — area(prod ∩ ref) / area(prod ∪ ref) of the buffer/zone layer.

Both are computed in a local UTM zone derived from the data, not in web mercator:
at ~60°N mercator inflates distances by 2x, which would silently loosen the
centroid tolerance and distort areas.

Reference layout (as delivered by fetch_reference.py):

    data/gold/reference/manifest.csv          base_index, scenario_id, link, local_dir, status
    data/gold/reference/<local_dir>/*.geojson expert layers for that gold row

The manifest anchors each folder to the gold rows it was linked from. Filenames
are free-form Russian, so a file is assigned to a row only on an exact anchor
(`--match`):

  * "unique" (default): the folder is linked to exactly one gold row and holds
    exactly one layer for the role — nothing is inferred;
  * "exact": additionally accepts a file that states the gold distance AND names
    the gold entity in full, and only when exactly one file in the folder does so.

Everything else is left unscored. Approximate matching (stem overlap, nearest
distance) is deliberately NOT available: scoring a model against a plausible-looking
but wrong reference layer produces a number that cannot be defended.

Leading row numbers ("67_буфер_100_водоем") look like the source spreadsheet's row
but are NOT usable as an anchor — measured against the manifest their offset is
inconsistent (3 and 4 across the folders where it can be checked).

The delivered layers are of three kinds, each compared against a different
produced layer: "<X> в зоне/в буфере <Y>" is the filtered result (vs `objects`),
"буфер <N> <X>" / "зона ограничения ..." is the zone (vs `generators`), and a
plain "<X>" file is the full entity layer (vs the produced layer named <X>).

Report the "unique" subset when the matching itself must be above suspicion.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.ops import unary_union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))
from gold_parser import load_gold, norm  # noqa: E402
from shapely.geometry import shape as _shape  # noqa: E402

CENTROID_TOL_M = 5.0  # two objects are "the same" within this many metres
ROW_OFFSET = 2  # base_index -> the row number used in the reference filenames


# --------------------------------------------------------------------------- #
# Reference loading
# --------------------------------------------------------------------------- #
@dataclass
class RefFile:
    path: Path
    role: str  # "objects" | "buffer"
    row_no: int | None  # leading number in the filename, if any
    distance: int | None  # buffer distance mentioned in the filename
    stems: set[str]


def _stems(text: str) -> set[str]:
    out = set()
    for w in re.split(r"[^а-яa-z0-9]+", norm(text)):
        if len(w) > 3 and not w.isdigit():
            out.add(w[: max(3, len(w) - 4)])
    return out


def _parse_ref_name(path: Path) -> RefFile:
    stem = path.stem
    n = norm(stem)
    lead = re.match(r"^\s*(\d+(?:\s*,\s*\d+)*)\s*[_\-. ]", stem)
    row_no = int(lead.group(1).split(",")[0]) if lead else None
    # The delivered files are of three kinds, and each compares against a
    # different produced layer (verified on the data: "55_56_жилые дома" has 5
    # features and the model's "жилой дом" layer has 5; "58_здания_в_зоне_
    # ограничений" has 400 against the model's filtered `objects` layer of 387):
    #   * "<X> в зоне/в буфере <Y>" — the FILTERED result   -> produced `objects`
    #   * "буфер <N> <X>" / "зона ограничения <N> <X>" — the ZONE -> `generators`
    #   * a plain "<X>" layer — the full entity layer      -> the named layer <X>
    if re.search(r"в[_ ]?(буфер|зоне)", n):
        role = "objects"
    elif re.search(r"(^|_)буфер", n.replace(str(row_no or ""), "", 1)) or re.search(
        r"зон\w*[_ ]ограничени", n
    ):
        role = "buffer"
    else:
        role = "entity"
    # Distances sit anywhere in these names — "буфер_300_X", "буфер_остановки_100",
    # "Зона_ограничения_200_метров_X", "буферы300_X" — so collect every standalone
    # 2-5 digit number instead of anchoring to a keyword (a keyword-anchored regex
    # silently misread "буфер_остановки_100" as 0). The leading row number, if any,
    # is not a distance.
    body = (
        n[len(str(row_no)) :] if row_no is not None and n.startswith(str(row_no)) else n
    )
    numbers = [int(x) for x in re.findall(r"(?<!\d)(\d{2,5})(?!\d)", body)]
    near_keyword = [
        int(x)
        for x in re.findall(
            r"(?:буфер|зон)[а-я]*[_ ]*(?:[а-я]+[_ ]*)*?(\d{2,5})(?!\d)", body
        )
    ] + [
        int(x) for x in re.findall(r"(?<!\d)(\d{2,5})(?!\d)[_ ]*м(?:етр)?[а-я]*", body)
    ]
    dist = (near_keyword or numbers or [None])[0]
    return RefFile(path, role, row_no, dist, _stems(stem))


def build_reference_index(
    ref_dir: Path, gold: list, match: str = "unique"
) -> dict[int, dict[str, tuple[Path, str]]]:
    """base_index -> {role: (geojson path, match kind)}.

    Only exact anchors are accepted, see the module docstring for `match`:
    "unique" (folder belongs to one gold row and holds one layer for the role) and,
    with match="exact", a file that states the gold distance and names the gold
    entity in full. Rows without such an anchor are not scored at all.
    """

    manifest = pd.read_csv(ref_dir / "manifest.csv")
    by_dir: dict[str, list[RefFile]] = {}
    for d in {str(x) for x in manifest["local_dir"].dropna()}:
        folder = ref_dir / d
        if folder.is_dir():
            by_dir[d] = [_parse_ref_name(p) for p in sorted(folder.rglob("*.geojson"))]

    rows_per_dir = manifest.groupby("local_dir")["base_index"].nunique().to_dict()

    index: dict[int, dict[str, Path]] = {}
    skipped = 0
    for _, row in manifest.iterrows():
        d = str(row.get("local_dir") or "")
        base = int(row["base_index"])
        files = by_dir.get(d) or []
        if not files or base >= len(gold):
            continue
        g = gold[base]

        for role in ("objects", "buffer", "entity"):
            cands = [f for f in files if f.role == role]
            if not cands:
                continue
            # Unambiguous case: this folder belongs to exactly one gold row and
            # holds exactly one file for the role.
            if len(cands) == 1 and rows_per_dir.get(d, 0) == 1:
                index.setdefault(base, {})[role] = (cands[0].path, "unique")
                continue
            if match == "unique":
                # Ambiguous folder: no exact anchor exists, so the row is not scored.
                skipped += 1
                continue

            # "exact": the file must state the gold distance AND name the gold
            # entity in full (not merely share a stem with it). The row numbers
            # some filenames carry are NOT usable as an anchor — measured against
            # the manifest their offset is inconsistent (3 vs 4).
            def exact(f: RefFile) -> bool:
                if g.distance_m is None or f.distance != g.distance_m:
                    return False
                names = [n for n in (g.source_entity, g.target_entity) if n]
                fname = norm(f.path.stem)
                return any(
                    all(w in fname for w in norm(nm).split() if len(w) > 3)
                    for nm in names
                )

            hits = [f for f in cands if exact(f)]
            if len(hits) != 1:  # zero, or two files equally entitled
                skipped += 1
                continue
            index.setdefault(base, {})[role] = (hits[0].path, "exact")

    if skipped:
        print(f"  {skipped} candidate layers left unmatched under match={match!r}")
    return index


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _utm_crs(gdf: gpd.GeoDataFrame) -> int:
    """Local UTM zone for the data's centroid (metric, distortion-free enough)."""
    c = gdf.to_crs(4326).union_all().centroid
    zone = int((c.x + 180) // 6) + 1
    return (32600 if c.y >= 0 else 32700) + zone


def load_territories(path: Path, gold_csv: Path) -> dict[int, object]:
    """base_index -> project-territory polygon (WGS84), for clipping.

    The expert layers were exported over a wider area than the scenario the
    pipeline queries, so both sides are clipped to the project boundary before
    scoring; without it the metric compares different territories.
    """
    if not path.exists():
        return {}
    geoms = json.loads(path.read_text(encoding="utf-8"))
    gold = pd.read_csv(gold_csv, sep=";", engine="python")
    ids = pd.to_numeric(gold["ID проекта"], errors="coerce")
    by_scenario = (
        pd.DataFrame({"sid": gold["scenario_id"], "pid": ids})
        .dropna()
        .drop_duplicates("sid")
        .set_index("sid")["pid"]
        .to_dict()
    )
    ids = ids.fillna(gold["scenario_id"].map(by_scenario))
    out: dict[int, object] = {}
    for base, pid in enumerate(ids):
        if pd.isna(pid):
            continue
        g = geoms.get(str(int(pid)))
        if g:
            out[base] = _shape(g)
    return out


def _frame(features: list[dict]) -> gpd.GeoDataFrame | None:
    if not features:
        return None
    geoms = [shape(f["geometry"]) for f in features if f.get("geometry")]
    geoms = [g for g in geoms if not g.is_empty]
    if not geoms:
        return None
    return gpd.GeoDataFrame(geometry=geoms, crs=4326)


def object_selection_prf(
    pred: gpd.GeoDataFrame | None, ref: gpd.GeoDataFrame | None
) -> dict:
    """Greedy centroid-nearest matching within CENTROID_TOL_M."""
    n_pred = 0 if pred is None else len(pred)
    n_ref = 0 if ref is None else len(ref)
    if n_pred == 0 or n_ref == 0:
        p = 1.0 if n_pred == 0 and n_ref == 0 else 0.0
        return {
            "precision": p,
            "recall": p,
            "f1": p,
            "n_pred": n_pred,
            "n_ref": n_ref,
            "tp": 0,
        }
    pc = list(pred.geometry.centroid)
    rc = list(ref.geometry.centroid)
    used: set[int] = set()
    tp = 0
    for a in pc:
        best, bj = CENTROID_TOL_M, -1
        for j, b in enumerate(rc):
            if j in used:
                continue
            d = a.distance(b)
            if d <= best:
                best, bj = d, j
        if bj >= 0:
            used.add(bj)
            tp += 1
    precision = tp / n_pred
    recall = tp / n_ref
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": n_pred,
        "n_ref": n_ref,
        "tp": tp,
    }


def geometry_iou(
    pred: gpd.GeoDataFrame | None, ref: gpd.GeoDataFrame | None
) -> float | None:
    if pred is None or ref is None:
        return None
    pu = unary_union(list(pred.geometry))
    ru = unary_union(list(ref.geometry))
    union = pu.union(ru).area
    return pu.intersection(ru).area / union if union else None


def layers_by_name(
    rec: dict, results_root: Path | None = None
) -> dict[str, list[dict]]:
    """The produced layers of one record, whichever harness wrote it.

    The HTTP harness inlined every feature collection into the record, which is
    what made a full results.jsonl run to gigabytes. The in-process runner writes
    each layer to its own GeoJSON under ``--save-layers`` and records the path, so
    a record stays small and a layer is read only when it is actually scored.
    Paths are resolved relative to the results file when they are not absolute,
    so a results directory stays movable.
    """

    out: dict[str, list[dict]] = {}
    for layer in rec.get("layers") or []:
        fc = layer.get("feature_collection") or {}
        out[norm(layer.get("name"))] = fc.get("features", []) or []
    for name, raw_path in (rec.get("layer_files") or {}).items():
        key = norm(name)
        if key in out:
            continue
        path = Path(raw_path)
        if not path.is_absolute() and results_root is not None:
            path = results_root / path
        try:
            with path.open(encoding="utf-8") as handle:
                fc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  warning: cannot read layer {name!r} at {path}: {exc}")
            continue
        out[key] = fc.get("features", []) or []
    return out


# produced layer names (post-fix display names OR legacy keys) -> reference role.
# "entity" has no fixed name — it is the produced layer whose name matches the
# reference file's entity, resolved per record in `_produced_for`.
PRODUCED_ALIASES = {
    "objects": ["objects", "объекты в зоне ограничений"],
    "buffer": ["generators", "источники ограничений"],
}
RESERVED_LAYERS = {
    "objects",
    "generators",
    "объекты в зоне ограничений",
    "источники ограничений",
}


def _produced_for(
    role: str, produced: dict[str, list[dict]], ref_path: Path
) -> list[dict]:
    """The produced layer that the reference file should be compared against."""
    if role in PRODUCED_ALIASES:
        for alias in PRODUCED_ALIASES[role]:
            if alias in produced:
                return produced[alias]
        return []
    want = _stems(ref_path.stem)
    best, best_overlap = [], 0
    for name, feats in produced.items():
        if name in RESERVED_LAYERS:
            continue
        overlap = len(_stems(name) & want)
        if overlap > best_overlap:
            best, best_overlap = feats, overlap
    return best


def evaluate(
    results_path: Path,
    base_of: dict[int, int],
    ref_index: dict[int, dict[str, Path]],
    territories: dict[int, object] | None = None,
) -> pd.DataFrame:
    cache: dict[Path, gpd.GeoDataFrame] = {}
    rows = []
    for line in results_path.open(encoding="utf-8"):
        rec = json.loads(line)
        base = base_of.get(rec.get("idx"))
        if base is None or base not in ref_index:
            continue
        produced = layers_by_name(rec, results_root=results_path.parent)
        row = {
            "idx": rec.get("idx"),
            "base_index": base,
            "scenario_id": rec.get("scenario_id"),
            "prompt": str(rec.get("prompt"))[:60],
        }
        scored = False
        for role in ("objects", "buffer", "entity"):
            entry = ref_index[base].get(role)
            if entry is None:
                continue
            path, kind = entry
            row[f"{role}_match"] = kind
            if path not in cache:
                try:
                    cache[path] = gpd.read_file(path).to_crs(4326)
                except Exception as e:  # noqa: BLE001
                    print(f"  skip {path.name}: {e}")
                    cache[path] = gpd.GeoDataFrame(geometry=[], crs=4326)
            ref_gdf = cache[path]
            if ref_gdf.empty:
                continue
            feats = _produced_for(role, produced, path)
            pred_gdf = _frame(feats)
            crs = _utm_crs(ref_gdf)
            ref_m = ref_gdf.to_crs(crs)
            pred_m = pred_gdf.to_crs(crs) if pred_gdf is not None else None
            territory = (territories or {}).get(base)
            if territory is not None:
                boundary = gpd.GeoSeries([territory], crs=4326).to_crs(crs).iloc[0]
                row["clipped"] = True
                ref_m = gpd.clip(ref_m, boundary)
                if pred_m is not None:
                    pred_m = gpd.clip(pred_m, boundary)
                if ref_m.empty:
                    # the reference has nothing inside the project territory at all
                    row[f"{role}_ref_outside_territory"] = True
                    continue
            prf = object_selection_prf(pred_m, ref_m)
            row[f"{role}_f1"] = round(prf["f1"], 3)
            row[f"{role}_precision"] = round(prf["precision"], 3)
            row[f"{role}_recall"] = round(prf["recall"], 3)
            row[f"{role}_n_pred"] = prf["n_pred"]
            row[f"{role}_n_ref"] = prf["n_ref"]
            iou = geometry_iou(pred_m, ref_m)
            row[f"{role}_iou"] = round(iou, 3) if iou is not None else None
            scored = True
        if scored:
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results",
        required=True,
        help="results.jsonl of one run. Either harness: the HTTP one "
        "inlines the layers, the in-process one records paths to "
        "GeoJSON written under --save-layers (resolved relative "
        "to this file).",
    )
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--dataset", default="benchmarks/data/gold/expanded_goldfirst.csv")
    ap.add_argument("--reference", default="benchmarks/data/gold/reference")
    ap.add_argument("--out", default="benchmarks/out/geometry_report.csv")
    ap.add_argument(
        "--match",
        choices=["unique", "exact"],
        default="unique",
        help="'unique': score only rows whose reference layer is "
        "unambiguous (folder belongs to one gold row and holds "
        "one layer for the role) — no interpretation at all. "
        "'exact': additionally accept a file that states the gold "
        "distance and names the gold entity in full.",
    )
    ap.add_argument(
        "--clip",
        action="store_true",
        help="clip both the reference and the produced layers to the "
        "project territory before scoring (needs territories.json "
        "from fetch_territories.py)",
    )
    ap.add_argument("--territories", default="benchmarks/data/gold/territories.json")
    ap.add_argument(
        "--base-only",
        action="store_true",
        help="score only the 202 base gold rows, not the paraphrases",
    )
    args = ap.parse_args()

    ref_dir = Path(args.reference)
    if not (ref_dir / "manifest.csv").exists():
        sys.exit(
            f"no manifest at {ref_dir}/manifest.csv — run fetch_reference.py first."
        )

    gold = load_gold(args.gold)
    ref_index = build_reference_index(ref_dir, gold, match=args.match)
    counts = {
        r: sum(1 for v in ref_index.values() if r in v)
        for r in ("objects", "buffer", "entity")
    }
    print(
        f"gold rows with a reference layer: {len(ref_index)}/{len(gold)} "
        f"(filtered-objects {counts['objects']}, zone {counts['buffer']}, "
        f"entity {counts['entity']})"
    )

    df = pd.read_csv(args.dataset, sep=None, engine="python")
    # Generated/paraphrase datasets carry an explicit base_index. A direct run
    # on the expert set is already in base order, so identity is the correct map.
    if "base_index" in df.columns:
        base_of = {i: int(b) for i, b in enumerate(df["base_index"])}
    else:
        base_of = {i: i for i in range(len(df))}
    if args.base_only:
        base_of = {i: b for i, b in base_of.items() if i < len(gold)}

    territories = (
        load_territories(Path(args.territories), Path(args.gold)) if args.clip else {}
    )
    if args.clip:
        print(f"project territories available for {len(territories)} gold rows")
    out = evaluate(Path(args.results), base_of, ref_index, territories)
    if out.empty:
        print(
            "no records scored — the results file carries neither inline layers "
            "nor layer_files paths (the in-process runner needs --save-layers)."
        )
        return
    out.to_csv(args.out, index=False)
    print(f"scored {len(out)} records -> {args.out}")
    for role in ("objects", "buffer", "entity"):
        cols = [c for c in out.columns if c.startswith(role)]
        if not cols:
            continue
        sub = out[cols].dropna(how="all")
        print(f"\n{role}: n={len(sub)}")
        print(sub.mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
