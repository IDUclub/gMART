"""Match each expert gold record to its reference GeoJSON files on Yandex.Disk.

Within a project folder the layers follow stable naming patterns (distance is the
strongest disambiguator across the several queries that share a project):

    <source>_<project>.geojson                     — source layer
    буфер_<dist>_<target>_<project>.geojson        — target buffer (zone)
    Зона_ограничения_<dist>_метров_<target>_…      — target zone (alt form)
    <source>_в_буфере_<dist>_<target>_<project>    — RESULT: source objects in zone

For scoring we need, per record:
    * ``result``  — source objects that fall in the zone  → object-selection P/R/F1
                    (compared with the model's ``objects`` layer)
    * ``zone``    — the target buffer polygons             → geometry IoU
                    (compared with the model's ``generators`` layer)

Matching is by normalised entity stems + the exact distance. Ambiguity within a
project is broken by the distance; unmatched roles are reported (not guessed).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from gold_parser import norm

REF_ROOT = Path("benchmarks/data/gold/reference")


# entity-stem lexicon: canonical gold entity -> filename stems that denote it
ENTITY_STEMS: dict[str, list[str]] = {
    "жилой дом": ["жил", "дом"],
    "водный объект": ["водн", "озёр", "озер", "рек", "пруд", "водоём", "водоем"],
    "школа": ["школ"],
    "детский сад": ["детсад", "детск"],
    "поликлиника": ["поликлин"],
    "больница": ["больниц"],
    "аптека": ["аптек"],
    "парк": ["парк", "сквер", "рекреац", "зелен"],
    "детская площадка": ["дет_площад", "детских_площад", "детские_площад", "площадок"],
    "спортивная площадка": ["спорт"],
    "супермаркет": ["магазин", "супермаркет", "продукт"],
    "кладбище": ["кладбищ"],
    "автозаправочная станция": ["азс", "автозаправ", "заправ"],
    "остановка наземного общественного транспорта": ["остановк"],
    "нежилое здание": ["нежил"],
    "промышленный объект": ["промышл"],
    "лес": ["лес"],
    "дорога": ["дорог"],
}


def _stems(entity: str | float | None) -> list[str]:
    if not isinstance(entity, str):
        return []
    n = norm(entity)
    for canon, st in ENTITY_STEMS.items():
        if canon in n or any(s in n for s in st):
            return st
    # fall back to the first content word stem
    m = re.findall(r"[а-яё]{4,}", n)
    return [m[0][:5]] if m else []


def _has_stem(fname_norm: str, stems: list[str]) -> bool:
    return any(s in fname_norm for s in stems)


def _dist_in(fname_norm: str, dist: int | None) -> bool:
    if not dist:
        return True  # no distance to disambiguate on
    # exact token match (avoid 50 matching 500): number not flanked by digits
    return re.search(rf"(?<!\d){dist}(?!\d)", fname_norm) is not None


def match_record(source: str, target: str, dist: int | None,
                 files: list[Path]) -> dict[str, Path | None]:
    """Return {'result': path|None, 'zone': path|None} for one gold record."""
    src, tgt = _stems(source), _stems(target)
    result = zone = None
    result_score = zone_score = -1

    for f in files:
        fn = norm(f.stem)
        d_ok = _dist_in(fn, dist)
        is_buf = ("буфер" in fn or "зона_ограничен" in fn or "зона ограничен" in fn
                  or "охранн" in fn or "сзз" in fn)
        in_zone = "в буфере" in fn or "в_буфере" in fn

        if in_zone:
            # RESULT layer: <source> в буфере <dist> <target>. Require the source
            # stem; distance/target are score boosters (disambiguate within a
            # project) rather than hard gates.
            if _has_stem(fn, src):
                score = 3 * d_ok + _has_stem(fn, tgt)
                if score > result_score:
                    result, result_score = f, score
        elif is_buf:
            # ZONE layer: буфер/зона <dist> <target>. Require the target stem.
            if _has_stem(fn, tgt):
                score = 3 * d_ok
                if score > zone_score:
                    zone, zone_score = f, score
    return {"result": result, "zone": zone}


def load_manifest(ref_root: Path = REF_ROOT) -> pd.DataFrame:
    return pd.read_csv(ref_root / "manifest.csv")


def files_for(local_dir: str, ref_root: Path = REF_ROOT) -> list[Path]:
    d = ref_root / local_dir
    return list(d.rglob("*.geojson")) if d.exists() else []


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from gold_parser import load_gold

    gold = load_gold("benchmarks/data/gold/exp_data.csv")
    man = load_manifest()
    # base_index -> first downloaded local_dir
    ok = man[man["status"].astype(str).str.startswith(("ok", "cached"))]
    dir_by_base = ok.groupby("base_index")["local_dir"].first().to_dict()

    n_cov = n_result = n_zone = n_both = 0
    for i, g in enumerate(gold):
        if i not in dir_by_base:
            continue
        n_cov += 1
        m = match_record(g.source_entity, g.target_entity, g.distance_m,
                         files_for(dir_by_base[i]))
        n_result += m["result"] is not None
        n_zone += m["zone"] is not None
        n_both += m["result"] is not None and m["zone"] is not None
    print(f"gold records with a downloaded project folder: {n_cov}")
    print(f"  matched RESULT (objects-in-zone) file: {n_result}")
    print(f"  matched ZONE (buffer) file:            {n_zone}")
    print(f"  matched BOTH:                          {n_both}")
