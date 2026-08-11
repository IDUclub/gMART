"""Parse the 202 expert gold records (exp_data.csv) into structured fields.

For every expert query we derive the ground truth the reviewer asks us to score
against (intent, source/target entity, distance parameter, expected layers,
expected object count). Extraction is deterministic and catalog-grounded; every
record carries per-field confidence flags so low-confidence rows can be surfaced
for manual expert verification instead of silently trusted.

The taxonomy mirrors the production pipeline
(src/agents/services/service_entities/restriction_plan.py):
    intent ∈ {buffers_only, restrictions, needs_clarification}
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

import pandas as pd

# --- column names in exp_data.csv -------------------------------------------
COL_Q = "Промт (вопрос)"
COL_A = "Промт (ответ)"
COL_SN = "service_names"
COL_PN = "phys_names"
COL_SID = "scenario_id"
COL_PROJ = "Наименование проекта"


def _layers_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if c.startswith("Слои"))


# --- normalisation ----------------------------------------------------------
def norm(s: object) -> str:
    """Case/space/ё-insensitive NFKC key used for all matching."""
    return (
        unicodedata.normalize("NFKC", str(s))
        .strip()
        .lower()
        .replace("ё", "е")
    )


# Lexicon: colloquial phrasing found in gold text -> canonical catalog lemma.
# Keys are matched as substrings against a normalised layer-spec / question.
# The canonical value is later resolved against the per-scenario catalog
# (service_names + phys_names) so an entry only "counts" if the scenario
# actually exposes that object type.
LEXICON: dict[str, str] = {
    "жилых дом": "жилой дом",
    "жилые дом": "жилой дом",
    "жилыми дом": "жилой дом",
    "жилой дом": "жилой дом",
    "многоквартирн": "жилой дом",
    "водных объект": "водный объект",
    "водные объект": "водный объект",
    "водными объект": "водный объект",
    "водоем": "водный объект",
    "школ": "школа",
    "образовательн": "школа",
    "детских сад": "детский сад",
    "детсад": "детский сад",
    "поликлиник": "поликлиника",
    "больниц": "больница",
    "аптек": "аптека",
    "парк": "парк",
    "сквер": "сквер",
    "детских площад": "детская площадка",
    "детские площад": "детская площадка",
    "спортивных площад": "спортивная площадка",
    "спортивные площад": "спортивная площадка",
    "спортивная площад": "спортивная площадка",
    "супермаркет": "супермаркет",
    "продуктов": "супермаркет",
    "магазин": "супермаркет",
    "кладбищ": "кладбище",
    "автозаправ": "автозаправочная станция",
    "азс": "автозаправочная станция",
    "остановк": "остановка наземного общественного транспорта",
    "нежил": "нежилое здание",
    "промышленн": "промышленный объект",
    "дорог": "дорога",
    "лес": "лес",
    "рекреацион": "рекреационная зона",
    "кафе": "кафе",
    "ресторан": "ресторан",
    "бар": "бар",
    "детск": "детский сад",
    "музе": "музей",
    "театр": "театр",
    "библиотек": "библиотека",
    "стадион": "стадион",
    "храм": "храм",
    "церк": "храм",
    "гараж": "гараж",
}


@dataclass
class GoldRecord:
    scenario_id: int
    project: str
    question: str
    answer: str
    layers_spec: str
    catalog: list[str]                       # service_names + phys_names for scenario
    # --- derived ground truth ---
    intent: str = "restrictions"             # buffers_only|restrictions|needs_clarification
    source_entity: str | None = None
    target_entity: str | None = None
    distance_m: int | None = None
    expected_layer_count: int | None = None
    expected_object_count: float | None = None
    expected_is_percent: bool = False
    # --- per-field confidence flags (True == confident) ---
    conf: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        d = asdict(self)
        d["catalog_size"] = len(self.catalog)
        d.pop("catalog")
        for k, v in self.conf.items():
            d[f"conf_{k}"] = v
        d.pop("conf")
        return d


# noise words stripped when falling back to the head noun of a layer clause
_CLAUSE_NOISE = re.compile(
    r"^\s*\d+\s*[-–—]?\s*|слой|слои|сло[йея]|с\s+полигон\w*|с\s+точк\w*|"
    r"с\s+буферн\w*\s+зон\w*|буфер\w*|учитыва\w*|вокруг|кажд\w*|"
    r"\d+\s*(?:м\w*|км)|от\b|зон\w*|террит\w*\s+|с\b|の"
)


def _head_noun(clause: str) -> str | None:
    """Strip layer-spec boilerplate, leaving the entity noun phrase."""
    t = norm(clause)
    t = _CLAUSE_NOISE.sub(" ", t)
    t = re.sub(r"[,.;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def _resolve_entity(
    text: str, catalog_norm: dict[str, str]
) -> tuple[str | None, str]:
    """Map a fragment of gold text to a canonical entity.

    Returns (display_name, match_kind) where match_kind is one of:
        'catalog'  – matched an entity present in the scenario catalog
        'lexicon'  – recognised known lemma (may be outside this catalog)
        'headnoun' – fell back to the clause head noun (unknown entity)
        'none'     – nothing extracted
    """
    t = norm(text)
    # direct catalog hit first (longest catalog names first)
    for cnorm, disp in sorted(catalog_norm.items(), key=lambda kv: -len(kv[0])):
        if cnorm and cnorm in t:
            return disp, "catalog"
    # same, tolerating Russian inflection: the gold names objects in whatever case
    # the sentence needs ("со спортивными площадками"), the catalog is nominative
    # singular ("Спортивная площадка"). Match on stems, requiring every word of the
    # catalog entry to be present so short stems cannot match on their own.
    for cnorm, disp in sorted(catalog_norm.items(), key=lambda kv: -len(kv[0])):
        stems = [w[: max(3, len(w) - 4)] for w in cnorm.split() if len(w) > 2]
        if stems and all(s in t for s in stems):
            return disp, "catalog"
    # lexicon lemma -> confirm against catalog when possible
    for frag, lemma in sorted(LEXICON.items(), key=lambda kv: -len(kv[0])):
        if frag in t:
            for cnorm, disp in catalog_norm.items():
                if lemma in cnorm or cnorm in lemma:
                    return disp, "catalog"
            return lemma, "lexicon"
    hn = _head_noun(text)
    if hn:
        return hn, "headnoun"
    return None, "none"


def _split_layers(spec: str) -> list[str]:
    """Split a layer spec into its individual layer clauses.

    Handles both '1 - слой ..., 2 - слой ...' and '1 слой ..., 1 слой ...'.
    """
    s = str(spec)
    # normalise separators: numbered markers "N -" / "N слой" become clause starts
    parts = re.split(r"(?:^|[,;]|\bи\b)\s*(?=\d+\s*(?:[-–—]|слой|сло))", s)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        # fall back: split on 'слой' keyword
        parts = [p.strip() for p in re.split(r"(?=слой)", s) if len(p.strip()) > 4]
    return parts


def parse_record(row: pd.Series, layers_col: str) -> GoldRecord:
    catalog = []
    for col in (COL_SN, COL_PN):
        val = row.get(col)
        if isinstance(val, str):
            catalog += [c.strip() for c in val.split(",") if c.strip()]
    catalog_norm = {norm(c): c for c in catalog}

    rec = GoldRecord(
        scenario_id=int(row[COL_SID]),
        project=str(row.get(COL_PROJ, "")),
        question=str(row[COL_Q]),
        answer=str(row[COL_A]),
        layers_spec=str(row[layers_col]),
        catalog=catalog,
    )
    q, a, spec = norm(rec.question), norm(rec.answer), rec.layers_spec

    # --- distance parameter (from the question; meters) ---
    # matches "200 м", "200 метров", "1 км", "1.5 km"
    dmatches = re.findall(r"(\d+(?:[.,]\d+)?)\s*(км|km|километ\w*|м\w*)", q)
    if dmatches:
        num, unit = dmatches[0]
        d = float(num.replace(",", "."))
        if unit.startswith(("км", "km", "кило")):
            d *= 1000
        rec.distance_m = int(round(d))
        rec.conf["distance"] = True
    else:
        rec.conf["distance"] = False

    # --- expected object count / percentage (from the answer) ---
    if "%" in a:
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*%", a)
        if m:
            rec.expected_object_count = float(m.group(1).replace(",", "."))
            rec.expected_is_percent = True
            rec.conf["count"] = True
    else:
        # the result count is a number directly followed by a NOUN (the object
        # being counted), e.g. "23 жилых дома", "0 АЗС", "среди 7 супермаркетов".
        # A number followed by a unit (м/метр/км) is a distance and is skipped.
        cand = None
        for m in re.finditer(r"(\d+)\s*([а-яёa-z%]+)", a):
            num, word = m.group(1), m.group(2)
            if word in ("м", "метр", "метра", "метров", "км", "km", "мин"):
                continue
            if word.startswith(("метр", "килом", "%")):
                continue
            cand = float(num)
            break
        if cand is None:  # bare integer fallback (exclude distance value)
            nums = [float(n) for n in re.findall(r"\b(\d+)\b", a)]
            nums = [n for n in nums if n != rec.distance_m] or nums
            cand = nums[0] if nums else None
        if cand is not None:
            rec.expected_object_count = cand
            rec.conf["count"] = True
    rec.conf.setdefault("count", False)

    # --- expected layers + source/target entities ---
    # "extracted" == we recovered an entity string at all (catalog|lexicon|headnoun)
    # "in_catalog" == that entity is present in the scenario catalog column
    # Roles follow the convention used by the evaluator: `target` is the BUFFERED
    # entity (the one the pipeline emits as `generators`), `source` is the entity
    # whose objects are counted inside that zone. Note this is inverted with
    # respect to the RestrictionPlan schema, where buffers are built around
    # `source_entities` — gold and derived labels just have to agree, and both
    # sides here use the buffered==target convention.
    clauses = _split_layers(spec)
    rec.expected_layer_count = len(clauses) if clauses else None
    if clauses:
        # target = the clause carrying the buffer, else the second clause
        buf_clause = next((c for c in clauses if "буфер" in norm(c)), None)
        if buf_clause is None and len(clauses) > 1:
            buf_clause = clauses[1]
        # source = the OTHER clause. Taking clauses[0] here collapsed both roles
        # onto the same entity whenever the spec put the buffer clause first —
        # which it does in 44% of the gold — and silently dropped the real target.
        obj_clause = next((c for c in clauses if c is not buf_clause), None)
        src, s_kind = _resolve_entity(obj_clause or clauses[0], catalog_norm)
        rec.source_entity = src
        rec.conf["source"] = s_kind != "none"
        rec.conf["source_in_catalog"] = s_kind == "catalog"
        if buf_clause is not None:
            tgt, t_kind = _resolve_entity(buf_clause, catalog_norm)
            rec.target_entity = tgt
            rec.conf["target"] = t_kind != "none"
            rec.conf["target_in_catalog"] = t_kind == "catalog"
        else:
            rec.conf["target"] = rec.conf["target_in_catalog"] = False
    else:
        rec.conf["source"] = rec.conf["target"] = False
        rec.conf["source_in_catalog"] = rec.conf["target_in_catalog"] = False

    # --- intent (decided from the gold ANSWER semantics) ---
    # needs_clarification: the gold answer itself asks the user to clarify
    clar = any(
        k in a
        for k in ("уточн", "переформул", "не могу", "не найден", "отсутству")
    )
    # restrictions: the answer reports a *filtered* spatial result — a
    # percentage, or a count tied to a zone/radius/buffer, or a filter verb.
    filter_verb = re.search(
        r"выявл|обнаруж|зафиксир|найден|попада|наход|нарушени|обеспечен|"
        r"в радиус|в предел|в границ|в \d+.?метр|охранн\w* зон|"
        r"санитарн\w*|водоохранн|сзз",
        a,
    )
    has_zone = "буфер" in spec.lower() or rec.distance_m is not None
    if clar:
        rec.intent = "needs_clarification"
    elif rec.expected_is_percent or filter_verb:
        rec.intent = "restrictions"
    elif has_zone:
        # buffer drawn but answer states no analytic filter result
        rec.intent = "buffers_only"
    else:
        rec.intent = "buffers_only"
    rec.conf["intent"] = True
    return rec


def load_gold(path: str) -> list[GoldRecord]:
    df = pd.read_csv(path, sep=";", engine="python")
    lc = _layers_col(df)
    df = df[df[COL_Q].notna() & df[COL_A].notna()]
    return [parse_record(r, lc) for _, r in df.iterrows()]


if __name__ == "__main__":
    import sys

    recs = load_gold(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/data/gold/exp_data.csv")
    out = pd.DataFrame([r.as_row() for r in recs])
    print(f"parsed {len(out)} gold records\n")
    print("intent distribution:")
    print(out["intent"].value_counts().to_string(), "\n")
    for fld in ("distance", "count", "source", "target",
                "source_in_catalog", "target_in_catalog"):
        col = f"conf_{fld}"
        print(f"  {fld:20s}: {out[col].sum():3d}/{len(out)}")
    print("\nentities NOT extracted at all (need manual check):")
    bad = out[~out["conf_source"] | ~out["conf_target"]]
    print(f"  {len(bad)} rows")
    out.to_csv("benchmarks/out/gold_parsed.csv", index=False)
    print("\nwrote benchmarks/out/gold_parsed.csv")
