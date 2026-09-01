#!/usr/bin/env python3
"""Keep only the gold rows the restrictions pipeline can actually be asked.

The expert set was written as a list of useful urban-planning questions, not as
a test suite for one pipeline. A large part of it asks for something this
pipeline does not compute, or names objects the scenario does not contain — and
in the run those rows come back as clarifications that are then read as model
failures. That is the reviewer's complaint about the completion proxy in a new
place: a metric charged to the model for something the model had no part in.

Five reasons a row is dropped, each verifiable rather than inferred:

``no_distance``
    No metric radius in the question. The plan needs ``buffer_size``; without
    one the only correct behaviour is to ask, so the row measures nothing.
``time_distance``
    The radius is given in *minutes* ("пешей доступности, 10 минут"), i.e. an
    isochrone. ``CreateBuffers`` builds metric buffers; a travel-time catchment
    is a different tool and a different service.
``complement``
    The question asks for what is **outside** the zone ("вне радиуса", "не
    оснащены ... в радиусе"). The pipeline selects objects *inside* a buffer;
    the complement is not the same operation with a different word.
``entity_absent``
    One of the two entities does not exist in that scenario's Urban API
    catalog, so no plan can name it and no layer can be built.
``data_unavailable``
    The layer cannot be fetched at all — the scenario is private (403), missing
    (404), or the entity hits the Urban API's line-geometry serialisation bug.
    Taken from the prefetch gap report, i.e. measured, not guessed.

What is deliberately **not** a reason to drop: a question phrased as a share or
as a verdict ("какой процент жилых домов…", "являются ли библиотеки
доступными…"). The spatial operation underneath is exactly buffer-and-contain;
only the presentation of the answer differs. Those rows are reported as the
``provision_phrasing`` class so the decision to keep or cut them is explicit and
reversible, instead of being smuggled in by a regex.

Entity resolution runs against the scenario's real catalog, read from the
offline store. Between the expert's prose and the catalog lemma sits ALIASES —
every entry there was added after reading an actual rejection, because a
matcher that silently drops a valid expert record is worse than no filter.

    python benchmarks/eval/restrictions_filter.py \\
        --gold benchmarks/data/gold/exp_data.csv \\
        --out benchmarks/data/gold/exp_data_restrictions.csv \\
        --report benchmarks/out/dataset_filter.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import get_close_matches
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from expand_catalog_dataset import catalog_from_store  # noqa: E402
from gold_parser import COL_Q, COL_SID, load_gold, norm  # noqa: E402

KEEP = "keep"
NO_DISTANCE = "no_distance"
TIME_DISTANCE = "time_distance"
COMPLEMENT = "complement"
ENTITY_ABSENT = "entity_absent"
DATA_UNAVAILABLE = "data_unavailable"
PROVISION = "provision_phrasing"

DROP_REASONS = (
    NO_DISTANCE,
    TIME_DISTANCE,
    COMPLEMENT,
    ENTITY_ABSENT,
    DATA_UNAVAILABLE,
)

# A radius stated in travel time. The pipeline has no isochrone tool, so these
# are not "restrictions questions with an unusual unit" — they are questions for
# a different service.
_TIME = re.compile(
    r"минут|\bмин\.|пешей доступност|пешеходной доступност|"
    r"транспортной доступност|изохрон"
)

# Selection of the complement of the buffer. "не оснащены X в радиусе 200 м"
# reads as "within 200 m" to a keyword matcher and means the opposite.
_COMPLEMENT = re.compile(
    r"не оснащ|не обеспеч|\bвне\b|за предел|не ближе|дальше чем|"
    r"не попада|не наход|отсутству"
)

# Share/verdict phrasing. Kept, counted, reported — see the module docstring.
_PROVISION = re.compile(
    r"процент|\bдоля\b|\bдолю\b|%|являются ли|соблюдаются ли|соблюдается ли|"
    r"обеспечены ли|достаточно ли"
)

# Expert prose -> catalog lemma. **Synonyms only.** An identity entry
# ("парковка" -> "парковка") is not merely redundant — the alias lookup below
# matches on a coarse three-letter-per-word signature, under which "парк" and
# "парковка" collide, so an identity entry for one silently swallows the other.
# Anything that is the catalog lemma already is matched literally without help.
# Every entry here was added after reading the rejection it fixes.
ALIASES: dict[str, str] = {
    "жилое здание": "жилой дом",  # 846, 1390, 1440, 5021, 10068
    "жилая застройка": "жилой дом",
    "многоквартирный дом": "жилой дом",
    "остановка общественного транспорта": "остановка наземного транспорта",  # 744
    # The layer spec often names the bare noun ("- остановок"); 846/861/1747/5575.
    "остановка": "остановка наземного транспорта",
    "спортивный центр": "спортивный зал",  # 198, 846, 875, 965
    "магазин одежды": "одежда и обувь",  # 772, 950, 1244, 1269, 1429, 1640
    "магазин продуктов": "продукты (магазин у дома)",  # 198
    "тбо": "полигон тбо",  # 965
    "линия электропередач": "линия электропередачи",
    "распределительный трубопровод газа": (
        "распределительный трубопровод для транспортировки газа"
    ),
    "промышленная зона": "промышленная территория",  # 744, 124
    "промышленный объект": "промышленная территория",
    "автозаправочная станция": "заправочная станция",  # 124
    "азс": "заправочная станция",
}

# Prepositional debris the gold extraction leaves on an entity name
# ("реками и в них", "со спортивными центрами", "в остановок").
_DEBRIS = re.compile(
    r"^\s*[-–—]\s*|"
    r"^\s*(?:со?|в|на|у|от|к|для|с учетом)\s+|"
    r"\s+и\s+в\s+них\s*$|\s+в\s+этом\s*$|\s+попавшими\s+в.*$|\s+и\s*$"
)


def clean_entity(name: str | None) -> str:
    if not name:
        return ""
    value = norm(name)
    previous = None
    while value != previous:
        previous = value
        value = _DEBRIS.sub("", value).strip()
    return value


def _signature(text: str) -> frozenset[str]:
    """Three letters per word — coarse on purpose, and only ever used on ALIASES.

    Russian inflection eats the tail («жилое» / «жилыми», «здание» / «зданиями»),
    so three letters is what actually survives it. That is far too coarse for the
    catalog — «парк» and «парковка» share a signature — which is why this is
    never matched against catalog entries, only against the hand-checked alias
    keys, where such a collision would be visible in the table.
    """

    words = [w for w in re.findall(r"[а-яёa-z]+", norm(text)) if len(w) > 2]
    return frozenset(w[:3] for w in words)


def _same_lemma(left: str, right: str) -> bool:
    """Whether two strings are inflections of one word rather than two words.

    Russian inflection changes the tail and keeps the stem, so the test is on
    how much of each string the shared prefix covers: «школами»/«школа» share
    all five letters of the shorter one and match, while «парк»/«парковка»
    share four and leave «овка» hanging off the longer one, so they do not.
    Length, not similarity, is what separates those two cases — by string ratio
    they are 0.83 and 0.67, far too close to put a threshold between.
    """

    if not left or not right:
        return False
    shared = 0
    for a, b in zip(left, right):
        if a != b:
            break
        shared += 1
    return shared >= 4 and shared >= len(left) - 3 and shared >= len(right) - 3


def _alias_lemmas(cleaned: str) -> list[str]:
    """Alias lemmas this phrase maps to, inflection-tolerant.

    A literal lookup misses every key the table has, because the gold text is
    inflected («жилыми зданиями» for «жилое здание»); a fuzzy string ratio
    cannot be used instead, since «жилыми зданиями»/«жилое здание» scores 0.67 —
    exactly what «парк»/«парковка» scores, and one of those must not match.
    """

    lemmas: list[str] = []
    if cleaned in ALIASES:
        lemmas.append(ALIASES[cleaned])
    signature = _signature(cleaned)
    if signature:
        for alias, lemma in ALIASES.items():
            alias_signature = _signature(alias)
            if alias_signature and alias_signature <= signature:
                lemmas.append(lemma)
    return lemmas


def resolve(name: str | None, catalog: list[str]) -> str | None:
    """The catalog lemma this expert phrase denotes, or None.

    Literal match first, then the synonym table, then a tight near-miss pass for
    the typos the expert text actually contains («отсановками», «останвоками»,
    «жилых домоа»). The near-miss cutoff is set where inflection of one lemma
    lands (0.85+) and below where two different lemmas do — «парковок»/«парковка»
    is 0.88 and matches, «парк»/«парковка» is 0.67 and does not. Rejecting a
    valid expert row over the expert's typo is the failure this filter must not
    have; quietly merging two catalog entries is the other one.
    """

    cleaned = clean_entity(name)
    if not cleaned:
        return None
    catalog_norm = {norm(entry): entry for entry in catalog}

    for candidate in (cleaned, *_alias_lemmas(cleaned)):
        if candidate in catalog_norm:
            return catalog_norm[candidate]

    for candidate in (cleaned, *_alias_lemmas(cleaned)):
        for entry_norm, entry in catalog_norm.items():
            if _same_lemma(candidate, entry_norm):
                return entry

    for candidate in (cleaned, *_alias_lemmas(cleaned)):
        close = get_close_matches(candidate, list(catalog_norm), n=1, cutoff=0.85)
        if close:
            return catalog_norm[close[0]]
    return None


def blocked_entities(report_paths: list[Path]) -> dict[int, set[str]]:
    """Per scenario, the entity names the Urban API could not serve.

    ``"*"`` means the scenario itself is unreachable (403/404).
    """

    blocked: dict[int, set[str]] = {}
    for path in report_paths:
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for scenario in report.get("scenarios", []):
            try:
                scenario_id = int(scenario["scenario_id"])
            except (KeyError, TypeError, ValueError):
                continue
            for failure in scenario.get("failed", []):
                name = failure.get("name")
                blocked.setdefault(scenario_id, set()).add(norm(name) if name else "*")
    return blocked


def classify(
    record,
    catalog: list[str],
    blocked: set[str],
) -> tuple[str, str]:
    """``(reason, detail)`` — ``KEEP`` when the row is a restrictions task."""

    if not catalog or "*" in blocked:
        return DATA_UNAVAILABLE, "scenario unreachable (403/404)"
    question = norm(record.question)
    if record.intent == "needs_clarification":
        return NO_DISTANCE, "gold intent is needs_clarification"
    if _COMPLEMENT.search(question):
        return COMPLEMENT, "selects objects outside the buffer"
    if record.distance_m is None:
        return NO_DISTANCE, "no radius in the question"
    if _TIME.search(question):
        return TIME_DISTANCE, "radius given as travel time"

    buffered = resolve(record.target_entity, catalog)
    counted = resolve(record.source_entity, catalog)
    missing = [
        f"{role}={name!r}"
        for role, name, hit in (
            ("buffered", record.target_entity, buffered),
            ("counted", record.source_entity, counted),
        )
        if not hit
    ]
    if missing:
        return ENTITY_ABSENT, "not in the scenario catalog: " + ", ".join(missing)
    for lemma, role in ((buffered, "buffered"), (counted, "counted")):
        if norm(lemma) in blocked:
            return DATA_UNAVAILABLE, f"{role}={lemma!r} hits the Urban API 500"
    return KEEP, f"buffered={buffered!r}, counted={counted!r}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    parser.add_argument(
        "--out", default="benchmarks/data/gold/exp_data_restrictions.csv"
    )
    parser.add_argument("--report", default="benchmarks/out/dataset_filter.md")
    parser.add_argument(
        "--urban-data-dir", default="runtime/urban_data", help="the offline store"
    )
    parser.add_argument(
        "--gaps",
        nargs="*",
        default=["runtime/urban_data/gaps_token.json", "runtime/urban_data/gaps.json"],
    )
    parser.add_argument(
        "--drop-provision",
        action="store_true",
        help="also drop share/verdict phrasing (provision-style questions)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    frame = pd.read_csv(args.gold, sep=";", engine="python")
    frame = frame[frame[COL_Q].notna() & frame[COL_SID].notna()].reset_index(drop=True)
    if len(frame) != len(gold):
        raise SystemExit(
            f"row mismatch: load_gold gave {len(gold)}, the frame has {len(frame)}; "
            "the filter indexes one by the other and must not guess"
        )

    store = Path(args.urban_data_dir)
    blocked = blocked_entities([Path(p) for p in args.gaps])
    catalogs: dict[int, list[str]] = {}
    for record in gold:
        if record.scenario_id not in catalogs:
            pair = catalog_from_store(store, record.scenario_id)
            catalogs[record.scenario_id] = sorted(
                {*pair["service"], *pair["physical_object"]}
            )

    decisions = []
    for index, record in enumerate(gold):
        reason, detail = classify(
            record,
            catalogs[record.scenario_id],
            blocked.get(record.scenario_id, set()),
        )
        decisions.append(
            {
                "index": index,
                "scenario_id": record.scenario_id,
                "reason": reason,
                "detail": detail,
                "provision": bool(_PROVISION.search(norm(record.question))),
                "question": record.question.strip()[:160],
            }
        )

    keep = [d for d in decisions if d["reason"] == KEEP]
    if args.drop_provision:
        for decision in keep:
            if decision["provision"]:
                decision["reason"] = PROVISION
                decision["detail"] = "share/verdict phrasing (provision task)"
        keep = [d for d in decisions if d["reason"] == KEEP]

    kept_frame = frame.iloc[[d["index"] for d in keep]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    kept_frame.to_csv(args.out, sep=";", index=False, encoding="utf-8")

    counts = Counter(d["reason"] for d in decisions)
    lines = [
        "# Gold set filtered to restrictions-pipeline tasks\n",
        f"\nSource: `{args.gold}` — {len(decisions)} rows.  ",
        f"Kept: **{len(keep)}** -> `{args.out}`\n",
        "\n| outcome | rows |\n|---|---|\n",
    ]
    for reason, count in counts.most_common():
        lines.append(f"| `{reason}` | {count} |\n")
    provision_kept = sum(1 for d in keep if d["provision"])
    lines.append(
        f"\nOf the kept rows, {provision_kept} are phrased as a share or a "
        "verdict (`provision_phrasing`). They are kept because the spatial "
        "operation is the same buffer-and-contain; `--drop-provision` removes "
        "them.\n"
    )
    for reason in DROP_REASONS:
        dropped = [d for d in decisions if d["reason"] == reason]
        if not dropped:
            continue
        lines.append(f"\n## Dropped — `{reason}` ({len(dropped)})\n\n")
        lines.append("| # | scen | why | question |\n|---|---|---|---|\n")
        for decision in dropped:
            question = decision["question"].replace("|", "/")
            lines.append(
                f"| {decision['index']} | {decision['scenario_id']} | "
                f"{decision['detail']} | {question} |\n"
            )
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("".join(lines), encoding="utf-8")

    print(f"{len(decisions)} rows -> kept {len(keep)}")
    for reason, count in counts.most_common():
        print(f"  {reason:<20} {count}")
    print(f"\nfiltered set: {args.out}\nreport: {args.report}")


if __name__ == "__main__":
    main()
