#!/usr/bin/env python3
"""Split every run row by whether the gold task is answerable in its scenario.

Answerability comes from the gold itself, not from the models' replies: each gold
record carries the scenario's expert-authored catalog, and `gold_parser` reports
per entity whether it resolved against that catalog. A gold question naming an
entity the scenario never had (e.g. "дома престарелых" in a scenario listing only
stops, water and buildings) is unanswerable — the correct behaviour there is to
clarify, and producing layers means the model substituted something else.

The interesting cell is "unanswerable + layers": a false success that any
layer-only metric counts as a win.

Rows are joined to gold through `base_index` in the expanded dataset, so
paraphrases inherit the entities of the base question they were generated from.

Every share is reported twice. Per row is the raw count, and it is dominated by a
few scenarios: the expansion produced 354 rows for one scenario and 4 for another,
so a row-weighted mean mostly measures which questions got many paraphrases. Per
question first averages within a base question and then across base questions, so
every gold question counts once regardless of how many paraphrases it received —
that is the number to quote.

Usage:
  answerability.py --models gpt-oss=out/slim_gpt-oss_20b.jsonl [...] \
                   [--dataset benchmarks/data/gold/expanded_goldfirst.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gold_parser import load_gold  # noqa: E402

CLAR_KEYS = ("уточн", "доступны", "доступные", "переформул", "отсутств", "не найден")


def outcome(rec: dict) -> str:
    if rec.get("error"):
        return "error"
    if rec.get("n_layers") or rec.get("layers"):
        return "layers"
    resp = (rec.get("llm_response") or "").lower()
    return "clarify" if any(k in resp for k in CLAR_KEYS) else "empty"


def macro_share(
    per_base: dict[int, dict[str, int]], bases: list[int], out: str
) -> float:
    """Share of `out`, averaged over base questions instead of over rows."""

    shares = []
    for b in bases:
        counts = per_base[b]
        total = sum(counts.values())
        if total:
            shares.append(counts.get(out, 0) / total)
    return sum(shares) / len(shares) if shares else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=slim.jsonl")
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--dataset", default="benchmarks/data/gold/expanded_goldfirst.csv")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    df = pd.read_csv(args.dataset, sep=";", engine="python")
    base_of = {i: int(b) for i, b in enumerate(df["base_index"])}

    # answerable == every entity the gold names resolved against the scenario catalog
    answerable = {
        i: bool(g.conf.get("source_in_catalog") and g.conf.get("target_in_catalog"))
        for i, g in enumerate(gold)
    }
    n_ans = sum(answerable.values())
    print(
        f"gold questions answerable in their own scenario: {n_ans}/{len(gold)} "
        f"({100 * n_ans / len(gold):.0f}%)"
    )

    for spec in args.models:
        name, path = spec.split("=", 1)
        cells: dict[tuple[str, str], int] = defaultdict(int)
        per_base: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        unmatched = 0
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            base = base_of.get(rec.get("idx"))
            if base is None or base not in answerable:
                unmatched += 1
                continue
            grp = "answerable" if answerable[base] else "unanswerable"
            cells[(grp, outcome(rec))] += 1
            per_base[base][outcome(rec)] += 1

        print(f"\n=== {name} ===")
        for grp in ("answerable", "unanswerable"):
            tot = sum(v for (g, _), v in cells.items() if g == grp)
            if not tot:
                continue
            bases = [b for b in per_base if (answerable[b]) == (grp == "answerable")]
            row = "  ".join(
                f"{o}={cells[(grp, o)]} ({100 * cells[(grp, o)] / tot:.0f}%)"
                for o in ("layers", "clarify", "error", "empty")
            )
            print(f"  {grp:12s} per row      n={tot:4d}  {row}")
            row = "  ".join(
                f"{o}={100 * macro_share(per_base, bases, o):.0f}%"
                for o in ("layers", "clarify", "error", "empty")
            )
            print(f"  {'':12s} per question n={len(bases):4d}  {row}")
        unans_bases = [b for b in per_base if not answerable[b]]
        unans = sum(v for (g, _), v in cells.items() if g == "unanswerable")
        if unans:
            fs = cells[("unanswerable", "layers")]
            print(
                f"  -> false success (unanswerable but returned layers): "
                f"{fs}/{unans} ({100 * fs / unans:.0f}%) per row, "
                f"{100 * macro_share(per_base, unans_bases, 'layers'):.0f}% per question"
            )
        if unmatched:
            print(f"  ({unmatched} rows could not be joined to gold)")


if __name__ == "__main__":
    main()
