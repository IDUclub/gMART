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
    print(f"gold questions answerable in their own scenario: {n_ans}/{len(gold)} "
          f"({100 * n_ans / len(gold):.0f}%)")

    for spec in args.models:
        name, path = spec.split("=", 1)
        cells: dict[tuple[str, str], int] = defaultdict(int)
        unmatched = 0
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            base = base_of.get(rec.get("idx"))
            if base is None or base not in answerable:
                unmatched += 1
                continue
            grp = "answerable" if answerable[base] else "unanswerable"
            cells[(grp, outcome(rec))] += 1

        print(f"\n=== {name} ===")
        for grp in ("answerable", "unanswerable"):
            tot = sum(v for (g, _), v in cells.items() if g == grp)
            if not tot:
                continue
            row = "  ".join(
                f"{o}={cells[(grp, o)]} ({100 * cells[(grp, o)] / tot:.0f}%)"
                for o in ("layers", "clarify", "error", "empty")
            )
            print(f"  {grp:12s} n={tot:4d}  {row}")
        unans = sum(v for (g, _), v in cells.items() if g == "unanswerable")
        if unans:
            fs = cells[("unanswerable", "layers")]
            print(f"  -> false success (unanswerable but returned layers): "
                  f"{fs}/{unans} ({100 * fs / unans:.0f}%)")
        if unmatched:
            print(f"  ({unmatched} rows could not be joined to gold)")


if __name__ == "__main__":
    main()
