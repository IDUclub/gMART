#!/usr/bin/env python3
"""Why did the clarify rate explode over the last ~350 rows of the run?

Compares the tail against the head per scenario and per model, so a data
property (that scenario's catalog lacks the entity) can be told apart from a
model property (this model refuses where the other one answers).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

CLAR_KEYS = ("уточн", "доступны", "доступные", "переформул", "отсутств", "не найден")


def outcome(rec: dict) -> str:
    if rec.get("error"):
        return "error"
    if rec.get("n_layers"):
        return "layers"
    resp = (rec.get("llm_response") or "").lower()
    return "clarify" if any(k in resp for k in CLAR_KEYS) else "empty"


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="name=path/to/slim.jsonl")
    ap.add_argument("--split", type=int, default=1836,
                    help="idx where the observed clarify jump starts")
    args = ap.parse_args()

    data = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        data[name] = {r["idx"]: r for r in load(path)}

    for name, recs in data.items():
        head = [r for i, r in recs.items() if i < args.split]
        tail = [r for i, r in recs.items() if i >= args.split]
        print(f"\n=== {name} ===")
        for label, part in (("head", head), ("tail", tail)):
            c = Counter(outcome(r) for r in part)
            n = len(part) or 1
            print(f"  {label:4s} n={len(part):4d}  " + "  ".join(
                f"{k}={c[k]} ({100*c[k]/n:.0f}%)"
                for k in ("layers", "clarify", "error", "empty")))
        # which scenarios live in the tail, and how each behaves overall
        tail_sids = Counter(r["scenario_id"] for r in tail)
        print(f"  tail scenarios: {len(tail_sids)} distinct -> "
              f"{tail_sids.most_common(8)}")
        by_sid: dict[int, Counter] = defaultdict(Counter)
        for r in recs.values():
            by_sid[r["scenario_id"]][outcome(r)] += 1
        print("  clarify rate for the tail scenarios (whole run):")
        for sid, cnt in tail_sids.most_common(8):
            c = by_sid[sid]
            tot = sum(c.values())
            print(f"    sid={sid:6d} n={tot:4d}  clarify={c['clarify']:4d} "
                  f"({100*c['clarify']/tot:3.0f}%)  layers={c['layers']:4d}")

    # cross-model agreement on the tail rows
    if len(data) == 2:
        (n1, r1), (n2, r2) = data.items()
        common = [i for i in r1 if i in r2 and i >= args.split]
        agree = sum(1 for i in common
                    if outcome(r1[i]) == outcome(r2[i]) == "clarify")
        only1 = sum(1 for i in common
                    if outcome(r1[i]) == "clarify" != outcome(r2[i]))
        only2 = sum(1 for i in common
                    if outcome(r2[i]) == "clarify" != outcome(r1[i]))
        print(f"\n=== tail agreement (n={len(common)}) ===")
        print(f"  both clarify:      {agree}")
        print(f"  only {n1}: {only1}")
        print(f"  only {n2}: {only2}")


if __name__ == "__main__":
    main()
