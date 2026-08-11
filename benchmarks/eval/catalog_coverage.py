#!/usr/bin/env python3
"""How much of the benchmark is answerable at all, given each scenario's LIVE catalog?

The clarification replies enumerate the scenario's catalog ("Доступные ...:"),
so the live catalog can be reconstructed per scenario from the run itself.
Cross it with the gold entities to split every row into:

  answerable      — both gold entities exist in the live catalog
  unanswerable    — a gold entity is absent (the correct behaviour is to clarify)

and then score each model on that split. The interesting cell is
"unanswerable + produced layers": the model invented a substitute entity and
returned a confident-looking result — a false success that a layers-only
metric counts as a win.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gold_parser import load_gold, norm  # noqa: E402

CLAR_KEYS = ("уточн", "доступны", "доступные", "переформул", "отсутств", "не найден")
TAIL_RE = re.compile(r"доступные\s+(?:сервисы|физические объекты)\s*:?\s*(.+)", re.I)


def outcome(rec: dict) -> str:
    if rec.get("error"):
        return "error"
    if rec.get("n_layers"):
        return "layers"
    resp = (rec.get("llm_response") or "").lower()
    return "clarify" if any(k in resp for k in CLAR_KEYS) else "empty"


def catalog_from_replies(recs: list[dict]) -> dict[int, set[str]]:
    """Reconstruct each scenario's live catalog from its clarification replies."""
    cat: dict[int, set[str]] = defaultdict(set)
    for r in recs:
        resp = r.get("llm_response") or ""
        for m in TAIL_RE.finditer(resp):
            line = m.group(1).split("\n")[0]
            for tok in re.split(r"[,;]", line.replace("[", "").replace("]", "")):
                tok = norm(tok.strip().strip("'\"."))
                if tok and len(tok) > 2:
                    cat[r["scenario_id"]].add(tok)
    return cat


def in_catalog(entity: str, cat: set[str]) -> bool:
    """An entity counts as present if any catalog item shares a content word."""
    if not entity:
        return True
    words = [w for w in norm(entity).split() if len(w) > 3]
    if not words:
        words = norm(entity).split()
    return any(any(w in item for item in cat) for w in words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=slim.jsonl")
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    args = ap.parse_args()

    gold = load_gold(args.gold)
    # gold entities per scenario (a scenario's paraphrases share its entities)
    ents_by_sid: dict[int, set[str]] = defaultdict(set)
    for g in gold:
        for e in (g.source_entity, g.target_entity):
            if e:
                ents_by_sid[g.scenario_id].add(e)

    loaded = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        loaded[name] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    # one catalog view built from every model's replies (more coverage)
    cat: dict[int, set[str]] = defaultdict(set)
    for recs in loaded.values():
        for sid, items in catalog_from_replies(recs).items():
            cat[sid] |= items

    covered = [s for s in cat if cat[s]]
    print(f"scenarios with a reconstructed catalog: {len(covered)}")

    for name, recs in loaded.items():
        cells: dict[tuple[str, str], int] = defaultdict(int)
        for r in recs:
            sid = r["scenario_id"]
            if sid not in cat or not cat[sid]:
                continue
            ents = ents_by_sid.get(sid, set())
            if not ents:
                continue
            answerable = all(in_catalog(e, cat[sid]) for e in ents)
            cells[("answerable" if answerable else "unanswerable", outcome(r))] += 1
        print(f"\n=== {name} ===")
        for grp in ("answerable", "unanswerable"):
            tot = sum(v for (g, _), v in cells.items() if g == grp)
            if not tot:
                continue
            row = "  ".join(f"{o}={cells[(grp, o)]} ({100*cells[(grp,o)]/tot:.0f}%)"
                            for o in ("layers", "clarify", "error", "empty"))
            print(f"  {grp:12s} n={tot:4d}  {row}")
        unans = sum(v for (g, _), v in cells.items() if g == "unanswerable")
        if unans:
            fs = cells[("unanswerable", "layers")]
            print(f"  -> false success (unanswerable but returned layers): "
                  f"{fs}/{unans} ({100*fs/unans:.0f}%)")


if __name__ == "__main__":
    main()
