#!/usr/bin/env python3
"""Compact one-shot status of the live re-run batch (called by the 200-record watch)."""

from __future__ import annotations

import glob
import json
import os

DIR = "benchmarks/data/results_rerun_full"
N_TOTAL = 2186


def summarise(path: str) -> dict:
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    base = [r for r in recs if r.get("idx", 0) < 202]
    layers = sum(1 for r in recs if r.get("layers"))
    err = sum(1 for r in recs if r.get("error"))
    clar = sum(1 for r in recs if not r.get("error") and not r.get("layers")
               and any(k in str(r.get("llm_response", "")).lower()
                       for k in ("уточн", "доступн", "переформул", "отсутств")))
    return {"n": len(recs), "base": len(base), "layers": layers,
            "err": err, "clar": clar}


def main() -> None:
    files = sorted(glob.glob(f"{DIR}/*/results.jsonl"))
    total = sum(len(open(f, encoding="utf-8").readlines()) for f in files)
    lines = [f"STATUS — {total} records done (batch: 2 models × {N_TOTAL})"]
    for f in files:
        model = os.path.basename(os.path.dirname(f))
        s = summarise(f)
        pct = 100 * s["n"] / N_TOTAL
        lines.append(
            f"  {model:12s} {s['n']:4d}/{N_TOTAL} ({pct:4.0f}%)  "
            f"base-202: {s['base']}/202  |  layers={s['layers']} "
            f"clarify={s['clar']} error={s['err']}"
        )
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
