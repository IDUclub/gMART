#!/usr/bin/env python3
"""Strip the heavy geometry out of a results.jsonl into a small analysis file.

The run logs every layer's full FeatureCollection, so results.jsonl is GBs and
cannot be loaded repeatedly. This keeps only what the failure analysis needs:
idx, scenario_id, prompt, layer names+counts, the (truncated) reply and error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def slim(rec: dict, resp_chars: int) -> dict:
    lc = rec.get("layer_counts") or {}
    if not lc and rec.get("layers"):
        for layer in rec["layers"]:
            try:
                lc[str(layer.get("name"))] = len(layer["feature_collection"]["features"])
            except Exception:  # noqa: BLE001
                lc[str(layer.get("name"))] = None
    err = rec.get("error")
    return {
        "idx": rec.get("idx"),
        "scenario_id": rec.get("scenario_id"),
        "prompt": rec.get("prompt"),
        "layer_counts": lc,
        "n_layers": len(lc),
        "llm_response": (str(rec.get("llm_response") or ""))[:resp_chars],
        "error": (str(err)[:resp_chars] if err else None),
        "elapsed": rec.get("elapsed"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resp-chars", type=int, default=2000)
    args = ap.parse_args()

    n = 0
    with open(args.results, encoding="utf-8") as fh, \
            open(args.out, "w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            out.write(json.dumps(slim(json.loads(line), args.resp_chars),
                                 ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} records -> {args.out} "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
