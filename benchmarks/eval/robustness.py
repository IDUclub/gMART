#!/usr/bin/env python3
"""Paraphrase robustness and a taxonomy of the failures.

Robustness: every base gold question was expanded into ~11 paraphrases that mean
the same thing, so a task-aware system should return the same KIND of result for
all of them. Reports, per model, how often a base question's variants agree, and
the mean share of the majority outcome — an "operationally successful" system
that flips between answering and refusing on rephrasing is not usable.

Taxonomy: groups the recorded tracebacks into infrastructure vs model-shaped
failures, so the paper can say which share of the errors the pipeline owns.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

CLAR_KEYS = ("уточн", "доступны", "доступные", "переформул", "отсутств", "не найден")

# exception class -> (bucket, human-readable cause)
ERROR_KINDS = {
    "ValidationError": ("model", "план не прошёл схему RestrictionPlan"),
    "ToolError": ("infra", "ошибка MCP-инструмента"),
    "ChunkedEncodingError": ("infra", "обрыв SSE-потока"),
    "AccessDeniedError": ("infra", "Urban API отказал в доступе"),
    "DownstreamServiceError": ("infra", "недоступен смежный сервис"),
    "TimeoutError": ("infra", "таймаут"),
}


def outcome(rec: dict) -> str:
    if rec.get("error"):
        return "error"
    if rec.get("n_layers") or rec.get("layers"):
        return "layers"
    resp = (rec.get("llm_response") or "").lower()
    return "clarify" if any(k in resp for k in CLAR_KEYS) else "empty"


def error_kind(err: str) -> tuple[str, str]:
    names = re.findall(r"([A-Za-z_]+(?:Error|Exception|Timeout))", str(err))
    for n in reversed(names):
        short = n.split(".")[-1]
        if short in ERROR_KINDS:
            return ERROR_KINDS[short]
    return ("other", names[-1] if names else "неизвестно")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=slim.jsonl")
    ap.add_argument("--dataset", default="benchmarks/data/gold/expanded_goldfirst.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.dataset, sep=";", engine="python")
    base_of = {i: int(b) for i, b in enumerate(df["base_index"])}

    for spec in args.models:
        name, path = spec.split("=", 1)
        by_base: dict[int, list[str]] = defaultdict(list)
        errors: list[str] = []
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            base = base_of.get(rec.get("idx"))
            if base is None:
                continue
            by_base[base].append(outcome(rec))
            if rec.get("error"):
                errors.append(rec["error"])

        groups = [v for v in by_base.values() if len(v) > 1]
        unanimous = sum(1 for v in groups if len(set(v)) == 1)
        majority = [Counter(v).most_common(1)[0][1] / len(v) for v in groups]
        flips = sum(1 for v in groups if {"layers", "clarify"} <= set(v))

        print(f"\n=== {name} ===")
        print(f"  base questions with >1 variant: {len(groups)} "
              f"(mean {sum(len(v) for v in groups) / max(len(groups), 1):.1f} variants)")
        print(f"  same outcome for every paraphrase: {unanimous} "
              f"({100 * unanimous / max(len(groups), 1):.0f}%)")
        print(f"  mean majority share:               "
              f"{100 * sum(majority) / max(len(majority), 1):.0f}%")
        print(f"  answered some paraphrases and refused others: {flips} "
              f"({100 * flips / max(len(groups), 1):.0f}%)")

        if errors:
            buckets = Counter()
            causes = Counter()
            for e in errors:
                b, c = error_kind(e)
                buckets[b] += 1
                causes[c] += 1
            print(f"  errors: {len(errors)}")
            for b, n in buckets.most_common():
                print(f"    {b:6s} {n:4d} ({100 * n / len(errors):3.0f}%)")
            for c, n in causes.most_common(5):
                print(f"      - {c}: {n}")


if __name__ == "__main__":
    main()
