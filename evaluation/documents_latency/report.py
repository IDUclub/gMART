"""Compare two benchmark runs in the report format of the optimisation plan.

    python evaluation/documents_latency/report.py baseline.jsonl current.jsonl \
        --labels labels.jsonl > report.md

Latency comes from the client timeline (comparable across any two versions).
LLM calls/tokens, the critic's own latency and first-draft acceptance come from
the server record (``DVD run metrics``) when the run has one; the client value is
the fallback. Quality columns need manual labels, one JSON object per line:

    {"label": "baseline", "id": "A-01", "run": 0, "correct": true,
     "grounded": true, "citation_ok": true, "critical_error": false}

Results are reported overall and per query class: a global average can hide a
regression in one class behind a gain in another.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    index = min(len(values) - 1, max(0, round(q / 100 * (len(values) - 1))))
    return values[index]


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def rate(values):
    values = [bool(v) for v in values if v is not None]
    return 100 * sum(values) / len(values) if values else None


def server(record, *path):
    value = record.get("server")
    for key in path:
        value = (value or {}).get(key)
    return value


def first_draft_accepted(record):
    accepted = server(record, "decisions", "first_draft_accepted")
    if accepted is not None:
        return accepted
    return None if record.get("error") else record.get("attempts") == 1


def metrics(records, labels):
    done = [r for r in records if not r.get("skipped")]
    ok = [r for r in done if not r.get("error")]
    labelled = [labels.get((r.get("label"), r["id"], r["run"])) for r in done]
    labelled = [x for x in labelled if x]
    return {
        "cases": len(done),
        "errors": len(done) - len(ok),
        "p50 total latency, s": percentile([r["total_ms"] for r in ok], 50),
        "p95 total latency, s": percentile([r["total_ms"] for r in ok], 95),
        "p50 TTFT, s": percentile([r.get("ttft_ms") for r in ok], 50),
        "p95 TTFT, s": percentile([r.get("ttft_ms") for r in ok], 95),
        "average LLM calls": mean([server(r, "llm", "calls") for r in ok]),
        "average output tokens": mean([server(r, "llm", "output_tokens") for r in ok]),
        "average planner latency, s": mean(
            [
                server(r, "stages_ms", "planner")
                or r["client_stages_ms"].get("planner", 0)
                for r in ok
            ]
        ),
        "average critic latency, s": mean(
            [
                server(r, "stages_ms", "critic")
                or r["client_stages_ms"].get("critic", 0)
                for r in ok
            ]
        ),
        "first draft accepted, %": rate([first_draft_accepted(r) for r in ok]),
        "retrieval retries": mean(
            [max(0, (server(r, "counters", "retrieval_rounds") or 1) - 1) for r in ok]
        ),
        "correct answer rate, %": rate([x.get("correct") for x in labelled]),
        "grounded answer rate, %": rate([x.get("grounded") for x in labelled]),
        "citation correctness, %": rate([x.get("citation_ok") for x in labelled]),
        "critical errors": (
            sum(bool(x.get("critical_error")) for x in labelled) if labelled else None
        ),
    }


SECONDS = {key for key in metrics([], {}) if key.endswith(", s")}


def fmt(key, value):
    if value is None:
        return "n/a"
    if key in SECONDS:
        return f"{value / 1000:.1f}"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def delta(key, base, current):
    if base is None or current is None:
        return "n/a"
    if key.endswith(", %") or key in {"cases", "errors", "critical errors"}:
        return f"{current - base:+.1f}"
    if not base:
        return "n/a"
    return f"{100 * (current - base) / base:+.0f}%"


def table(title, base, current):
    rows = [
        f"### {title}",
        "",
        "| Metric | Baseline | Current | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in base:
        rows.append(
            f"| {key} | {fmt(key, base[key])} | {fmt(key, current[key])} | "
            f"{delta(key, base[key], current[key])} |"
        )
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--labels", type=Path)
    args = parser.parse_args()
    labels = {}
    if args.labels and args.labels.exists():
        for item in load(args.labels):
            labels[(item.get("label"), item["id"], item.get("run", 0))] = item
    base, current = load(args.baseline), load(args.current)
    print(table("All classes", metrics(base, labels), metrics(current, labels)))
    for cls in sorted({r["class"] for r in base + current}):
        print(
            table(
                f"Class {cls}",
                metrics([r for r in base if r["class"] == cls], labels),
                metrics([r for r in current if r["class"] == cls], labels),
            )
        )


if __name__ == "__main__":
    main()
