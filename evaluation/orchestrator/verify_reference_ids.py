"""Verify returned Urban MCP IDs against the independent REST snapshots."""

import argparse
import json
from pathlib import Path

from review_answers import ROOT, rest_checks
from run_planner import save

parser = argparse.ArgumentParser()
parser.add_argument("--runs", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
refs = {
    p.stem: json.loads(p.read_text(encoding="utf-8"))
    for p in (ROOT / "benchmarks/data/orchestrator_20260910/reference").glob("*.json")
}
rows = []
for path in sorted(args.runs.glob("*.json")):
    trace = json.loads(path.read_text(encoding="utf-8"))
    checks = rest_checks(trace, refs)
    if checks:
        rows.append({"id": trace["id"], "checks": checks})
checks = [c for row in rows for c in row["checks"]]
result = {
    "cases": rows,
    "calls": len(checks),
    "passed": sum(c["ids_exist_in_rest"] for c in checks),
    "scope": "existence of returned IDs; does not prove completeness of filter or correctness of prose",
}
save(args.out, result)
print(f"REST ID checks: {result['passed']}/{result['calls']}")
