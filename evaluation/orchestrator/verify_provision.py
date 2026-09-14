"""Compare live numerical answer fields with captured Effects results.

This verifies faithful reporting, not the correctness of the Effects algorithm.
"""

import argparse
import json
from pathlib import Path

from run_planner import save


def verify(trace):
    chunks = []
    for event in trace["events"]:
        if event["type"] != "step_event":
            continue
        wrapper = event["content"]
        child = wrapper["event"]
        if wrapper["agent"] == "provision" and child["type"] == "chunk":
            chunks.append(child["content"].get("text", ""))
    answer = "".join(chunks)
    checks = []
    labels = {
        "services_count": "Объектов сервиса в расчёте",
        "average_provision_value": "Средняя обеспеченность по зданиям",
        "median_provision_value": "Медианная обеспеченность по зданиям",
    }
    for call in trace["tools"].get("effects", []):
        if call["args"][0] != "CalculateServicesProvision":
            continue
        for service in (call.get("result") or {}).get("services", {}).values():
            summary = service.get("summary") or {}
            marker = f"Текущая обеспеченность сервисом «{service.get('name', '')}»:"
            section = (
                answer.split(marker, 1)[1].split(
                    "Текущая обеспеченность сервисом «", 1
                )[0]
                if marker in answer
                else ""
            )
            for key, label in labels.items():
                value = summary.get(key)
                if value is not None:
                    expected = round(value, 3) if isinstance(value, float) else value
                    checks.append(
                        {
                            "field": key,
                            "expected": expected,
                            "passed": f"{label}: {expected}" in section,
                        }
                    )
            demand = summary.get("total_demand")
            within = summary.get("satisfied_demand_within")
            if (
                isinstance(demand, (int, float))
                and demand > 0
                and isinstance(within, (int, float))
            ):
                percent = f"{within / demand * 100:.1f}%"
                checks.append(
                    {
                        "field": "population_coverage",
                        "expected": percent,
                        "passed": f"Доля суммарного спроса, удовлетворённого в нормативной доступности: {percent}"
                        in section,
                    }
                )
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for folder in args.runs:
        for path in sorted(folder.glob("*.json")):
            trace = json.loads(path.read_text(encoding="utf-8"))
            checks = verify(trace)
            if checks:
                rows.append({"id": trace["id"], "checks": checks})
    checks = [check for row in rows for check in row["checks"]]
    result = {
        "cases": rows,
        "checks": len(checks),
        "passed": sum(check["passed"] for check in checks),
        "scope": "reporting Effects values; excludes missing calculations and effects mode",
    }
    save(args.out, result)
    print(
        f"Numerical fields: {result['passed']}/{result['checks']} in {len(rows)} traces"
    )
