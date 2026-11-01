"""Summaries of a run: ``summary.json``, ``report.md``, ``review.md``, ``grades.csv``.

``report.md`` answers "which configuration is best": a stage x profile matrix of pass
rate, score, latency and tokens, the best profile per stage, and every failed check.
``review.md`` puts the free-text outputs of manual stages side by side per case;
``grades.csv`` is where a reviewer scores them (0-2). Re-running ``report`` folds the
filled grades back into the summary.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

GRADE_FIELDS = ["profile", "stage", "case", "repeat", "model", "grade", "comment"]


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / "results.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_grades(results_dir: Path) -> dict[tuple, float]:
    path = results_dir / "grades.csv"
    if not path.exists():
        return {}
    grades = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("grade") or "").strip().replace(",", ".")
            if not raw:
                continue
            key = (row["profile"], row["stage"], row["case"], str(row["repeat"]))
            grades[key] = float(raw)
    return grades


def summarize(
    results: list[dict[str, Any]], grades: dict[tuple, float] | None = None
) -> dict[str, Any]:
    grades = grades or {}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        if result.get("status") == "skipped":
            continue
        groups[(result["profile"], result["stage"])].append(result)

    cells = []
    for (profile, stage), items in sorted(groups.items()):
        latencies = [item.get("latency_s") or 0.0 for item in items]
        by_case: dict[str, set[str]] = defaultdict(set)
        for item in items:
            by_case[item["case"]].add(item["status"])
        repeats = max(item.get("repeat", 1) for item in items)
        stable = (
            sum(len(statuses) == 1 for statuses in by_case.values()) / len(by_case)
            if repeats > 1
            else None
        )
        graded = [
            grades[key]
            for item in items
            if (
                key := (
                    item["profile"],
                    item["stage"],
                    item["case"],
                    str(item.get("repeat", 1)),
                )
            )
            in grades
        ]
        cells.append(
            {
                "profile": profile,
                "stage": stage,
                "kind": items[0].get("kind"),
                "model": items[0].get("model"),
                "runs": len(items),
                "pass": sum(item["status"] == "pass" for item in items),
                "fail": sum(item["status"] == "fail" for item in items),
                "error": sum(item["status"] == "error" for item in items),
                "pass_rate": sum(item["status"] == "pass" for item in items)
                / len(items),
                "mean_score": _mean([item.get("score") or 0.0 for item in items]),
                "mean_latency_s": _mean(latencies),
                "p95_latency_s": _p95(latencies),
                "mean_input_tokens": _mean(
                    [item.get("input_tokens") or 0 for item in items]
                ),
                "mean_output_tokens": _mean(
                    [item.get("output_tokens") or 0 for item in items]
                ),
                "mean_llm_calls": _mean([item.get("llm_calls") or 0 for item in items]),
                "stability": stable,
                "graded": len(graded),
                "mean_grade": _mean(graded) if graded else None,
            }
        )

    profiles: dict[str, dict[str, Any]] = {}
    for profile in sorted({cell["profile"] for cell in cells}):
        auto = [c for c in cells if c["profile"] == profile and c["kind"] == "auto"]
        runs = sum(c["runs"] for c in auto)
        profiles[profile] = {
            "auto_runs": runs,
            "auto_pass_rate": (sum(c["pass"] for c in auto) / runs) if runs else None,
            # Every stage weighs the same, however many cases it has.
            "stage_macro_score": (
                _mean([c["mean_score"] for c in auto]) if auto else None
            ),
            "errors": sum(c["error"] for c in cells if c["profile"] == profile),
            "total_latency_s": sum(
                c["mean_latency_s"] * c["runs"]
                for c in cells
                if c["profile"] == profile
            ),
        }

    best: dict[str, dict[str, Any]] = {}
    for stage in sorted({cell["stage"] for cell in cells}):
        candidates = [c for c in cells if c["stage"] == stage]
        if candidates[0]["kind"] == "manual":
            graded = [c for c in candidates if c["mean_grade"] is not None]
            if not graded:
                continue
            top = max(graded, key=lambda c: (c["mean_grade"], -c["mean_latency_s"]))
        else:
            top = max(
                candidates,
                key=lambda c: (c["pass_rate"], c["mean_score"], -c["mean_latency_s"]),
            )
        best[stage] = {
            "profile": top["profile"],
            "model": top["model"],
            "pass_rate": top["pass_rate"],
            "mean_score": top["mean_score"],
            "mean_grade": top["mean_grade"],
            "mean_latency_s": top["mean_latency_s"],
        }
    return {"cells": cells, "profiles": profiles, "best_per_stage": best}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _short(value: Any, limit: int = 160) -> str:
    text = (
        json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    )
    text = text.replace("\n", " ").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_report(
    summary: dict[str, Any], results: list[dict[str, Any]], meta: dict
) -> str:
    lines = [f"# Бенчмарк LLM-этапов gMART — {meta.get('started', '')}", ""]
    lines += [
        f"Профили: {', '.join(meta.get('profiles', []))} · кейсов: {meta.get('cases', 0)} · "
        f"повторов: {meta.get('repeat', 1)} · температура: {meta.get('temperature')}",
        "",
        "## Итог по профилям (автоматически оцениваемые этапы)",
        "",
        "| Профиль | Прогонов | Pass rate | Средний балл этапов | Ошибок | Суммарное время, с |",
        "|---|---|---|---|---|---|",
    ]
    for name, data in summary["profiles"].items():
        lines.append(
            f"| {name} | {data['auto_runs']} | {_pct(data['auto_pass_rate'])} | "
            f"{_pct(data['stage_macro_score'])} | {data['errors']} | "
            f"{data['total_latency_s']:.0f} |"
        )

    lines += ["", "## Лучший профиль по этапам", ""]
    lines += [
        "| Этап | Профиль | Модель | Pass rate | Балл | Оценка (ручн.) | Латентность, с |"
    ]
    lines += ["|---|---|---|---|---|---|---|"]
    for stage, top in summary["best_per_stage"].items():
        grade = "—" if top["mean_grade"] is None else f"{top['mean_grade']:.2f}"
        lines.append(
            f"| {stage} | {top['profile']} | {top['model']} | {_pct(top['pass_rate'])} | "
            f"{_pct(top['mean_score'])} | {grade} | {top['mean_latency_s']:.1f} |"
        )

    lines += ["", "## Матрица этап × профиль", ""]
    lines += [
        "| Этап | Тип | Профиль | Модель | Pass | Fail | Err | Балл | Стаб. | "
        "Лат. ср/p95, с | Токены вх/вых | Вызовов | Оценка |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell in sorted(summary["cells"], key=lambda c: (c["stage"], c["profile"])):
        grade = (
            "—"
            if cell["mean_grade"] is None
            else f"{cell['mean_grade']:.2f} ({cell['graded']})"
        )
        lines.append(
            f"| {cell['stage']} | {cell['kind']} | {cell['profile']} | {cell['model']} | "
            f"{cell['pass']} | {cell['fail']} | {cell['error']} | {_pct(cell['mean_score'])} | "
            f"{_pct(cell['stability'])} | {cell['mean_latency_s']:.1f}/{cell['p95_latency_s']:.1f} | "
            f"{cell['mean_input_tokens']:.0f}/{cell['mean_output_tokens']:.0f} | "
            f"{cell['mean_llm_calls']:.1f} | {grade} |"
        )

    failures = [r for r in results if r.get("status") in ("fail", "error")]
    lines += ["", f"## Провалы и ошибки ({len(failures)})", ""]
    for result in sorted(
        failures,
        key=lambda r: (r["stage"], r["case"], r["profile"], r.get("repeat", 1)),
    ):
        head = (
            f"- **{result['stage']} / {result['case']}** · {result['profile']} "
            f"#{result.get('repeat', 1)} · {result['status']}"
        )
        if result.get("error"):
            lines.append(f"{head} — `{_short(result['error'], 300)}`")
            continue
        lines.append(head)
        for check in result.get("checks") or []:
            if not check["passed"]:
                lines.append(
                    f"  - ✗ `{check['name']}`: ожидалось `{_short(check['expected'])}`, "
                    f"получено `{_short(check['actual'])}`"
                )
    lines.append("")
    return "\n".join(lines)


def render_review(results: list[dict[str, Any]]) -> str:
    manual = [r for r in results if r.get("needs_review")]
    by_case: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in manual:
        by_case[(result["stage"], result["case"])].append(result)
    lines = [
        "# Ручная проверка ответов",
        "",
        "Шкала (колонка `grade` в grades.csv): **2** — верно и полно, без выдумок; "
        "**1** — в целом верно, есть пропуски или стилистические нарушения правил; "
        "**0** — выдумка, ошибка по сути или нарушение запрета.",
        "",
    ]
    for (stage, case), items in sorted(by_case.items()):
        first = items[0]
        lines += [f"## {stage} / {case}", ""]
        if first.get("review"):
            lines += [f"**Что проверить:** {first['review'].strip()}", ""]
        lines += [
            "<details><summary>Вход</summary>",
            "",
            "```json",
            json.dumps(first.get("input"), ensure_ascii=False, indent=2),
            "```",
            "",
            "</details>",
            "",
        ]
        for item in sorted(items, key=lambda r: (r["profile"], r.get("repeat", 1))):
            failed = [c["name"] for c in item.get("checks") or [] if not c["passed"]]
            status = item["status"] + (
                f" (формальные: ✗ {', '.join(failed)})" if failed else ""
            )
            lines += [
                f"### {item['profile']} · {item.get('model')} · #{item.get('repeat', 1)} · {status} · "
                f"{item.get('latency_s', 0):.1f} с",
                "",
            ]
            if item.get("error"):
                lines += [f"Ошибка: `{item['error']}`", ""]
                continue
            output = item.get("output")
            text = output.get("text") if isinstance(output, dict) else None
            if isinstance(text, str):
                lines += [text.strip() or "_(пустой ответ)_", ""]
            else:
                lines += [
                    "```json",
                    json.dumps(output, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
        lines.append("")
    return "\n".join(lines)


def write_grades_template(results_dir: Path, results: list[dict[str, Any]]) -> None:
    path = results_dir / "grades.csv"
    existing = load_grades(results_dir)
    rows = [
        {
            "profile": r["profile"],
            "stage": r["stage"],
            "case": r["case"],
            "repeat": r.get("repeat", 1),
            "model": r.get("model"),
            "grade": existing.get(
                (r["profile"], r["stage"], r["case"], str(r.get("repeat", 1))), ""
            ),
            "comment": "",
        }
        for r in sorted(
            (
                r
                for r in results
                if r.get("needs_review") and r.get("status") != "error"
            ),
            key=lambda r: (r["stage"], r["case"], r["profile"], r.get("repeat", 1)),
        )
    ]
    if not rows or (path.exists() and existing):
        # Never overwrite grades someone already entered.
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GRADE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(results_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    results = load_results(results_dir)
    write_grades_template(results_dir, results)
    summary = summarize(results, load_grades(results_dir))
    (results_dir / "summary.json").write_text(
        json.dumps({"meta": meta, **summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (results_dir / "report.md").write_text(
        render_report(summary, results, meta), encoding="utf-8"
    )
    review = render_review(results)
    if any(r.get("needs_review") for r in results):
        (results_dir / "review.md").write_text(review, encoding="utf-8")
    return summary
