"""Command line: ``python -m benchmarks {run,report,list}``.

python -m benchmarks list
python -m benchmarks run --config benchmarks/config.yaml --models qwen27b,gptoss
python -m benchmarks run --config benchmarks/config.yaml --profiles mixed --stages "orchestrator.*"
python -m benchmarks report benchmarks/results/<run>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from loguru import logger

from benchmarks.harness.cases import load_cases, select_cases
from benchmarks.harness.config import load_config
from benchmarks.harness.report import write_reports
from benchmarks.harness.runner import Runner
from benchmarks.harness.stage import load_stages

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _split(value: str | None) -> list[str] | None:
    return (
        [item.strip() for item in value.split(",") if item.strip()] if value else None
    )


def cmd_list(_: argparse.Namespace) -> int:
    stages = load_stages()
    counts = Counter(case.stage for case in load_cases())
    for stage_id, stage in sorted(stages.items()):
        print(
            f"{stage_id:28} {stage.kind:6} {counts.get(stage_id, 0):3} кейсов  {stage.description}"
        )
    unknown = sorted(set(counts) - set(stages))
    if unknown:
        print(f"\nКейсы для незарегистрированных этапов: {unknown}")
        return 1
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.repeat:
        config.repeat = args.repeat
    if args.concurrency:
        config.concurrency = args.concurrency
    profile_names = _split(args.profiles) or []
    if args.models:
        profile_names += config.add_single_model_profiles(_split(args.models))
    profiles = config.select_profiles(profile_names or None)
    if not profiles:
        logger.error("Нет профилей: задайте profiles в конфиге или --models")
        return 2

    stages = load_stages()
    cases = select_cases(
        load_cases(),
        stages=_split(args.stages) or config.stages,
        ids=_split(args.cases),
        tags=_split(args.tags) or config.tags or None,
    )
    unknown = sorted({case.stage for case in cases} - set(stages))
    if unknown:
        logger.error(f"Кейсы для незарегистрированных этапов: {unknown}")
        return 2
    if not cases:
        logger.error("Ни один кейс не подошёл под фильтры")
        return 2

    started = datetime.now()
    out_dir = (
        Path(args.out)
        if args.out
        else RESULTS_DIR
        / (started.strftime("%Y%m%d-%H%M%S") + (f"-{args.label}" if args.label else ""))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "started": started.isoformat(timespec="seconds"),
        "config": str(args.config),
        "profiles": [profile.name for profile in profiles],
        "profile_models": {
            profile.name: {"default": profile.default, "stages": profile.stages}
            for profile in profiles
        },
        "models": {
            name: {
                "endpoint": model.endpoint,
                "model": model.model,
                "options": model.options,
            }
            for name, model in config.models.items()
        },
        "cases": len(cases),
        "repeat": config.repeat,
        "temperature": config.temperature,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"{len(cases)} кейсов × {len(profiles)} профилей × {config.repeat} → {out_dir}"
    )
    runner = Runner(
        config,
        stages,
        save_prompts=args.save_prompts,
        case_timeout_s=args.case_timeout,
    )
    asyncio.run(runner.run(cases, profiles, out_dir / "results.jsonl"))
    summary = write_reports(out_dir, meta)
    for name, data in summary["profiles"].items():
        rate = data["auto_pass_rate"]
        logger.info(
            f"{name}: pass rate {'—' if rate is None else f'{rate:.0%}'}, "
            f"ошибок {data['errors']}"
        )
    logger.info(f"Отчёт: {out_dir / 'report.md'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir)
    meta_path = results_dir / "meta.json"
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    write_reports(results_dir, meta)
    logger.info(f"Отчёт обновлён: {results_dir / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="этапы и число кейсов").set_defaults(fn=cmd_list)

    run = sub.add_parser("run", help="прогнать кейсы")
    run.add_argument("--config", required=True)
    run.add_argument("--profiles", help="профили из конфига через запятую")
    run.add_argument("--models", help="модели из конфига: по профилю на каждую")
    run.add_argument("--stages", help="glob этапов через запятую (orchestrator.*)")
    run.add_argument("--cases", help="glob id кейсов через запятую")
    run.add_argument("--tags", help="теги кейсов через запятую")
    run.add_argument("--repeat", type=int)
    run.add_argument("--concurrency", type=int)
    run.add_argument("--case-timeout", type=float, default=900.0)
    run.add_argument(
        "--save-prompts", action="store_true", help="сохранять промпты и схемы"
    )
    run.add_argument("--label", help="суффикс каталога результатов")
    run.add_argument("--out", help="каталог результатов")
    run.set_defaults(fn=cmd_run)

    report = sub.add_parser("report", help="пересобрать отчёт (с оценками grades.csv)")
    report.add_argument("results_dir")
    report.set_defaults(fn=cmd_report)

    args = parser.parse_args(argv)
    # Windows consoles default to a legacy code page; the report text is Russian.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logger.remove()
    # The benchmark's own progress, and only warnings from the gMART code it drives.
    logger.add(
        sys.stderr,
        level="INFO",
        filter=lambda record: record["name"].startswith("benchmarks")
        or record["level"].no >= 30,
    )
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
