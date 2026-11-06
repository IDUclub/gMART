"""Command line: ``python -m benchmarks {list,export}``.

The benchmark itself lives in its own repository (gmart-llm-bench) and runs without
gMART. This package only freezes the requests gMART's stages send to the model:

python -m benchmarks list --bench ../gmart-llm-bench
python -m benchmarks export --bench ../gmart-llm-bench
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from loguru import logger

from benchmarks.harness.cases import load_cases, select_cases
from benchmarks.harness.export import export
from benchmarks.harness.stage import load_stages

GMART_ROOT = Path(__file__).resolve().parent.parent


def _split(value: str | None) -> list[str] | None:
    return (
        [item.strip() for item in value.split(",") if item.strip()] if value else None
    )


def cmd_list(args: argparse.Namespace) -> int:
    stages = load_stages()
    counts = Counter(case.stage for case in load_cases(Path(args.bench) / "cases"))
    for stage_id, stage in sorted(stages.items()):
        print(
            f"{stage_id:28} {stage.kind:6} {counts.get(stage_id, 0):3} кейсов  {stage.description}"
        )
    unknown = sorted(set(counts) - set(stages))
    if unknown:
        print(f"\nКейсы для незарегистрированных этапов: {unknown}")
        return 1
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    bench = Path(args.bench)
    stages = load_stages()
    cases = select_cases(load_cases(bench / "cases"), stages=_split(args.stages))
    unknown = sorted({case.stage for case in cases} - set(stages))
    if unknown:
        logger.error(f"Кейсы для незарегистрированных этапов: {unknown}")
        return 2
    manifest = export(
        stages,
        cases,
        bench / "data",
        gmart_root=GMART_ROOT,
        context_window=args.context_window,
    )
    logger.info(
        f"Запросов: {manifest['requests']} из {len(cases)} кейсов → {bench / 'data'}"
    )
    for key in manifest["decided_by_code"]:
        logger.warning(f"{key}: решено кодом без модели — кейс не войдёт в датасет")
    for key, error in manifest["errors"].items():
        logger.error(f"{key}: {error}")
    if manifest["gmart"]["dirty"]:
        logger.warning("В рабочей копии gMART есть незакоммиченные изменения")
    return 1 if manifest["errors"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="этапы и число кейсов")
    listing.add_argument("--bench", required=True, help="каталог gmart-llm-bench")
    listing.set_defaults(fn=cmd_list)

    exporting = sub.add_parser("export", help="заморозить запросы этапов в датасет")
    exporting.add_argument("--bench", required=True, help="каталог gmart-llm-bench")
    exporting.add_argument("--stages", help="glob этапов через запятую")
    exporting.add_argument(
        "--context-window",
        type=int,
        default=32768,
        help="окно модели, которым этапы считают бюджет ответа",
    )
    exporting.set_defaults(fn=cmd_export)

    args = parser.parse_args(argv)
    # Windows consoles default to a legacy code page; the output is Russian.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logger.remove()
    # The exporter's own progress, and only warnings from the gMART code it drives.
    logger.add(
        sys.stderr,
        level="INFO",
        filter=lambda record: record["name"].startswith(("benchmarks", "__main__"))
        or record["level"].no >= 30,
    )
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
