#!/usr/bin/env python3
"""Live progress bars for a benchmark run — redrawn in place, not appended.

Reads the result files the harness appends to, so it needs nothing from the run
itself and can be started, stopped and restarted at any time. Errors are NOT
printed here: they belong in the run log, and the bar only counts them.

    python3 benchmarks/harness/progress.py --total 2186 \
        benchmarks/data/results_main benchmarks/data/results_ablation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

NAME_WIDTH = 18
MIN_BAR = 3
CLEAR_LINE = "\033[2K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


def term_width() -> int:
    """Usable columns — one short of the real width, so nothing ever wraps."""
    return max(shutil.get_terminal_size((100, 24)).columns - 1, 20)


def draw(out: list[str], lines_drawn: int) -> int:
    """Redraw the frame in place; return how many lines it occupied.

    Every line is clipped to the terminal width: a line that wraps takes two
    physical rows while the cursor-up below counts one, so the frames would
    drift down the screen instead of overwriting each other.
    """

    out = [line[: term_width()] for line in out]
    if not sys.stdout.isatty():
        # piped into a file: escape codes would only add noise, append instead
        sys.stdout.write("\n".join(out) + "\n\n")
        sys.stdout.flush()
        return 0
    if lines_drawn:
        sys.stdout.write(f"\033[{lines_drawn}A")
    sys.stdout.write("".join(CLEAR_LINE + line + "\n" for line in out))
    sys.stdout.flush()
    return len(out)


def counts(path: Path) -> tuple[int, int]:
    """(rows, errors) in one results.jsonl, read tolerantly while it is written."""
    rows = errors = 0
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                # a truncated last line is normal — the writer may be mid-append
                try:
                    if json.loads(line).get("error"):
                        errors += 1
                except json.JSONDecodeError:
                    rows -= 1
    except FileNotFoundError:
        return 0, 0
    return rows, errors


def bar(done: int, total: int, width: int) -> str:
    width = max(width, MIN_BAR)
    filled = 0 if total <= 0 else min(width, round(width * done / total))
    return "█" * filled + "░" * (width - filled)


def human(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds or seconds > 1e6:
        return "--:--"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def stage_progress(log: str) -> tuple[str, int, int]:
    """(stage title, done, total) for the preparation stages.

    The chain spends its first hours generating the dataset, and the result files
    do not exist yet; the log's own "[i/N]" counters are the only progress signal
    there, so the watcher shows those instead of an empty screen.
    """

    title, done, total = "", 0, 0
    try:
        tail = subprocess.run(
            ["tail", "-400", log], capture_output=True, text=True
        ).stdout.splitlines()
    except Exception:  # noqa: BLE001
        return title, done, total
    for line in tail:
        if line.startswith("=== "):
            title = line.strip("= ").strip()
        m = re.search(r"\[(\d+)/(\d+)\]", line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
    return title, done, total


def is_running(pattern: str) -> bool:
    return (
        subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        ).returncode
        == 0
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result directories to watch")
    ap.add_argument("--total", type=int, default=0, help="rows expected per model")
    ap.add_argument("--dataset", default="",
                    help="infer --total from this CSV (waits for it to appear)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--watch-process", default="run_benchmark[.]py",
                    help="pgrep pattern; the bar exits when it stops matching")
    ap.add_argument("--log", default="", help="run log, last line shown as status")
    args = ap.parse_args()

    if args.dataset and not args.total:
        # the dataset may still be being generated when the watcher starts
        drawn = 0
        while not os.path.exists(args.dataset):
            title, done, total = (
                stage_progress(args.log) if args.log else ("", 0, 0)
            )
            width = term_width()
            lines = [
                "подготовка датасета — прогон ещё не начался",
                f"  {title[:width - 2] or 'стадия неизвестна'}",
                f"  {bar(done, total, width - 14)} {done:>5}/{total}"
                if total
                else "  (счётчик стадии пока не появился)",
            ]
            drawn = draw(lines, drawn)
            time.sleep(args.interval)
        import csv

        # newline="" is required: the questions contain embedded newlines, so a
        # line count would overstate the dataset (3029 lines for 1974 records)
        with open(args.dataset, encoding="utf-8", newline="") as fh:
            args.total = max(sum(1 for _ in csv.reader(fh, delimiter=";")) - 1, 1)

    started = time.time()
    first_seen: dict[str, tuple[float, int]] = {}
    lines_drawn = 0
    if sys.stdout.isatty():
        sys.stdout.write(HIDE_CURSOR)
    try:
        while True:
            files = sorted(
                f for d in args.dirs for f in Path(d).glob("*/results.jsonl")
            )
            block: list[str] = []
            total_done = 0
            width = term_width()
            for f in files:
                name = f.parent.name
                done, errors = counts(f)
                total_done += done
                # rate is measured from when this model first produced a row, so a
                # model waiting its turn does not drag the estimate down
                if name not in first_seen and done:
                    first_seen[name] = (time.time(), done)
                rate = 0.0
                if name in first_seen:
                    t0, n0 = first_seen[name]
                    elapsed = max(time.time() - t0, 1e-6)
                    rate = (done - n0) / elapsed
                left = max(args.total - done, 0)
                eta = human(left / rate) if rate > 0 else "--:--"
                pct = 100 * done / args.total if args.total else 0
                # the bar takes whatever the rest of the row leaves, so the line
                # never wraps: "{bar}" is a 5-char stand-in for its own width
                row = (
                    f"  {name[:NAME_WIDTH]:<{NAME_WIDTH}} {{bar}} "
                    f"{done:>5}/{args.total} {pct:5.1f}% {rate * 3600:5.0f}/ч "
                    f"ост. {eta:>8} ош. {errors}"
                )
                block.append(
                    row.format(bar=bar(done, args.total, width - len(row) + 5))
                )
            if not files:
                block.append("  (результатов пока нет — прогон только запускается)")

            running = is_running(args.watch_process)
            head = (
                f"прогон: {'идёт' if running else 'ЗАВЕРШЁН/ОСТАНОВЛЕН'} | "
                f"всего {total_done} строк | в работе {human(time.time() - started)}"
            )
            tail = []
            if args.log and os.path.exists(args.log):
                try:
                    last = subprocess.run(
                        ["tail", "-1", args.log], capture_output=True, text=True
                    ).stdout.strip()
                except Exception:  # noqa: BLE001
                    last = ""
                if last:
                    tail.append(f"  лог: {last[:100]}")

            lines_drawn = draw([head] + block + tail, lines_drawn)

            if not running and files and all(
                counts(f)[0] >= args.total for f in files
            ):
                break
            time.sleep(args.interval)
    except (KeyboardInterrupt, BrokenPipeError):
        # BrokenPipeError: someone piped the watcher into `head` and walked away
        pass
    finally:
        if sys.stdout.isatty():
            try:
                sys.stdout.write(SHOW_CURSOR)
                sys.stdout.flush()
            except BrokenPipeError:
                pass


if __name__ == "__main__":
    main()
