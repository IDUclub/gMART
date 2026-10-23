"""
Daily size-based rotation of the application log file.

The log file is opened in append mode by every uvicorn worker, so renaming it
would leave the other workers writing into the renamed file. Rotation is done
copy-truncate style instead: once a day the file is checked, and if it has grown
past ``ROTATE_BYTES`` its contents are gzipped into a timestamped archive next
to it and the file is truncated in place. Writers keep their handles and carry
on at the new end of the file. Archives are pruned oldest first so the log file
and its archives together stay within ``MAX_TOTAL_BYTES``.
"""

import glob
import gzip
import os
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from loguru import logger

try:
    import fcntl
except ImportError:  # Windows: no inter-process lock, local runs use one process
    fcntl = None

MAX_TOTAL_BYTES = 100 * 1024 * 1024
ROTATE_BYTES = MAX_TOTAL_BYTES // 2
CHECK_INTERVAL_SECONDS = 24 * 60 * 60

_started_paths: set[Path] = set()
_started_lock = threading.Lock()


def start_daily_log_rotation(
    log_path: Path, interval_seconds: float = CHECK_INTERVAL_SECONDS
) -> None:
    """
    Check the log file at startup and then once per interval in a daemon thread.

    Calling it again for the same file is a no-op, so it is safe to call from
    logger configuration that runs more than once.

    Args:
        log_path (Path): Log file to rotate.
        interval_seconds (float): Delay between checks.
    """

    log_path = Path(log_path).resolve()
    with _started_lock:
        if log_path in _started_paths:
            return
        _started_paths.add(log_path)
    threading.Thread(
        target=_rotation_loop,
        args=(log_path, interval_seconds),
        name="log-rotation",
        daemon=True,
    ).start()


def rotate_log_if_needed(
    log_path: Path,
    rotate_bytes: int = ROTATE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> Path | None:
    """
    Archive and truncate the log file if it is too large, then prune archives.

    Args:
        log_path (Path): Log file to rotate.
        rotate_bytes (int): Size above which the file is archived and truncated.
        max_total_bytes (int): Limit for the log file and its archives together.
    Returns:
        Path | None: The new archive, or None when the file was not rotated.
    """

    log_path = Path(log_path)
    with _exclusive(log_path):
        archive = None
        if log_path.exists() and log_path.stat().st_size > rotate_bytes:
            archive = _archive_and_truncate(log_path)
        _prune_archives(log_path, rotate_bytes, max_total_bytes)
    return archive


def _rotation_loop(log_path: Path, interval_seconds: float) -> None:
    while True:
        try:
            archive = rotate_log_if_needed(log_path)
            if archive is not None:
                logger.info(f"Rotated log file {log_path} into {archive}")
        except Exception as exc:
            logger.warning(f"Could not rotate log file {log_path}: {exc}")
        time.sleep(interval_seconds)


@contextmanager
def _exclusive(log_path: Path) -> Iterator[None]:
    """Serialise rotation between the worker processes sharing the log file."""

    if fcntl is None:
        yield
        return
    lock_path = log_path.with_name(f"{log_path.name}.lock")
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _archive_and_truncate(log_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive = log_path.with_name(f"{log_path.name}.{stamp}.gz")
    with open(log_path, "rb") as source, gzip.open(archive, "wb") as target:
        shutil.copyfileobj(source, target)
        # Truncate right after reading to the end, before the archive is flushed,
        # to keep the window in which concurrent writes are lost as short as possible.
        os.truncate(log_path, 0)
    return archive


def _prune_archives(log_path: Path, rotate_bytes: int, max_total_bytes: int) -> None:
    """Keep the newest archives that fit next to a log file of ``rotate_bytes``."""

    current_size = log_path.stat().st_size if log_path.exists() else 0
    budget = max_total_bytes - max(current_size, rotate_bytes)
    pattern = f"{glob.escape(log_path.name)}.*.gz"
    archives = sorted(log_path.parent.glob(pattern), reverse=True)
    for archive in archives:
        size = archive.stat().st_size
        if size <= budget:
            budget -= size
        else:
            budget = 0
            archive.unlink(missing_ok=True)
