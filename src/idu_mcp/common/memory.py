"""Memory hygiene for the geometry tools, and a way to see what is holding on.

A single restrictions call turns a scenario's layers into GeoDataFrames, joins
them and serialises the result — hundreds of megabytes of short-lived objects,
in a worker thread. Two things then keep RSS high after the call:

* CPython frees the objects, but glibc keeps the freed blocks in the arena the
  thread allocated them from and does not return them to the OS. Every worker
  thread gets its own arena, so the high-water mark is per thread and the
  process only ever grows. ``malloc_trim`` is what hands those blocks back.
* Reference cycles (GeoDataFrames hold plenty) survive until the collector runs,
  which under steady load can be a long while.

``release_memory`` does both and is cheap next to the work it follows: a
collection over a few hundred thousand objects costs milliseconds against
seconds of geometry.
"""

import ctypes
import gc
import os
from collections import Counter

from loguru import logger

_LIBC: ctypes.CDLL | None = None
_TRIM_UNAVAILABLE = False


def _libc() -> ctypes.CDLL | None:
    """glibc, when we are on it — musl has no malloc_trim."""

    global _LIBC, _TRIM_UNAVAILABLE
    if _LIBC is not None or _TRIM_UNAVAILABLE:
        return _LIBC
    try:
        candidate = ctypes.CDLL("libc.so.6")
        candidate.malloc_trim
    except (OSError, AttributeError) as exc:
        _TRIM_UNAVAILABLE = True
        logger.info(f"malloc_trim unavailable, skipping arena release: {exc}")
        return None
    _LIBC = candidate
    return _LIBC


def release_memory() -> None:
    """Collect cycles and hand freed arenas back to the OS."""

    if os.getenv("MCP_RELEASE_MEMORY", "1").strip().lower() in {"0", "false", "no"}:
        return
    gc.collect()
    libc = _libc()
    if libc is not None:
        libc.malloc_trim(0)


def rss_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def memory_snapshot(top: int = 15) -> dict:
    """RSS and the most numerous live object types.

    Watching the counts between two calls is what tells a leak (a type that
    keeps growing) apart from a working set the allocator simply has not
    returned (counts flat, RSS high).
    """

    gc.collect()
    counts: Counter[str] = Counter()
    for obj in gc.get_objects():
        try:
            counts[type(obj).__name__] += 1
        except Exception:  # noqa: BLE001 — some objects break on type access
            continue
    return {
        "rss_mb": round(rss_bytes() / 1024 / 1024, 1),
        "objects": sum(counts.values()),
        "gc_counts": gc.get_count(),
        "top_types": dict(counts.most_common(top)),
    }
