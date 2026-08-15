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
from types import FrameType

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


def _describe(obj: object) -> str:
    """A referrer named well enough to find in the source."""

    kind = type(obj).__name__
    if isinstance(obj, dict):
        keys = [k for k in list(obj)[:6] if isinstance(k, str)]
        return f"dict(keys={keys})" if keys else "dict"
    if isinstance(obj, (list, tuple, set)):
        return f"{kind}(len={len(obj)})"
    module = getattr(type(obj), "__module__", "")
    return f"{module}.{kind}" if module else kind


def _is_own(obj: object) -> bool:
    """This module's own frames refer to everything it inspects."""

    return isinstance(obj, FrameType) and obj.f_globals.get("__name__") == __name__


def _owner_chain(obj: object, mine: set[int], max_depth: int = 8) -> list[str]:
    """Walk referrers up to the first thing that is not a plain container.

    Containers say nothing on their own — a list inside a dict inside a tuple is
    every payload in the process. The frame or object that owns them is the
    answer, so the walk continues until it finds one, and reports the trail.
    """

    chain: list[str] = []
    current = obj
    seen = {id(obj)}
    # gc.get_referrers builds a fresh list that refers to what it found, so the
    # walk would otherwise climb its own scaffolding. The lists are kept alive
    # deliberately: freeing them would let their ids be reused by real objects.
    scratch: list[list] = []
    for _ in range(max_depth):
        found = gc.get_referrers(current)
        scratch.append(found)
        skip = mine | {id(item) for item in scratch}
        referrers = [
            r
            for r in found
            if id(r) not in seen and id(r) not in skip and not _is_own(r)
        ]
        scratch.append(referrers)
        if not referrers:
            break
        current = referrers[0]
        seen.add(id(current))
        if isinstance(current, FrameType):
            code = current.f_code
            chain.append(
                f"frame {code.co_filename}:{code.co_firstlineno} {code.co_name}"
            )
            break
        chain.append(_describe(current))
        if not isinstance(current, (list, dict, tuple, set)):
            break
    return chain


def retention_report(sample: int = 6) -> dict:
    """What is holding the largest containers alive.

    Flat object counts under a high RSS mean the allocator; counts in the
    millions mean something keeps references, and this says what: for the
    biggest containers it names the first real owner up the referrer chain — a
    frame, a session, a cache — rather than the anonymous containers between.
    """

    gc.collect()
    biggest: list[tuple[int, object]] = []
    for obj in gc.get_objects():
        if not isinstance(obj, (list, dict)):
            continue
        try:
            size = len(obj)
        except Exception:  # noqa: BLE001
            continue
        if size > 1000:
            biggest.append((size, obj))
    biggest.sort(key=lambda item: item[0], reverse=True)

    # This function's own bookkeeping refers to every object it reports on, so
    # the walk has to be told to step over it.
    report: list[dict] = []
    mine = {id(biggest), id(report)} | {id(item) for item in biggest}
    for size, obj in biggest[:sample]:
        report.append(
            {
                "type": type(obj).__name__,
                "size": size,
                "owner_chain": _owner_chain(obj, mine),
            }
        )
    return {
        "rss_mb": round(rss_bytes() / 1024 / 1024, 1),
        "big_containers": len(biggest),
        "biggest": report,
    }


def count_types(names: list[str]) -> dict[str, int]:
    """How many instances of each named type are alive.

    The referrer walk says the payloads hang off MCP protocol objects; this says
    whether those come with a session that was never torn down (transports keep
    climbing) or with buffered messages inside a live session (only the message
    types climb). The two have different fixes.
    """

    gc.collect()
    wanted = set(names)
    counts: Counter[str] = Counter({name: 0 for name in names})
    for obj in gc.get_objects():
        try:
            name = type(obj).__name__
        except Exception:  # noqa: BLE001
            continue
        if name in wanted:
            counts[name] += 1
    return dict(counts)


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
