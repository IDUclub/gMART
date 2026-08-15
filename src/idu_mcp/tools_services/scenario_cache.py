import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_CACHE_DIR = "runtime/scenario_cache"


def _cache_size() -> int:
    """Number of scenarios to keep. 0 (the default) disables the cache."""

    try:
        return max(0, int(os.getenv("SCENARIO_CACHE_SIZE", "0")))
    except ValueError:
        return 0


def _cache_ttl() -> float:
    try:
        return max(0.0, float(os.getenv("SCENARIO_CACHE_TTL", "3600")))
    except ValueError:
        return 3600.0


def _cache_dir() -> Path:
    return Path(os.getenv("SCENARIO_CACHE_DIR", DEFAULT_CACHE_DIR))


class ScenarioCache:
    """Disk cache of Urban API responses, scoped to a bounded set of scenarios.

    Benchmark runs replay thousands of queries over a few dozen scenarios, so the
    same scenario layers are re-fetched hundreds of times. Keeping the hottest
    ``SCENARIO_CACHE_SIZE`` scenarios removes that repeated load; the least
    recently used scenario is dropped whole (with all of its entries) once the
    limit is exceeded.

    Entries live under ``SCENARIO_CACHE_DIR`` as gzipped JSON, one file per
    request, rather than in the process: a scenario's objects run to tens of
    megabytes as Python objects, and holding several of them resident pushed the
    host into reclaiming memory from everything else. On disk the same set costs
    a few hundred megabytes and survives a restart.

    Disabled unless ``SCENARIO_CACHE_SIZE`` is set: entries are keyed by scenario
    and request parameters only, **not** by caller identity, so it must not be
    enabled on a deployment serving several users with different data access.
    """

    def __init__(
        self,
        max_scenarios: int | None = None,
        ttl: float | None = None,
        cache_dir: Path | str | None = None,
    ):
        self.max_scenarios = _cache_size() if max_scenarios is None else max_scenarios
        self.ttl = _cache_ttl() if ttl is None else ttl
        self.root = Path(cache_dir) if cache_dir is not None else _cache_dir()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_scenarios > 0

    def _scenario_dir(self, scenario_id: int) -> Path:
        return self.root / str(scenario_id)

    @staticmethod
    def _entry_file(scenario_dir: Path, key: tuple) -> Path:
        digest = hashlib.sha256(
            json.dumps(key, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()[:32]
        return scenario_dir / f"{digest}.json.gz"

    def get(self, scenario_id: int, key: tuple) -> Any | None:
        if not self.enabled:
            return None
        path = self._entry_file(self._scenario_dir(scenario_id), key)
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            self.misses += 1
            return None
        if self.ttl and age > self.ttl:
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                value = json.load(fh)
        except (OSError, json.JSONDecodeError, EOFError) as exc:
            # A half-written or corrupted entry is a miss, never an error: the
            # cache exists to save work, and must not be able to fail a request.
            logger.warning(f"scenario cache: dropping unreadable entry {path}: {exc}")
            path.unlink(missing_ok=True)
            self.misses += 1
            return None
        self._touch(scenario_id)
        self.hits += 1
        return value

    def set(self, scenario_id: int, key: tuple, value: Any) -> None:
        if not self.enabled:
            return
        scenario_dir = self._scenario_dir(scenario_id)
        path = self._entry_file(scenario_dir, key)
        tmp = path.with_suffix(".tmp")
        try:
            scenario_dir.mkdir(parents=True, exist_ok=True)
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fh:
                json.dump(value, fh, ensure_ascii=False)
            # Readers must never see a partial file, hence the rename.
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"scenario cache: could not store {path}: {exc}")
            tmp.unlink(missing_ok=True)
            return
        self._touch(scenario_id)
        self._evict()

    def _touch(self, scenario_id: int) -> None:
        """Recency lives in the scenario directory's mtime."""

        try:
            os.utime(self._scenario_dir(scenario_id), None)
        except OSError:
            pass

    def _scenario_dirs(self) -> list[Path]:
        try:
            return [p for p in self.root.iterdir() if p.is_dir()]
        except OSError:
            return []

    def _evict(self) -> None:
        dirs = self._scenario_dirs()
        if len(dirs) <= self.max_scenarios:
            return
        for path in sorted(dirs, key=lambda p: p.stat().st_mtime)[
            : len(dirs) - self.max_scenarios
        ]:
            entries = len(list(path.glob("*.json.gz")))
            shutil.rmtree(path, ignore_errors=True)
            logger.info(
                f"scenario cache: evicted scenario {path.name} "
                f"({entries} entries), hits={self.hits} misses={self.misses}"
            )
