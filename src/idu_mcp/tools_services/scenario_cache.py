import os
import time
from collections import OrderedDict
from typing import Any

from loguru import logger


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


class ScenarioCache:
    """LRU cache of Urban API responses, scoped to a bounded set of scenarios.

    Benchmark runs replay thousands of queries over a few dozen scenarios, so the
    same scenario layers are re-fetched hundreds of times. Keeping the hottest
    ``SCENARIO_CACHE_SIZE`` scenarios removes that repeated load; the least
    recently used scenario is dropped whole (with all of its entries) once the
    limit is exceeded.

    Disabled unless ``SCENARIO_CACHE_SIZE`` is set: entries are keyed by scenario
    and request parameters only, **not** by caller identity, so it must not be
    enabled on a deployment serving several users with different data access.
    """

    def __init__(self, max_scenarios: int | None = None, ttl: float | None = None):
        self.max_scenarios = _cache_size() if max_scenarios is None else max_scenarios
        self.ttl = _cache_ttl() if ttl is None else ttl
        # scenario_id -> {entry_key: (stored_at, value)}; ordered by recency
        self._store: OrderedDict[int, dict[tuple, tuple[float, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_scenarios > 0

    def get(self, scenario_id: int, key: tuple) -> Any | None:
        if not self.enabled:
            return None
        entries = self._store.get(scenario_id)
        if entries is None or key not in entries:
            self.misses += 1
            return None
        stored_at, value = entries[key]
        if self.ttl and time.monotonic() - stored_at > self.ttl:
            del entries[key]
            self.misses += 1
            return None
        self._store.move_to_end(scenario_id)
        self.hits += 1
        return value

    def set(self, scenario_id: int, key: tuple, value: Any) -> None:
        if not self.enabled:
            return
        entries = self._store.setdefault(scenario_id, {})
        entries[key] = (time.monotonic(), value)
        self._store.move_to_end(scenario_id)
        while len(self._store) > self.max_scenarios:
            evicted, dropped = self._store.popitem(last=False)
            logger.info(
                f"scenario cache: evicted scenario {evicted} "
                f"({len(dropped)} entries), hits={self.hits} misses={self.misses}"
            )
