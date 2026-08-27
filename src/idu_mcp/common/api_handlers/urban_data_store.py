"""On-disk store of Urban API responses, for reproducible offline experiment runs.

Every Urban API read in this service goes through one place —
``JsonApiHandler._request`` — and every one of those requests is already scoped to
a single entity:

    v1/service_types?name=<X>                                     one per name
    v1/scenarios/<sid>/services_with_geometry?service_type_id=<N> one per type
    v1/scenarios/<sid>/physical_objects_with_geometry?...         one per type
    v1/scenarios/<sid>/service_types                              the catalog
    v1/scenarios/<sid>/physical_object_types                      the catalog

So keying on ``(endpoint, params)`` stores each entity separately, and *any*
combination of entity names a later query asks for is answered from pieces already
on disk. That is the difference from ``ScenarioCache``, which keys on the whole
sorted tuple of names a single call happened to ask for and therefore misses on
every combination it has not seen.

Three modes (``URBAN_DATA_MODE``):

``live`` (default)
    No store at all — production behaviour, nothing is read or written.
``record``
    Every response is served from the network and written to the store. Nothing
    expires and nothing is evicted: the store is a growing corpus, not a cache.
``replay``
    The network is never contacted. A request that is not in the store raises
    ``UrbanDataUnavailable`` rather than falling back to the network or to an
    empty result.

The reason ``replay`` raises instead of returning nothing is measurement, not
strictness: an empty layer travels through the restrictions pipeline perfectly
happily and is recorded as "the model selected no objects". A dropped VPN would
then be scored as a model failure. A distinct exception keeps a data gap out of
the model's numbers.

**Single-user only.** Like ``ScenarioCache``, the key carries no caller identity,
so two users with different data access would share entries. The store is meant
for a benchmark run on one operator's credentials and must not be enabled on a
deployment serving several users.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

DEFAULT_STORE_DIR = "runtime/urban_data"

LIVE = "live"
RECORD = "record"
REPLAY = "replay"
MODES = (LIVE, RECORD, REPLAY)

# Requests under /v1/scenarios/<id>/... are filed by scenario so a run can be
# inspected, copied or thrown away one territory at a time; everything else
# (service_types, physical_object_types) is scenario-independent.
_SCENARIO_RE = re.compile(r"/?v1/scenarios/(\d+)(?:/|$)")
_GLOBAL_BUCKET = "_global"


class UrbanDataUnavailable(RuntimeError):
    """A replayed request has no stored response.

    Carries the request that missed so the runner can record *which* data was
    absent, and so a prefetch pass can be told exactly what to fetch.
    """

    def __init__(self, endpoint: str, params: dict | None) -> None:
        self.endpoint = endpoint
        self.params = dict(params or {})
        super().__init__(
            f"Urban API data unavailable offline: {endpoint} "
            f"params={self.params} is not in the store"
        )


def store_mode(default: str = LIVE) -> str:
    """The mode from ``URBAN_DATA_MODE``; unknown values fall back to ``live``."""

    value = (os.getenv("URBAN_DATA_MODE", "") or default).strip().lower()
    if value not in MODES:
        logger.warning(
            f"URBAN_DATA_MODE={value!r} is not one of {MODES}; using {LIVE!r}"
        )
        return LIVE
    return value


def store_dir() -> Path:
    return Path(os.getenv("URBAN_DATA_DIR", DEFAULT_STORE_DIR))


class UrbanDataStore:
    """Content-addressed store of Urban API GET responses.

    Attributes:
        mode (str): one of ``live`` / ``record`` / ``replay``.
        root (Path): directory the entries live under.
        hits (int): responses served from the store.
        misses (int): lookups with nothing stored (in ``replay`` each one raises).
        writes (int): responses written.
    """

    def __init__(
        self,
        mode: str | None = None,
        root: Path | str | None = None,
    ) -> None:
        self.mode = store_mode() if mode is None else mode
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        self.root = Path(root) if root is not None else store_dir()
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @property
    def enabled(self) -> bool:
        return self.mode != LIVE

    @property
    def offline(self) -> bool:
        return self.mode == REPLAY

    # ----------------------------------------------------------------- keys --
    @staticmethod
    def _bucket(endpoint: str) -> str:
        match = _SCENARIO_RE.search(endpoint)
        return match.group(1) if match else _GLOBAL_BUCKET

    @staticmethod
    def _key(base_url: str, endpoint: str, params: dict | None) -> str:
        # Params are sorted so that a dict built in a different order still hits,
        # and values are stringified because the same id arrives as int from one
        # caller and as str from another.
        normalised = {
            "base": base_url.rstrip("/"),
            "endpoint": "/" + endpoint.strip("/"),
            "params": sorted(
                (str(k), str(v)) for k, v in (params or {}).items() if v is not None
            ),
        }
        raw = json.dumps(normalised, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _entry_path(self, base_url: str, endpoint: str, params: dict | None) -> Path:
        return (
            self.root
            / self._bucket(endpoint)
            / f"{self._key(base_url, endpoint, params)}.json.gz"
        )

    # ------------------------------------------------------------- read/write --
    def get(self, base_url: str, endpoint: str, params: dict | None) -> Any:
        """The stored response, or raise ``KeyError`` when there is none.

        ``KeyError`` rather than ``None`` because ``null`` is a response a stored
        endpoint legitimately returns, and a miss must not be confusable with it.
        """

        if not self.enabled:
            raise KeyError(endpoint)
        path = self._entry_path(base_url, endpoint, params)
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            self.misses += 1
            raise KeyError(endpoint) from None
        except (OSError, json.JSONDecodeError, EOFError) as exc:
            # A truncated entry (an interrupted run) is a miss, never an error:
            # in record mode it is simply refetched.
            logger.warning(f"urban data store: unreadable entry {path}: {exc}")
            path.unlink(missing_ok=True)
            self.misses += 1
            raise KeyError(endpoint) from None
        self.hits += 1
        return payload["response"]

    def set(
        self, base_url: str, endpoint: str, params: dict | None, response: Any
    ) -> None:
        if self.mode != RECORD:
            return
        path = self._entry_path(base_url, endpoint, params)
        tmp = path.with_suffix(".tmp")
        entry = {
            # The request is stored alongside the response so the store stays
            # auditable: the file name is a digest and tells a reader nothing.
            "base_url": base_url.rstrip("/"),
            "endpoint": "/" + endpoint.strip("/"),
            "params": {str(k): v for k, v in (params or {}).items()},
            "response": response,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as handle:
                json.dump(entry, handle, ensure_ascii=False)
            # Readers must never see a partial file, hence the rename.
            tmp.replace(path)
        except (OSError, TypeError, ValueError) as exc:
            logger.warning(f"urban data store: could not store {path}: {exc}")
            tmp.unlink(missing_ok=True)
            return
        self.writes += 1

    def has(self, base_url: str, endpoint: str, params: dict | None) -> bool:
        """Whether an entry exists, without counting a hit or a miss."""

        return self._entry_path(base_url, endpoint, params).exists()

    # ---------------------------------------------------------------- report --
    def scenarios(self) -> list[int]:
        """Scenario ids the store holds entries for."""

        try:
            return sorted(
                int(path.name)
                for path in self.root.iterdir()
                if path.is_dir() and path.name.isdigit()
            )
        except OSError:
            return []

    def stats(self) -> dict[str, Any]:
        entries = total_bytes = 0
        for path in self.root.rglob("*.json.gz"):
            entries += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        return {
            "mode": self.mode,
            "root": str(self.root),
            "entries": entries,
            "megabytes": round(total_bytes / 1024 / 1024, 1),
            "scenarios": len(self.scenarios()),
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
        }
