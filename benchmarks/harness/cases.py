"""Benchmark cases: ``cases/*.yaml`` of the standalone benchmark checkout.

One file per stage::

    stage: provision.plan
    defaults:            # merged under every case's input
      services_catalog: [Школа, Поликлиника]
    cases:
      - id: single_service_provision
        tags: [smoke]
        input: {user_query: "Какая обеспеченность школами?"}
        expect:
          - {path: mode, eq: provision}
        review: "что проверить глазами (только для manual-этапов)"

A case may set its own ``stage`` to override the file's. The export reads only
``id``, ``stage`` and ``input``; ``expect`` and ``review`` are the runner's.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Case:
    id: str
    stage: str
    input: dict[str, Any]
    expect: list[dict[str, Any]] = field(default_factory=list)
    tags: tuple[str, ...] = ()
    review: str | None = None
    source: str = ""

    @property
    def key(self) -> str:
        return f"{self.stage}/{self.id}"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_cases(directory: Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for path in sorted(directory.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        defaults = raw.get("defaults") or {}
        for item in raw.get("cases") or []:
            stage_id = item.get("stage") or raw.get("stage")
            if not stage_id or not item.get("id"):
                raise ValueError(f"{path.name}: every case needs an id and a stage")
            case = Case(
                id=str(item["id"]),
                stage=str(stage_id),
                input=_merge(defaults, item.get("input") or {}),
                expect=list(item.get("expect") or []),
                tags=tuple(item.get("tags") or ()),
                review=item.get("review"),
                source=path.name,
            )
            if case.key in seen:
                raise ValueError(f"{path.name}: duplicate case {case.key}")
            seen.add(case.key)
            cases.append(case)
    return cases


def select_cases(
    cases: list[Case],
    *,
    stages: list[str] | None = None,
    ids: list[str] | None = None,
    tags: list[str] | None = None,
) -> list[Case]:
    def matches(value: str, patterns: list[str] | None) -> bool:
        return not patterns or any(fnmatch.fnmatchcase(value, p) for p in patterns)

    return [
        case
        for case in cases
        if matches(case.stage, stages)
        and (matches(case.id, ids) or matches(case.key, ids))
        and (not tags or set(tags) & set(case.tags))
    ]
