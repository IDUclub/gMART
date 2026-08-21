"""Resolve CheckPlan entities against global canonical Urban API dictionaries."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic

from src.agents.services.restriction_catalog import normalize_name
from src.agents.services.service_entities.compliance import (
    DeclaredRequirements,
    ResolvedRequirement,
)


@dataclass(frozen=True)
class CatalogResolution:
    requirements: DeclaredRequirements
    unresolved: list[ResolvedRequirement] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def executable(self) -> bool:
        return not self.missing


class ComplianceCatalogResolver:
    """Validate entity names without conflating type existence with scenario data.

    Scenario catalogs contain only types that currently have instances and therefore
    cannot distinguish an unknown name from a canonical type with zero objects. The
    resolver uses global Urban API dictionaries; scenario availability is evaluated
    later from complete FeatureCollections.
    """

    _ENTITY_TYPES = ("service", "physical_object")

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._cache: dict[tuple[str, str], tuple[float, str | None]] = {}
        self._lock = asyncio.Lock()

    async def resolve(
        self,
        mcp_client,
        scenario_id: int,
        requirements: DeclaredRequirements,
    ) -> CatalogResolution:
        del scenario_id  # Canonical dictionaries are scenario-independent.
        requested = {
            entity_type: list(
                dict.fromkeys(
                    item.entity
                    for item in requirements.layers
                    if item.entity_type.value == entity_type
                )
            )
            for entity_type in self._ENTITY_TYPES
        }
        try:
            catalogs = await self._resolve_types(mcp_client, requested)
        except Exception as exc:
            return CatalogResolution(
                requirements=requirements,
                missing=["catalog:urban_api:unavailable"],
                warnings=[f"urban_catalog_lookup_failed:{type(exc).__name__}"],
            )

        canonical_layers = []
        unresolved = []
        missing = []
        for requirement in requirements.layers:
            catalog = catalogs.get(requirement.entity_type.value)
            if catalog is None:
                canonical_layers.append(requirement)
                continue
            canonical_name = catalog.get(normalize_name(requirement.entity))
            if canonical_name is None:
                unresolved.append(
                    ResolvedRequirement(
                        role=requirement.role,
                        requirement_type="layer",
                        resolved=False,
                        reason="urban_catalog_entity_not_found",
                    )
                )
                if requirement.required:
                    missing.append(
                        "catalog:"
                        f"{requirement.role}:{requirement.entity_type.value}:not_found"
                    )
                canonical_layers.append(requirement)
                continue
            canonical_layers.append(
                requirement.model_copy(update={"entity": canonical_name})
            )

        return CatalogResolution(
            requirements=requirements.model_copy(update={"layers": canonical_layers}),
            unresolved=unresolved,
            missing=missing,
        )

    async def _resolve_types(
        self,
        mcp_client,
        requested: dict[str, list[str]],
    ) -> dict[str, dict[str, str | None]]:
        now = monotonic()
        catalogs: dict[str, dict[str, str | None]] = {
            entity_type: {} for entity_type in self._ENTITY_TYPES
        }
        missing: dict[str, list[str]] = {
            entity_type: [] for entity_type in self._ENTITY_TYPES
        }
        for entity_type, names in requested.items():
            for name in names:
                normalized = normalize_name(name)
                cached = self._cache.get((entity_type, normalized))
                if cached and cached[0] > now:
                    catalogs[entity_type][normalized] = cached[1]
                else:
                    missing[entity_type].append(name)

        if not any(missing.values()):
            return catalogs

        async with self._lock:
            now = monotonic()
            unresolved: dict[str, list[str]] = {
                entity_type: [] for entity_type in self._ENTITY_TYPES
            }
            for entity_type, names in missing.items():
                for name in names:
                    normalized = normalize_name(name)
                    cached = self._cache.get((entity_type, normalized))
                    if cached and cached[0] > now:
                        catalogs[entity_type][normalized] = cached[1]
                    else:
                        unresolved[entity_type].append(name)

            if any(unresolved.values()):
                payload = await mcp_client.resolve_urban_entity_types(
                    service_names=unresolved["service"],
                    physical_object_names=unresolved["physical_object"],
                )
                payload = payload or {}
                expires_at = monotonic() + self.ttl_seconds
                for entity_type, names in unresolved.items():
                    raw_entries = payload.get(entity_type) or {}
                    entries = {
                        normalize_name(str(key)): value
                        for key, value in raw_entries.items()
                    }
                    for name in names:
                        normalized = normalize_name(name)
                        info = entries.get(normalized) or {}
                        canonical_name = (
                            str(info.get("canonical_name"))
                            if info.get("found") and info.get("canonical_name")
                            else None
                        )
                        catalogs[entity_type][normalized] = canonical_name
                        self._cache[(entity_type, normalized)] = (
                            expires_at,
                            canonical_name,
                        )

                if len(self._cache) > 1024:
                    expired = [
                        key for key, value in self._cache.items() if value[0] <= now
                    ]
                    for key in expired:
                        self._cache.pop(key, None)
        return catalogs
