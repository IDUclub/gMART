"""Run-scoped, replayable source → zones → target retrieval over MCP only."""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.services.compilance.compliance_inventory import (
    PROJECT_TERRITORY_LAYER,
    _covers_every_zone,
    _zone_roles,
    source_radius,
)
from src.agents.services.compilance.compliance_requirements import attribute_entities
from src.agents.services.service_entities.compliance import DeclaredRequirements


def intersection_type(plan) -> str:
    # TODO: spatial_predicate in CheckPlan (NormGraph, schema 1.1); extra=forbid.
    predicate = plan.params.get("predicate")
    if plan.template == "zonal_attribute_threshold":
        predicate = plan.params.get("join_predicate", predicate)
    return "within" if predicate == "within" else "intersects"


class ScenarioComplianceSource:
    """One source and target query per type; attribute entities keep full records."""

    def __init__(self, executor, client, scenario_id):
        self.executor, self.client, self.scenario_id = executor, client, scenario_id
        self.canonical = {}
        self.sources, self.targets, self.zones = {}, {}, {}
        self.empty_targets = {}
        self.calls = []
        self.territory = None
        self.errors = {}
        self.target_errors = {}
        self.source_call_count = 0
        self.target_entities = []
        self.inventory = False
        self.full_physical_entities = set()
        self.colliding_names = set()
        self.legacy_pools = False
        self._semaphore = asyncio.Semaphore(
            max(1, int(os.getenv("COMPLIANCE_BY_GEOMETRY_CONCURRENCY", "8")))
        )

    def missing(self, plan, params, requirements):
        return self.errors.get(plan.source.restriction_id, [])

    @staticmethod
    def _key(item):
        return f"{item.entity_type.value}:{item.entity}"

    def _spatial(self, item):
        return item.entity_type.value == "service" or (
            item.entity_type.value == "physical_object"
            and item.entity not in self.full_physical_entities
            and not self.legacy_pools
        )

    def _pool_key(self, item, pool):
        key = self._key(item)
        # Inputs prepared before physical by_geometry used names without types,
        # and all physical records came from the full GET.
        if self.legacy_pools and key not in pool:
            return item.entity
        return key

    def _layer_name(self, entity_type, entity, source):
        suffix = f" ({entity_type})" if entity in self.colliding_names else ""
        return f"{'Источники' if source else 'Цели'} — {entity}{suffix}"

    async def load_sources(self, plans, builder, *, inventory=False):
        self.inventory = inventory
        territory_error = None
        try:
            self.territory, call = await builder.project_territory(
                self.client, self.scenario_id
            )
        except TokenExpiredError:
            raise
        except Exception:
            logger.bind(scenario_id=self.scenario_id).exception(
                "Compliance territory retrieval failed"
            )
            self.territory, call = None, None
            territory_error = "retrieval_failed"
        if call:
            self.calls.append(call)
        for raw in plans:
            canonical = await self.executor.canonical_requirements(
                self.client, raw, source_only=inventory
            )
            if canonical is None:
                continue
            plan, params, requirements = canonical
            self.canonical[plan.source.restriction_id] = canonical
            self.full_physical_entities.update(attribute_entities(requirements))

        types_by_name = {}
        for _, _, requirements in self.canonical.values():
            for item in requirements.layers:
                types_by_name.setdefault(item.entity, set()).add(item.entity_type.value)
        self.colliding_names = {
            name for name, types in types_by_name.items() if len(types) > 1
        }
        radii, other = {}, {}
        for plan, params, requirements in self.canonical.values():
            if plan.template == "service_provision" and not inventory:
                continue
            if self.territory is None:
                self.errors[plan.source.restriction_id] = [
                    f"layer:project_territory:{territory_error or 'missing'}"
                ]
            roles = _zone_roles(plan, params)
            if _covers_every_zone(
                plan,
                params,
                {item.role: item.entity for item in requirements.layers},
                inventory=inventory,
            ):
                roles = set()
            for item in requirements.layers:
                if not self._spatial(item) and (not inventory or item.role in roles):
                    other[(item.entity_type.value, item.entity)] = item
                elif item.role in roles:
                    key = (item.entity_type.value, item.entity)
                    radii[key] = max(radii.get(key, 0), source_radius(plan, params))
        for item in other.values():
            try:
                layers, calls = await self.executor._retrieve_layers(
                    self.client,
                    DeclaredRequirements(layers=[item]),
                    self.scenario_id,
                    project_territory=self.territory,
                    full_physical_entities=self.full_physical_entities,
                )
                for rid, (plan, params, reqs) in self.canonical.items():
                    roles = _zone_roles(plan, params)
                    if plan.template == "object_attribute_threshold":
                        roles = set()
                    for req in reqs.layers:
                        if (
                            req.entity == item.entity
                            and req.entity_type == item.entity_type
                        ):
                            pool = self.sources if req.role in roles else self.targets
                            if req.entity in layers:
                                pool[self._key(req)] = layers[req.entity]
                self.calls.extend(calls)
            except TokenExpiredError:
                raise
            except Exception:
                logger.bind(entity=item.entity, scenario_id=self.scenario_id).exception(
                    "Compliance layer retrieval failed"
                )
                for rid, (plan, params, reqs) in self.canonical.items():
                    if plan.template == "service_provision" and not inventory:
                        continue
                    roles = (
                        _zone_roles(plan, params)
                        if plan.template != "object_attribute_threshold"
                        else set()
                    )
                    for req in reqs.layers:
                        if (
                            req.entity != item.entity
                            or req.entity_type != item.entity_type
                        ):
                            continue
                        if req.role in roles:
                            self.errors[rid] = ["source_layers:retrieval_failed"]
                        else:
                            self.target_errors[rid] = [
                                "objects_by_geometry:retrieval_failed"
                            ]

        async def fetch(key, radius):
            entity_type, entity = key
            name = self._layer_name(entity_type, entity, True)
            layer = await self._query(
                entity,
                [PROJECT_TERRITORY_LAYER],
                name,
                {PROJECT_TERRITORY_LAYER: self.territory},
                radius,
                entity_type=entity_type,
            )
            self.sources[f"{entity_type}:{entity}"] = layer

        outcomes = await asyncio.gather(
            *(fetch(e, r) for e, r in radii.items()), return_exceptions=True
        )
        for (entity_type, entity), outcome in zip(radii, outcomes):
            if isinstance(outcome, TokenExpiredError):
                raise outcome
            if isinstance(outcome, Exception):
                logger.bind(entity=entity, scenario_id=self.scenario_id).opt(
                    exception=outcome
                ).error("Compliance source retrieval failed")
                for rid, (plan, params, reqs) in self.canonical.items():
                    roles = (
                        _zone_roles(plan, params)
                        if plan.template != "object_attribute_threshold"
                        and (plan.template != "service_provision" or self.inventory)
                        else set()
                    )
                    if any(
                        item.entity == entity
                        and item.entity_type.value == entity_type
                        and item.role in roles
                        for item in reqs.layers
                    ):
                        self.errors[rid] = ["source_layers:retrieval_failed"]

    async def _query(
        self,
        entity,
        zones,
        name,
        layers,
        radius=0,
        predicate="intersects",
        *,
        entity_type="service",
    ):
        if any(layers.get(key) is None for key in zones):
            raise ValueError("Отсутствует территория проекта или зона поиска")
        tool_name = (
            "GetServicesByGeometry"
            if entity_type == "service"
            else "GetPhysicalObjectsByGeometry"
        )
        name_argument = (
            "services_names" if entity_type == "service" else "physical_objects_names"
        )
        arguments = {
            "scenario_id": self.scenario_id,
            name_argument: [entity],
            "zones_layers": zones,
            "layer_name": name,
            "buffer_m": radius,
            "intersection_type": predicate,
        }
        async with self._semaphore:
            response = await self.executor.tools.execute_named_tool(
                self.client, tool_name, {**arguments, "layers": layers}
            )
        self.calls.append({"function": {"name": tool_name, "arguments": arguments}})
        return response[name]

    async def load_targets(self, on_progress=None):
        self.source_call_count = len(self.calls)
        requests = {}
        for rid, (plan, params, requirements) in self.canonical.items():
            if plan.template == "service_provision" and not self.inventory:
                continue
            zone = self.zones.get(rid)
            source_roles = _zone_roles(plan, params)
            if plan.template in {"object_attribute_threshold", "service_provision"}:
                source_roles = set()
            if (
                zone is not None
                and zone.status in {"shown", "no_objects"}
                and not zone.zone_count
            ):
                self.empty_targets[rid] = {
                    self._key(item): {
                        "type": "FeatureCollection",
                        "features": [],
                        "meta": {"complete": True, "truncated": False},
                    }
                    for item in requirements.layers
                    if item.role not in source_roles
                }
                continue
            if zone is None or not zone.zones or not zone.zone_count:
                continue
            if rid in self.target_errors:
                self.errors[rid] = self.target_errors[rid]
            for item in requirements.layers:
                if not self._spatial(item) or item.role in source_roles:
                    continue
                entry = requests.setdefault(
                    (item.entity_type.value, item.entity),
                    {"layers": {}, "predicates": [], "ids": []},
                )
                # Use the exact provided ref recorded by CreateRestrictionZones.
                name = zone.tool_calls[-1]["function"]["arguments"]["layer_name"]
                entry["layers"][name] = zone.zones
                entry["predicates"].append(intersection_type(plan))
                entry["ids"].append(rid)
        self.target_entities = [entity for _, entity in requests]
        completed = 0

        async def fetch(key, entry):
            nonlocal completed
            entity_type, entity = key
            predicate = (
                "within" if set(entry["predicates"]) == {"within"} else "intersects"
            )
            try:
                self.targets[f"{entity_type}:{entity}"] = await self._query(
                    entity,
                    list(entry["layers"]),
                    self._layer_name(entity_type, entity, False),
                    entry["layers"],
                    predicate=predicate,
                    entity_type=entity_type,
                )
            except TokenExpiredError:
                raise
            except Exception:
                logger.bind(entity=entity, scenario_id=self.scenario_id).exception(
                    "Compliance target retrieval failed"
                )
                for rid in entry["ids"]:
                    self.errors[rid] = ["objects_by_geometry:retrieval_failed"]
            completed += 1
            if on_progress is not None:
                await on_progress(completed, len(requests), entity)

        tasks = [
            asyncio.create_task(fetch(entity, entry))
            for entity, entry in requests.items()
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def retrieve(self, plan, params, requirements):
        layers = {}
        source_roles = _zone_roles(plan, params)
        if plan.template == "object_attribute_threshold" or (
            plan.template == "service_provision" and not self.inventory
        ):
            source_roles = set()
        for item in requirements.layers:
            pool = (
                self.sources
                if item.role in source_roles
                else self.empty_targets.get(plan.source.restriction_id, self.targets)
            )
            key = self._pool_key(item, pool)
            if key in pool:
                layers[self._key(item)] = pool[key]
        if self.territory is not None:
            layers[PROJECT_TERRITORY_LAYER] = self.territory
        # Shared retrieval calls are emitted once, before the checks.
        return layers, []

    def replay_layers(self, plan, params, requirements, layers):
        """Resolve role aliases to the names actually provided by recorded calls."""
        source_roles = _zone_roles(plan, params)
        if plan.template == "object_attribute_threshold" or (
            plan.template == "service_provision" and not self.inventory
        ):
            source_roles = set()
        mapped = []
        renamed = {PROJECT_TERRITORY_LAYER: self.territory} if self.territory else {}
        for item in requirements.layers:
            entity = item.entity
            if self._spatial(item):
                is_source = item.role in source_roles
                entity = self._layer_name(item.entity_type.value, entity, is_source)
                pool = (
                    self.sources
                    if is_source
                    else self.empty_targets.get(
                        plan.source.restriction_id, self.targets
                    )
                )
                if not is_source and plan.source.restriction_id in self.empty_targets:
                    entity += f" (пустая зона {plan.source.restriction_id})"
                key = self._pool_key(item, pool)
                if key in pool:
                    renamed[entity] = pool[key]
            elif self._key(item) in layers:
                renamed[entity] = layers[self._key(item)]
            mapped.append(item.model_copy(update={"entity": entity}))
        return requirements.model_copy(update={"layers": mapped}), renamed

    def snapshot(self):
        return {
            "sources": self.sources,
            "targets": self.targets,
            "empty_targets": self.empty_targets,
            "territory": self.territory,
            "errors": self.errors,
            "calls": self.calls,
            "full_physical_entities": sorted(self.full_physical_entities),
            "colliding_names": sorted(self.colliding_names),
            "target_errors": self.target_errors,
            "inventory": self.inventory,
            "pool_key_version": 1 if self.legacy_pools else 2,
        }

    def restore(self, data):
        self.legacy_pools = data.get("pool_key_version", 1) < 2
        self.full_physical_entities = set(data.get("full_physical_entities", []))
        self.colliding_names = set(data.get("colliding_names", []))
        self.target_errors = data.get("target_errors", {})
        self.inventory = data.get("inventory", False)
        self.sources, self.targets = data["sources"], data["targets"]
        self.empty_targets = data.get("empty_targets", {})
        self.territory, self.errors, self.calls = (
            data["territory"],
            data["errors"],
            data["calls"],
        )
