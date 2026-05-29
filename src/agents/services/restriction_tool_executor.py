from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.agents.services.restriction_catalog import normalize_name
from src.agents.services.service_entities import GeometryToolCallResult
from src.agents.services.service_entities.restriction_plan import (
    EntityRef,
    RestrictionPlan,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


class RestrictionToolExecutor:
    @staticmethod
    def to_plain_data(value):
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {
                k: RestrictionToolExecutor.to_plain_data(v) for k, v in value.items()
            }
        if isinstance(value, list):
            return [RestrictionToolExecutor.to_plain_data(v) for v in value]
        return value

    @staticmethod
    async def execute_named_tool(
        mcp_client: IduMcpClient,
        tool_name: str,
        arguments: dict,
        meta: dict | None = None,
    ) -> dict[str, dict]:
        result = await mcp_client.execute_tool(tool_name, arguments, meta=meta)
        return RestrictionToolExecutor.to_plain_data(result)

    async def retrieve_layers_for_plan(
        self,
        mcp_client: IduMcpClient,
        plan: RestrictionPlan,
        scenario_id: int,
    ) -> GeometryToolCallResult:
        entities_by_type = self._entities_by_type(
            [*plan.source_entities, *plan.target_entities]
        )
        layers: dict[str, dict] = {}
        tool_calls: list[dict] = []

        self._append_if_present(
            tool_calls,
            await self._update_layers(
                layers,
                mcp_client,
                "GetServices",
                {
                    "services_names": entities_by_type["service"],
                    "scenario_id": scenario_id,
                },
            ),
        )
        self._append_if_present(
            tool_calls,
            await self._update_layers(
                layers,
                mcp_client,
                "GetPhysicalObjects",
                {
                    "physical_objects_names": entities_by_type["physical_object"],
                    "scenario_id": scenario_id,
                },
            ),
        )
        return GeometryToolCallResult(
            tool_result=layers,
            tool_calls=tool_calls,
            messages=[
                {
                    "role": "system",
                    "content": plan.model_dump_json(ensure_ascii=False),
                }
            ],
        )

    async def _update_layers(
        self,
        layers: dict[str, dict],
        mcp_client: IduMcpClient,
        tool_name: str,
        arguments: dict,
        meta: dict | None = None,
    ) -> dict | None:
        if not next(iter(arguments.values())):
            return None
        layers.update(
            await self.execute_named_tool(mcp_client, tool_name, arguments, meta)
        )
        return {"function": {"name": tool_name, "arguments": arguments}}

    @staticmethod
    def _append_if_present(items: list, item) -> None:
        if item is not None:
            items.append(item)

    async def run_buffer_plan(
        self,
        mcp_client: IduMcpClient,
        plan: RestrictionPlan,
        layers: dict[str, dict],
    ) -> GeometryToolCallResult:
        buffer_info = self._build_buffer_info(plan, list(layers))
        if not buffer_info:
            raise ValueError("No source layers found for buffer construction")

        arguments = {"buffer_info": buffer_info}
        tool_result = await self.execute_named_tool(
            mcp_client,
            "CreateBuffers",
            arguments,
            meta={"objects": layers},
        )
        return self._tool_result("CreateBuffers", arguments, tool_result, plan)

    async def run_restriction_plan(
        self,
        mcp_client: IduMcpClient,
        plan: RestrictionPlan,
        base_layers: dict[str, dict],
        buffers: dict[str, dict],
    ) -> GeometryToolCallResult:
        layers = {**base_layers, **buffers}
        arguments = self._build_restriction_arguments(plan, list(layers), list(buffers))
        if (
            not arguments["generators"]
            or not arguments["objects"]
            or not arguments["restrictions"]
        ):
            raise ValueError("No valid restriction relations found in the plan")

        tool_result = await self.execute_named_tool(
            mcp_client,
            "CreateRestrictions",
            arguments,
            meta={"layers": layers},
        )
        return self._tool_result("CreateRestrictions", arguments, tool_result, plan)

    @staticmethod
    def _entities_by_type(entities: list[EntityRef]) -> dict[str, list[str]]:
        result = {"service": [], "physical_object": []}
        for entity in entities:
            result[entity.entity_type].append(entity.name)
        return result

    def _build_buffer_info(
        self,
        plan: RestrictionPlan,
        layer_keys: list[str],
    ) -> dict[str, dict[str, int | str]]:
        buffer_info = {}
        for rule in plan.buffer_rules:
            layer_key = self._resolve_layer_key(layer_keys, rule.source_name)
            if not layer_key:
                continue
            buffer_info[layer_key] = {
                "buffer_size": rule.buffer_size,
                "buffer_type": rule.buffer_type,
                "title": rule.title,
            }
        return buffer_info

    def _build_restriction_arguments(
        self,
        plan: RestrictionPlan,
        layer_keys: list[str],
        buffer_keys: list[str],
    ) -> dict[str, list | dict]:
        generators: list[str] = []
        objects: list[str] = []
        restrictions: dict[str, dict[str, str | list[str]]] = {}

        for rule in plan.restriction_rules:
            generator_key = self._resolve_layer_key(buffer_keys, rule.source_name)
            generator_key = generator_key or self._resolve_layer_key(
                layer_keys, rule.source_name
            )
            target_keys = self._resolve_target_keys(layer_keys, rule.target_names)
            if not generator_key or not target_keys:
                continue

            self._append_unique(generators, generator_key)
            for target_key in target_keys:
                self._append_unique(objects, target_key)
            restrictions[generator_key] = {
                "title": rule.title,
                "description": rule.description,
                "to": target_keys,
            }

        return {
            "generators": generators,
            "objects": objects,
            "restrictions": restrictions,
        }

    def _resolve_target_keys(
        self, layer_keys: list[str], target_names: list[str]
    ) -> list[str]:
        return [
            target_key
            for target_name in target_names
            if (target_key := self._resolve_layer_key(layer_keys, target_name))
        ]

    @staticmethod
    def _resolve_layer_key(layer_keys: list[str], name: str) -> str | None:
        layer_map = {normalize_name(layer_key): layer_key for layer_key in layer_keys}
        return layer_map.get(normalize_name(name))

    @staticmethod
    def _append_unique(items: list[str], item: str) -> None:
        if item not in items:
            items.append(item)

    @staticmethod
    def _tool_result(
        tool_name: str,
        arguments: dict,
        tool_result: dict,
        plan: RestrictionPlan,
    ) -> GeometryToolCallResult:
        return GeometryToolCallResult(
            tool_result=tool_result,
            tool_calls=[{"function": {"name": tool_name, "arguments": arguments}}],
            messages=[
                {
                    "role": "system",
                    "content": plan.model_dump_json(ensure_ascii=False),
                }
            ],
        )
