from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.agents.mcp_clients.effects_mcp_client import EffectsMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient


@dataclass
class ProvisionStepResult:
    data: dict
    tool_calls: list[dict]


class ProvisionToolExecutor:
    @staticmethod
    def to_plain_data(value) -> dict | list | object:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {k: ProvisionToolExecutor.to_plain_data(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ProvisionToolExecutor.to_plain_data(v) for v in value]
        return value

    async def get_service_id(
        self,
        idu_mcp: IduMcpClient,
        service_name: str,
    ) -> ProvisionStepResult:
        """
        Call GetServiceTypeIdByName on IDU MCP to resolve service_type_id by name.
        Args:
            idu_mcp (IduMcpClient): IDU MCP client with bearer token.
            service_name (str): Human-readable service name (e.g. "Школы").
        Returns:
            ProvisionStepResult: Resolved service_type_id.
        """
        logger.info(f"Tool call: GetServiceTypeIdByName(service_name={service_name!r})")
        raw = await idu_mcp.execute_tool(
            "GetServiceTypeIdByName",
            {"service_name": service_name},
        )
        service_type_id = self._parse_service_type_id(raw, service_name)
        return ProvisionStepResult(
            data={"service_type_id": service_type_id},
            tool_calls=[
                {
                    "function": {
                        "name": "GetServiceTypeIdByName",
                        "arguments": {"service_name": service_name},
                    }
                }
            ],
        )

    @staticmethod
    def _parse_service_type_id(raw: object, service_name: str) -> int:
        """
        Parse the service_type_id from the GetServiceTypeIdByName response.
        The tool may return an int directly, or a dict containing the id.
        """
        if isinstance(raw, int):
            return raw
        if isinstance(raw, dict):
            for key in ("service_type_id", "type_id", "id"):
                if key in raw:
                    return int(raw[key])
        # Last resort: coerce scalar to int (e.g. float or numeric string)
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
        raise ValueError(
            f"Could not resolve service_type_id for '{service_name}'. "
            f"Unexpected response from GetServiceTypeIdByName: {type(raw).__name__} = {raw!r}"
        )

    async def calculate_effects(
        self,
        effects_mcp: EffectsMcpClient,
        service_type_id: int,
        scenario_id: int,
        target_population: int | None,
    ) -> ProvisionStepResult:
        """
        Call CalculateObjectEffects on the effects MCP server.
        Args:
            effects_mcp (EffectsMcpClient): Effects MCP client with bearer token.
            service_type_id (int): Resolved service type identifier.
            scenario_id (int): Scenario identifier passed as tool argument.
            target_population (int | None): Optional population override.
        Returns:
            ProvisionStepResult with keys:
                before_prove_data: {buildings, services, links} — FeatureCollections
                after_prove_data:  {buildings, services, links} — FeatureCollections
                effects:           FeatureCollection
                pivot:             dict
                text_pivot:        str (optional, MCP-generated LLM context)
        """
        logger.info(
            f"Tool call: CalculateObjectEffects("
            f"service_type_id={service_type_id}, "
            f"scenario_id={scenario_id}, "
            f"target_population={target_population})"
        )
        raw = await effects_mcp.calculate_object_effects(
            service_type_id=service_type_id,
            scenario_id=scenario_id,
            target_population=target_population,
        )
        effects: dict = self.to_plain_data(raw) if not isinstance(raw, dict) else raw
        arguments: dict = {
            "service_type_id": service_type_id,
            "scenario_id": scenario_id,
        }
        if target_population is not None:
            arguments["target_population"] = target_population
        return ProvisionStepResult(
            data=effects,
            tool_calls=[
                {"function": {"name": "CalculateObjectEffects", "arguments": arguments}}
            ],
        )

    async def calculate_services_provision(
        self,
        effects_mcp: EffectsMcpClient,
        scenario_id: int,
        services: dict[int, dict],
        target_population: int | None = None,
    ) -> ProvisionStepResult:
        """
        Call CalculateServicesProvision on the effects MCP server.
        Args:
            effects_mcp (EffectsMcpClient): Effects MCP client with bearer token.
            scenario_id (int): Scenario identifier passed as tool argument.
            services (dict[int, dict]): Per-service settings keyed by
                service_type_id: {"name": str, "as_layer": bool}.
            target_population (int | None): Optional population override shared
                by all services.
        Returns:
            ProvisionStepResult with data:
                services: {service_type_id: {name, summary, layers, error}}
        """
        logger.info(
            f"Tool call: CalculateServicesProvision("
            f"scenario_id={scenario_id}, services={services}, "
            f"target_population={target_population})"
        )
        raw = await effects_mcp.calculate_services_provision(
            scenario_id=scenario_id,
            services=services,
            target_population=target_population,
        )
        data: dict = self.to_plain_data(raw) if not isinstance(raw, dict) else raw
        arguments: dict = {
            "scenario_id": scenario_id,
            "services": {str(type_id): info for type_id, info in services.items()},
        }
        if target_population is not None:
            arguments["target_population"] = target_population
        return ProvisionStepResult(
            data=data,
            tool_calls=[
                {
                    "function": {
                        "name": "CalculateServicesProvision",
                        "arguments": arguments,
                    }
                }
            ],
        )
