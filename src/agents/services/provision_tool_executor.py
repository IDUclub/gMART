from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING

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
        scenario_id: int,
    ) -> ProvisionStepResult:
        """
        Call GetServiceTypeIdByName on IDU MCP to resolve service_type_id by name.
        Args:
            idu_mcp (IduMcpClient): IDU MCP client with bearer token.
            service_name (str): Human-readable service name (e.g. "Школы").
            scenario_id (int): Scenario identifier passed in MCP meta.
        Returns:
            ProvisionStepResult: Resolved service_type_id.
        """
        raw = await idu_mcp.execute_tool(
            "GetServiceTypeIdByName",
            {"service_name": service_name},
            meta={"scenario_id": scenario_id},
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
        project_id: int,
        target_population: int | None,
    ) -> ProvisionStepResult:
        """
        Call CalculateObjectEffects on the effects MCP server.
        Args:
            effects_mcp (EffectsMcpClient): Effects MCP client with bearer token.
            service_type_id (int): Resolved service type identifier.
            scenario_id (int): Scenario identifier passed in MCP meta.
            project_id (int): Project identifier passed in MCP meta.
            target_population (int | None): Optional population override.
        Returns:
            ProvisionStepResult with keys:
                before_prove_data: {buildings, services, links} — FeatureCollections
                after_prove_data:  {buildings, services, links} — FeatureCollections
                effects:           FeatureCollection
                pivot:             dict
                text_pivot:        str (optional, MCP-generated LLM context)
        """
        raw = await effects_mcp.calculate_object_effects(
            service_type_id=service_type_id,
            scenario_id=scenario_id,
            project_id=project_id,
            target_population=target_population,
        )
        effects: dict = self.to_plain_data(raw) if not isinstance(raw, dict) else raw
        arguments: dict = {"service_type_id": service_type_id}
        if target_population is not None:
            arguments["target_population"] = target_population
        return ProvisionStepResult(
            data=effects,
            tool_calls=[
                {"function": {"name": "CalculateObjectEffects", "arguments": arguments}}
            ],
        )
