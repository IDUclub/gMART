from src.agents.common.exceptions.token_exceptions import TokenExpiredError
from src.agents.mcp_clients.base_mcp_client import BaseMcpClient, _is_token_expired


class EffectsMcpClient(BaseMcpClient):
    async def calculate_object_effects(
        self,
        service_type_id: int,
        scenario_id: int,
        target_population: int | None = None,
    ) -> dict:
        """
        Call CalculateObjectEffects on the effects MCP server.
        Args:
            service_type_id (int): Service type identifier.
            scenario_id (int): Scenario ID passed as tool argument.
            target_population (int | None): Optional population override.
        Returns:
            dict: Effects result with before_prove_data, after_prove_data, effects, pivot.
        """
        arguments: dict = {
            "service_type_id": service_type_id,
            "scenario_id": scenario_id,
        }
        if target_population is not None:
            arguments["target_population"] = target_population
        try:
            return await self.execute_tool("CalculateObjectEffects", arguments)
        except Exception as exc:
            if _is_token_expired(exc):
                raise TokenExpiredError(str(exc)) from exc
            raise

    async def calculate_services_provision(
        self,
        scenario_id: int,
        services: dict[int, dict],
        target_population: int | None = None,
    ) -> dict:
        """
        Call CalculateServicesProvision on the effects MCP server.
        Args:
            scenario_id (int): Scenario ID passed as tool argument.
            services (dict[int, dict]): Per-service settings keyed by
                service_type_id: {"name": str, "as_layer": bool}.
            target_population (int | None): Optional population override shared
                by all services.
        Returns:
            dict: Per-service results: {"services": {id: {name, summary, layers, error}}}.
        """
        arguments: dict = {
            "scenario_id": scenario_id,
            "services": {str(type_id): info for type_id, info in services.items()},
        }
        if target_population is not None:
            arguments["target_population"] = target_population
        try:
            return await self.execute_tool("CalculateServicesProvision", arguments)
        except Exception as exc:
            if _is_token_expired(exc):
                raise TokenExpiredError(str(exc)) from exc
            raise

    async def calculate_normative_provision(
        self,
        scenario_id: int,
        service_type_id: int,
        capacity_per_1000: float | None = None,
        accessibility_type: str | None = None,
        accessibility_value: float | None = None,
    ) -> dict:
        """
        Call CalculateNormativeProvision on the effects MCP server.
        Args:
            scenario_id (int): Scenario ID passed as tool argument.
            service_type_id (int): Service type identifier.
            capacity_per_1000 (float | None): Places per 1000 residents from the
                norm; None keeps the Urban API normative.
            accessibility_type (str | None): "time" (minutes) or "dist" (metres).
            accessibility_value (float | None): Accessibility from the norm.
        Returns:
            dict: {"normative": {...}, "summary": {...}, "buildings": FeatureCollection}.
        """
        arguments: dict = {
            "scenario_id": scenario_id,
            "service_type_id": service_type_id,
        }
        for key, value in (
            ("capacity_per_1000", capacity_per_1000),
            ("accessibility_type", accessibility_type),
            ("accessibility_value", accessibility_value),
        ):
            if value is not None:
                arguments[key] = value
        try:
            return await self.execute_tool("CalculateNormativeProvision", arguments)
        except Exception as exc:
            if _is_token_expired(exc):
                raise TokenExpiredError(str(exc)) from exc
            raise
