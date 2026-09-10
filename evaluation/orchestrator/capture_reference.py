"""Independent, read-only REST snapshots under the same identity as run_live."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import dotenv_values
from run_live import ROOT, KeycloakTokenClient, KeycloakTokenConfig, save


async def main():
    env = dotenv_values(ROOT / "env/.env.agents.dev")
    config = KeycloakTokenConfig(
        auth_server_url="http://localhost:8085",
        realm="local",
        client_id="gmart",
        client_secret="local-integration-only",
        background_refresh=True,
    )
    async with KeycloakTokenClient(config) as auth:
        async with httpx.AsyncClient(timeout=120) as http:
            for sid in (772, 848):
                result = {
                    "scenario_id": sid,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                }
                for suffix in (
                    "",
                    "/physical_objects",
                    "/services",
                    "/physical_object_types",
                    "/service_types",
                ):
                    token = await auth.get_access_token()
                    response = await http.get(
                        env["URBAN_API_URL"].rstrip("/")
                        + f"/v1/scenarios/{sid}{suffix}",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    response.raise_for_status()
                    result[suffix or "scenario"] = response.json()
                save(
                    ROOT
                    / f"benchmarks/data/orchestrator_20260910/reference/{sid}.json",
                    result,
                )
                print(f"Saved independent REST reference: {sid}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
