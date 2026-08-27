#!/usr/bin/env python3
"""Fill the offline Urban API store so experiment runs need no network.

Run this once while the VPN is up. Afterwards every arm — models, ablations,
repeats, a re-run after a harness fix — replays the same bytes, so a dropped
connection cannot end a run and two arms differ only by the thing under test.

It fetches through ``LocalIduMcpClient``, i.e. the exact calls the pipeline
makes, so what is recorded is keyed the way replay will ask for it. Prefetching
through some other path is how a store ends up full of entries a run never hits.

Coverage per scenario:

* both catalogs (``GetAvailableServices`` / ``GetAvailablePhysicalObjects``) —
  the pipeline reads them on *every* query, so they are the hottest entries and
  were not cached at all before;
* one layer request per entity name, never per combination. Each Urban API call
  underneath is already scoped to a single entity, so recording names one at a
  time lets replay answer any combination a plan happens to ask for.

Scope:

``catalog`` (default)
    every entity the scenario's live catalog offers, union the names in the gold
    row. The model is grounded on that catalog, so this covers every entity a
    valid plan can name — no legitimate choice can miss offline.
``gold``
    only the names in the gold row's ``service_names`` / ``phys_names``. Cheaper,
    but a plan naming a catalog entity outside the gold row will miss on replay
    and be recorded as ``data_unavailable``.

Usage::

    python benchmarks/harness/prefetch_scenarios.py \\
        --dataset benchmarks/data/gold/exp_data.csv \\
        --token "$URBAN_API_JWT" --urban-api-url https://urban-api
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402

COL_SID = "scenario_id"
COL_SN = "service_names"
COL_PN = "phys_names"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="benchmarks/data/gold/exp_data.csv")
    parser.add_argument("--token", default=os.getenv("URBAN_API_JWT", ""))
    parser.add_argument(
        "--urban-api-url", default=os.getenv("URBAN_API_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--urban-data-dir", default=os.getenv("URBAN_DATA_DIR", "runtime/urban_data")
    )
    parser.add_argument("--scope", choices=["catalog", "gold"], default="catalog")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        type=int,
        help="only these scenario ids (default: every one in the dataset)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="entity fetches in flight; a large territory's layer is tens of MB",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="attempts per entity before it is written to the gap report — a "
        "prefetch is exactly when the VPN drops",
    )
    parser.add_argument(
        "--report",
        default="",
        help="where to write the gap report (default: <urban-data-dir>/gaps.json)",
    )
    args = parser.parse_args(argv)
    if not args.token:
        raise SystemExit("--token (or URBAN_API_JWT) is required")
    return args


def read_table(path: Path) -> pd.DataFrame:
    """Read the dataset whatever delimiter it uses.

    The gold export is semicolon-separated (``gold_parser.load_gold`` hardcodes
    ``sep=";"``) while the paraphrase expansion is written with commas. Sniffing
    covers both; a hardcoded separator silently yields a single fused column and
    fails later as a missing-column error.
    """

    return pd.read_csv(path, sep=None, engine="python")


def scenario_names(frame: pd.DataFrame) -> dict[int, tuple[set[str], set[str]]]:
    """Gold entity names per scenario: (service names, physical-object names)."""

    out: dict[int, tuple[set[str], set[str]]] = {}
    for _, row in frame.iterrows():
        if pd.isna(row.get(COL_SID)):
            continue
        scenario_id = int(row[COL_SID])
        services, physical = out.setdefault(scenario_id, (set(), set()))
        for column, target in ((COL_SN, services), (COL_PN, physical)):
            value = row.get(column)
            if isinstance(value, str):
                target.update(part.strip() for part in value.split(",") if part.strip())
    return out


async def prefetch_scenario(
    client,
    scenario_id: int,
    services: set[str],
    physical: set[str],
    *,
    scope: str,
    concurrency: int,
    retries: int,
) -> dict:
    """Record every entity of one scenario. Returns what could not be fetched."""

    from src.agents.services.restriction_catalog import parse_catalog_prompt

    result = {"scenario_id": scenario_id, "failed": [], "fetched": 0}

    try:
        services_prompt = await client.get_available_services_prompt(scenario_id)
        physical_prompt = await client.get_available_physical_objects_prompt(
            scenario_id
        )
    except Exception as exc:  # noqa: BLE001
        result["failed"].append(
            {"what": "catalog", "error": f"{type(exc).__name__}: {exc}"}
        )
        return result

    if scope == "catalog":
        services = services | set(parse_catalog_prompt(services_prompt))
        physical = physical | set(parse_catalog_prompt(physical_prompt))

    semaphore = asyncio.Semaphore(concurrency)

    async def fetch(tool: str, argument: str, name: str) -> None:
        async with semaphore:
            for attempt in range(1, retries + 1):
                try:
                    await client.execute_tool(
                        tool, {argument: [name], "scenario_id": scenario_id}
                    )
                    result["fetched"] += 1
                    return
                except Exception as exc:  # noqa: BLE001
                    if attempt == retries:
                        result["failed"].append(
                            {
                                "what": tool,
                                "name": name,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        return
                    await asyncio.sleep(2 ** (attempt - 1))

    await asyncio.gather(
        *(fetch("GetServices", "services_names", name) for name in sorted(services)),
        *(
            fetch("GetPhysicalObjects", "physical_objects_names", name)
            for name in sorted(physical)
        ),
    )
    return result


async def main_async(args: argparse.Namespace) -> None:
    from src.agents.mcp_clients.local_idu_mcp_client import LocalIduMcpClient
    from src.idu_mcp.common.api_handlers.urban_data_store import UrbanDataStore

    frame = read_table(Path(args.dataset))
    per_scenario = scenario_names(frame)
    if args.scenarios:
        wanted = set(args.scenarios)
        per_scenario = {k: v for k, v in per_scenario.items() if k in wanted}
    print(f"prefetching {len(per_scenario)} scenarios, scope={args.scope}")

    client = LocalIduMcpClient(args.token, urban_api_url=args.urban_api_url)
    reports = []
    started = time.time()
    for position, (scenario_id, (services, physical)) in enumerate(
        sorted(per_scenario.items()), start=1
    ):
        report = await prefetch_scenario(
            client,
            scenario_id,
            services,
            physical,
            scope=args.scope,
            concurrency=args.concurrency,
            retries=args.retries,
        )
        reports.append(report)
        status = "ok" if not report["failed"] else f"{len(report['failed'])} FAILED"
        print(
            f"  [{position}/{len(per_scenario)}] scenario {scenario_id}: "
            f"{report['fetched']} entities, {status}",
            flush=True,
        )

    store = UrbanDataStore()
    stats = store.stats()
    failed = sum(len(report["failed"]) for report in reports)
    report_path = Path(args.report or Path(args.urban_data_dir) / "gaps.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"stats": stats, "scenarios": reports}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(
        f"\ndone in {round(time.time() - started)}s: {stats['entries']} entries, "
        f"{stats['megabytes']} MB, {stats['scenarios']} scenarios"
    )
    print(f"gap report: {report_path}")
    if failed:
        # Not an error: a gap is a fact the run must know about, and re-running
        # this script fills only what is missing (stored entries are hits).
        print(
            f"WARNING: {failed} entities could not be fetched. Re-run this "
            f"script when the connection is back — it refetches only the gaps."
        )


def main() -> None:
    args = parse_args()
    # record mode is the whole point of this script, so it is not optional here
    os.environ["URBAN_DATA_MODE"] = "record"
    os.environ["URBAN_DATA_DIR"] = args.urban_data_dir
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
