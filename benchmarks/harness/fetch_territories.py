#!/usr/bin/env python3
"""Download the project-territory polygon for every project in the gold set.

The expert reference layers were exported over a wider area than the scenario the
pipeline queries (median extent 85 km² against ~1 km²), so scoring them as-is
compares different territories. Clipping both sides to the project boundary makes
the comparison well-posed; this fetches those boundaries once.

    GET /api/v1/projects/{project_id}/territory -> {"geometry": <GeoJSON>, ...}

Credentials come from the same env as the benchmark harness (KEYCLOAK_USER /
KEYCLOAK_PASSWORD / AUTH_HELPER_API_KEY).

    python3 benchmarks/harness/fetch_territories.py
    -> benchmarks/data/gold/territories.json   {project_id: geometry}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark import TokenProvider  # noqa: E402

COL_PID = "ID проекта"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--out", default="benchmarks/data/gold/territories.json")
    ap.add_argument("--urban-api", default="https://urban-api.testing.idulab.ru/api")
    ap.add_argument("--agents-base", default="http://localhost:80")
    ap.add_argument("--token-url", default="https://idu-auth-helper.idulab.ru/api/token")
    args = ap.parse_args()

    tokens = TokenProvider(
        args.agents_base,
        token=os.getenv("URBAN_API_JWT", ""),
        username=os.getenv("KEYCLOAK_USER", ""),
        password=os.getenv("KEYCLOAK_PASSWORD", ""),
        token_url=args.token_url,
        api_key=os.getenv("AUTH_HELPER_API_KEY", ""),
    )

    gold = pd.read_csv(args.gold, sep=";", engine="python")
    # one gold row carries the project NAME in the id column; scenario -> project
    # is 1:1, so recover it from the other rows of the same scenario
    ids = pd.to_numeric(gold[COL_PID], errors="coerce")
    by_scenario = (
        pd.DataFrame({"sid": gold["scenario_id"], "pid": ids})
        .dropna()
        .drop_duplicates("sid")
        .set_index("sid")["pid"]
        .to_dict()
    )
    ids = ids.fillna(gold["scenario_id"].map(by_scenario))
    missing = int(ids.isna().sum())
    pids = sorted({int(p) for p in ids.dropna()})
    print(f"projects in gold: {len(pids)}"
          + (f" ({missing} rows have no resolvable project id)" if missing else ""))

    out_path = Path(args.out)
    known: dict[str, dict] = {}
    if out_path.exists():
        known = json.loads(out_path.read_text(encoding="utf-8"))

    failed = []
    for pid in pids:
        if str(pid) in known:
            continue
        url = f"{args.urban_api}/v1/projects/{pid}/territory"
        try:
            r = requests.get(
                url, headers={"Authorization": f"Bearer {tokens.get()}"}, timeout=60
            )
            r.raise_for_status()
            geom = r.json().get("geometry")
            if not geom:
                raise ValueError("no geometry in response")
            known[str(pid)] = geom
            print(f"  {pid}: {geom.get('type')}")
        except Exception as e:  # noqa: BLE001
            failed.append((pid, str(e)[:80]))
            print(f"  {pid}: FAILED {e}")

    out_path.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(known)}/{len(pids)} territories -> {out_path}")
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
