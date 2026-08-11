#!/usr/bin/env python3
"""Re-run the restrictions benchmark against a live gMART deployment.

Design goals (per the experiment plan):

* **Preflight checks** — verify the Ollama server and the gMART agents service
  are reachable before doing any work; fail loudly with a clear message if not.
* **Model lifecycle on the server** — for each model: if it is already present
  on the Ollama host, use it and leave it in place; if it is missing, pull it,
  run the benchmark, then delete it afterwards so the shared server is left as
  we found it. Models we did not pull are never deleted.
* **Rich logging** — unlike the previous runs (which stored only final layers),
  every record captures the `RestrictionPlan` (mode / source & target entities /
  buffer parameter) and per-layer feature counts, so the downstream semantic and
  geometric evaluation can score intent, entities, parameter and object
  selection — not just "did a layer appear".

Output schema (JSONL, one row per query) is backward compatible with the old
`results.jsonl` (idx, model, prompt, scenario_id, llm_response, layers, error,
duration_sec) plus new fields: `restriction_plan`, `layer_counts`, `end_state`,
`ablation`.

Usage:
    python benchmarks/harness/run_benchmark.py \
        --agents-base http://localhost:80 \
        --ollama-host http://a.dgx:11434 \
        --token "$URBAN_API_JWT" \
        --dataset benchmarks/data/gold/exp_data.csv \
        --models gemma3:12b gpt-oss:20b \
        --out-dir benchmarks/data/results_rerun

Config may also come from the environment: AGENTS_BASE, OLLAMA_HOST,
URBAN_API_JWT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

COL_Q = "Промт (вопрос)"
COL_SID = "scenario_id"


# --------------------------------------------------------------------------- #
# Auth — obtain / refresh a Keycloak access token via gMART's /auth/token proxy
# --------------------------------------------------------------------------- #
class TokenProvider:
    """Supplies a bearer token for orchestrator calls.

    Two modes:
      * static  — a token string given up front (``--token``);
      * login   — Keycloak username/password exchanged for an access_token at
        ``POST {agents_base}/auth/token`` (the gMART auth-helper proxy). The
        token is refreshed automatically before it expires, so a long batch
        outlives the token TTL. Credentials are read from env/CLI, never logged.

    There is no tokenless path: every orchestrator call requires a bearer token
    and Urban API downstream verifies it too.
    """

    def __init__(self, agents_base: str, token: str = "", username: str = "",
                 password: str = "", token_url: str = "", api_key: str = ""):
        self.agents_base = agents_base
        self.username = username
        self.password = password
        self.token_url = token_url or f"{agents_base}/auth/token"
        self.api_key = api_key  # X-Auth-Helper-Api-Key for the IDU auth helper
        self._token = token
        self._expires_at = float("inf") if token else 0.0
        self._lock = threading.Lock()  # concurrent workers share one token

    @property
    def can_login(self) -> bool:
        return bool(self.username and self.password)

    def _login(self) -> None:
        headers = {"X-Auth-Helper-Api-Key": self.api_key} if self.api_key else {}
        r = requests.post(
            self.token_url,
            json={"username": self.username, "password": self.password,
                  "scope": "openid profile email"},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        self._token = data["access_token"]
        # refresh 60s before the real expiry; default 5min if not reported
        self._expires_at = time.time() + int(data.get("expires_in", 300)) - 60
        print(f"    auth: obtained token (expires in "
              f"~{int(self._expires_at - time.time())}s)")

    def get(self, force: bool = False) -> str:
        # Serialise refreshes so N workers don't all re-login at once; once one
        # thread refreshes, the others see a valid token and skip.
        with self._lock:
            if (force or time.time() >= self._expires_at) and self.can_login:
                self._login()
            return self._token


# --------------------------------------------------------------------------- #
# Ollama lifecycle
# --------------------------------------------------------------------------- #
def ollama_tags(host: str) -> list[str]:
    r = requests.get(f"{host}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def ollama_has(host: str, model: str) -> bool:
    tags = ollama_tags(host)
    # ollama reports "name:tag"; accept exact or ":latest" convenience match
    return model in tags or f"{model}:latest" in tags or any(
        t.split(":")[0] == model.split(":")[0] and t == model for t in tags
    ) or model in {t for t in tags}


def ollama_pull(host: str, model: str) -> None:
    print(f"    pulling {model} …", flush=True)
    with requests.post(f"{host}/api/pull", json={"name": model},
                       stream=True, timeout=None) as r:
        r.raise_for_status()
        last = ""
        for line in r.iter_lines():
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = msg.get("status", "")
            if status != last:
                print(f"      {status}", flush=True)
                last = status
            if msg.get("error"):
                raise RuntimeError(f"pull failed: {msg['error']}")
    print(f"    pulled {model}", flush=True)


def ollama_delete(host: str, model: str) -> None:
    print(f"    deleting {model} (was pulled by us) …", flush=True)
    r = requests.delete(f"{host}/api/delete", json={"name": model}, timeout=60)
    if r.status_code not in (200, 404):
        print(f"      warning: delete returned {r.status_code}: {r.text[:200]}")


# --------------------------------------------------------------------------- #
# SSE parsing of one orchestrator run
# --------------------------------------------------------------------------- #
def _looks_like_plan(obj) -> bool:
    return isinstance(obj, dict) and "mode" in obj and (
        "source_entities" in obj or "buffer_rules" in obj or "target_entities" in obj
    )


def _find_plan(event) -> dict | None:
    """Depth-first search for a RestrictionPlan-shaped dict inside any event."""
    stack = [event]
    while stack:
        cur = stack.pop()
        if _looks_like_plan(cur):
            return cur
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
                elif isinstance(v, str) and v.strip().startswith("{") and '"mode"' in v:
                    try:
                        parsed = json.loads(v)
                        if _looks_like_plan(parsed):
                            return parsed
                    except json.JSONDecodeError:
                        pass
        elif isinstance(cur, list):
            stack.extend(cur)
    return None


ENDPOINTS = {
    # the restrictions experiment targets the restriction pipeline directly, so
    # the plan mode (buffers_only/restrictions/needs_clarification), entities and
    # buffer parameter are the agent's own — not the orchestrator's routing.
    "restrictions": "/restrictions/generate_restrictions/stream",
    "orchestrator": "/orchestrator/route/stream",
}


def run_one(agents_base: str, tokens: "TokenProvider", model: str, prompt: str,
            scenario_id: int, temperature: float, timeout: float,
            extra_params: dict | None = None, _retried: bool = False,
            endpoint: str = "restrictions") -> dict:
    params = {
        "model": model,
        "request": prompt,
        "scenario_id": scenario_id,
        "temperature": temperature,
    }
    if extra_params:
        params.update(extra_params)
    url = f"{agents_base}{ENDPOINTS[endpoint]}?" + urlencode(params)
    headers = {"Authorization": f"Bearer {tokens.get()}",
               "Accept": "text/event-stream"}

    layers: list[dict] = []
    layer_counts: dict[str, int] = {}
    final_text_parts: list[str] = []
    clarification: str | None = None
    plan: dict | None = None
    error: str | None = None

    token_expired = False
    t0 = time.time()
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            if r.status_code == 401 and not _retried and tokens.can_login:
                tokens.get(force=True)
                return run_one(agents_base, tokens, model, prompt, scenario_id,
                               temperature, timeout, extra_params, _retried=True,
                               endpoint=endpoint)
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    ev = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if plan is None:
                    p = _find_plan(ev)
                    if p:
                        plan = p

                etype = ev.get("type")
                content = ev.get("content", {})
                if etype == "feature_collection":
                    name = content.get("name")
                    fc = content.get("feature_collection")
                    layers.append({"name": name, "feature_collection": fc})
                    try:
                        layer_counts[str(name)] = len(fc["features"])
                    except Exception:
                        pass
                elif etype == "chunk":
                    txt = content.get("text") or content.get("content")
                    if txt:
                        final_text_parts.append(str(txt))
                elif etype == "clarification":
                    clarification = content.get("question") or content.get("text")
                elif etype == "orchestrator_final":
                    summ = content.get("summary") or content.get("text")
                    if summ:
                        final_text_parts.append(str(summ))
                elif etype == "token_expired":
                    token_expired = True
                elif etype in ("error", "pipeline_failed"):
                    error = json.dumps(content, ensure_ascii=False)
    except requests.exceptions.Timeout:
        error = f"Timeout: no completion within {timeout}s"
    except requests.exceptions.RequestException as e:
        error = f"{type(e).__name__}: {e}"
    duration = round(time.time() - t0, 2)

    # token expired mid-pipeline: refresh once and re-run the whole query
    if token_expired and not _retried and tokens.can_login:
        tokens.get(force=True)
        return run_one(agents_base, tokens, model, prompt, scenario_id,
                       temperature, timeout, extra_params, _retried=True,
                               endpoint=endpoint)

    llm_response = clarification or "".join(final_text_parts)
    return {
        "model": model,
        "prompt": prompt,
        "scenario_id": scenario_id,
        "llm_response": llm_response,
        "layers": layers,
        "layer_counts": layer_counts,
        "restriction_plan": plan,
        "error": error,
        "duration_sec": duration,
    }


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
def preflight(agents_base: str, ollama_host: str, tokens: "TokenProvider") -> None:
    print("preflight:")
    try:
        tags = ollama_tags(ollama_host)
        print(f"  ✓ Ollama reachable at {ollama_host} ({len(tags)} models present)")
    except Exception as e:
        sys.exit(f"  ✗ Ollama NOT reachable at {ollama_host}: {e}")
    try:
        r = requests.get(f"{agents_base}/docs", timeout=10)
        ok = r.status_code < 500
        print(f"  {'✓' if ok else '✗'} gMART agents reachable at {agents_base} "
              f"(HTTP {r.status_code})")
    except Exception as e:
        sys.exit(f"  ✗ gMART agents NOT reachable at {agents_base}: {e}")
    # auth
    if tokens.can_login:
        try:
            tokens.get(force=True)
            print("  ✓ auth: logged in with Keycloak credentials")
        except Exception as e:
            sys.exit(f"  ✗ auth: /auth/token login failed: {e}")
    elif tokens.get():
        print("  ✓ auth: using the static token provided")
    else:
        sys.exit("  ✗ auth: no token and no credentials — provide --token OR "
                 "--username/--password (KEYCLOAK_USER/KEYCLOAK_PASSWORD). "
                 "Requests cannot run without a token.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents-base", default=os.getenv("AGENTS_BASE", "http://localhost:80"))
    ap.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://a.dgx:11434"))
    ap.add_argument("--token", default=os.getenv("URBAN_API_JWT", ""),
                    help="static bearer token (alternative to Keycloak login)")
    ap.add_argument("--username", default=os.getenv("KEYCLOAK_USER", ""),
                    help="Keycloak username (IDU realm); enables auto login+refresh")
    ap.add_argument("--password", default=os.getenv("KEYCLOAK_PASSWORD", ""),
                    help="Keycloak password (read from env; never logged)")
    ap.add_argument("--token-url", default=os.getenv("AUTH_TOKEN_URL", ""),
                    help="override token endpoint (default {agents-base}/auth/token)")
    ap.add_argument("--auth-api-key", default=os.getenv("AUTH_HELPER_API_KEY", ""),
                    help="X-Auth-Helper-Api-Key for the IDU auth helper /api/token")
    ap.add_argument("--dataset", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Ollama model tags, e.g. gemma3:12b gpt-oss:20b")
    ap.add_argument("--out-dir", default="benchmarks/data/results_rerun")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=0, help="run only first N rows (debug)")
    ap.add_argument("--endpoint", choices=["restrictions", "orchestrator"],
                    default="restrictions",
                    help="which pipeline to benchmark (default: restrictions — "
                         "matches the expert gold set and the previous runs)")
    ap.add_argument("--ablation", choices=["none", "no_catalog"], default="none",
                    help="label the output arm. The 'no_catalog' ablation is "
                         "gated server-side by the ABLATION_NO_CATALOG env var — "
                         "start the gMART agents service with ABLATION_NO_CATALOG=1 "
                         "for that pass; this flag only tags the output dir.")
    ap.add_argument("--keep-pulled", action="store_true",
                    help="do NOT delete models we pulled (default: delete)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results.jsonl and rerun from scratch")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel in-flight requests per model. The workload is "
                         "geometry/IO-bound, so >1 raises throughput and GPU use; "
                         "raises load on Urban API + the deployment.")
    args = ap.parse_args()

    tokens = TokenProvider(args.agents_base, token=args.token,
                           username=args.username, password=args.password,
                           token_url=args.token_url, api_key=args.auth_api_key)
    preflight(args.agents_base, args.ollama_host, tokens)

    df = pd.read_csv(args.dataset, sep=";", engine="python")
    df = df[df[COL_Q].notna() & df[COL_SID].notna()]
    if args.limit:
        df = df.head(args.limit)
    rows = [(int(r[COL_SID]), str(r[COL_Q])) for _, r in df.iterrows()]
    print(f"dataset: {len(rows)} queries from {args.dataset}")

    if args.ablation == "no_catalog":
        print("NOTE: --ablation no_catalog only tags the output. Ensure the gMART "
              "agents service was started with ABLATION_NO_CATALOG=1.")
    extra = None
    out_root = Path(args.out_dir)

    for model in args.models:
        print(f"\n=== model {model} (ablation={args.ablation}) ===")
        try:
            had_model = ollama_has(args.ollama_host, model)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {model}: cannot reach Ollama ({e})")
            continue
        print(f"  present on server: {had_model}")
        if not had_model:
            try:
                ollama_pull(args.ollama_host, model)
            except Exception as e:  # noqa: BLE001
                # a bad tag / failed download must not kill the whole batch
                print(f"  SKIP {model}: pull failed ({e})")
                continue

        safe = model.replace(":", "_").replace("/", "_")
        sub = f"{safe}__{args.ablation}" if args.ablation != "none" else safe
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "results.jsonl"
        # resumable: a multi-hour run must survive a restart. Skip rows already
        # present (by idx) and append; --fresh forces a clean rerun.
        done: set[int] = set()
        if out_path.exists() and not args.fresh:
            for line in out_path.open(encoding="utf-8"):
                try:
                    done.add(int(json.loads(line)["idx"]))
                except Exception:  # noqa: BLE001
                    pass
            if done:
                print(f"  resuming: {len(done)}/{len(rows)} rows already done")
        todo = [(idx, sid, prompt) for idx, (sid, prompt) in enumerate(rows)
                if idx not in done]
        write_lock = threading.Lock()

        def _work(item):
            idx, sid, prompt = item
            rec = run_one(args.agents_base, tokens, model, prompt, sid,
                          args.temperature, args.timeout, extra,
                          endpoint=args.endpoint)
            rec["idx"] = idx
            rec["ablation"] = args.ablation
            rec["endpoint"] = args.endpoint
            return idx, sid, rec

        try:
            with out_path.open("a", encoding="utf-8") as fh:
                completed = len(done)
                with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                    futs = [ex.submit(_work, it) for it in todo]
                    for fut in as_completed(futs):
                        idx, sid, rec = fut.result()
                        with write_lock:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fh.flush()
                        completed += 1
                        tag = ("ok" if not rec["error"]
                               else rec["error"][:30].replace("\n", " "))
                        print(f"    [{completed}/{len(rows)}] idx={idx} sid={sid} "
                              f"{rec['duration_sec']:6.1f}s "
                              f"layers={len(rec['layers'])} {tag}", flush=True)
        finally:
            if not had_model and not args.keep_pulled:
                ollama_delete(args.ollama_host, model)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
