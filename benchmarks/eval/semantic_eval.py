"""Task-aware evaluation of the restrictions pipeline over the expert gold set.

Replaces the old universal completion proxy (which only asked "did any layer /
objects+generators appear?") with:

  * mutually-exclusive END STATES that sum to 100% per model;
  * a quantitative FAILURE TAXONOMY separating model / infrastructure / backend;
  * task-aware SUCCESS for restrictions (both objects & generators layers built,
    no execution error) evaluated on the 202 expert queries and the full set;
  * an object-COUNT agreement cross-check vs the gold answer (caveated — the
    reliable object-selection / geometry scoring is done against the reference
    GeoJSON in geometry_eval.py).

Metrics that require the RestrictionPlan (intent / source & target entity /
buffer parameter) are NOT computable from the existing results — the old runs
logged only final layers, not the plan. run_benchmark.py logs the plan so those
are scored on the re-run. That gap is reported explicitly, not hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gold_parser import load_gold, norm  # noqa: E402

MODELS = [
    "gemma3_12b", "gpt-oss_20b", "qwen3_30b-a3b", "llama3_1_8b",
    "mistral-small3_2_24b", "qwen3_14b", "qwen3_32b", "deepseek-r1_32b",
]
# the two candidates the reviewer wants fully evaluated
CANDIDATES = ["gemma3_12b", "gpt-oss_20b"]

END_STATES = [
    "full_success", "partial_spatial", "clarification",
    "planning_failure", "infra_tool_failure", "timeout", "empty",
]
FAILURE_TYPES = [
    "oversized_geojson", "invalid_plan", "timeout", "backend_mcp_crs",
    "tool_execution", "empty_final", "other",
]


def classify_error(err: str) -> str:
    e = err.lower()
    if "linetoolong" in e or "131072" in e or "got more than" in e:
        return "oversized_geojson"
    if "timeout" in e or "timed out" in e:
        return "timeout"
    if any(k in e for k in ("crs", "projection", "epsg", "urban", "mcp",
                            "clientresponse", "connection", "502", "503", "504")):
        return "backend_mcp_crs"
    if any(k in e for k in ("validation", "invalid plan", "restriction plan",
                            "jsondecode", "pydantic")):
        return "invalid_plan"
    if "traceback" in e:
        return "tool_execution"
    return "other"


def layer_map(rec: dict) -> dict[str, dict]:
    return {str(l.get("name")).lower(): l.get("feature_collection")
            for l in (rec.get("layers") or [])}


def is_clarification(text: str) -> bool:
    t = str(text).lower()
    return any(k in t for k in ("уточн", "переформул", "доступны", "доступные",
                                "отсутств", "не найден"))


@dataclass
class Outcome:
    end_state: str
    failure_type: str | None
    task_success: bool          # restrictions: objects+generators built, no error
    any_geojson: bool
    text_explanation: bool
    objects_count: int | None   # features in the 'objects' layer (analytic result)


def classify(rec: dict) -> Outcome:
    err = rec.get("error")
    layers = layer_map(rec)
    has_obj = "objects" in layers
    has_gen = "generators" in layers
    any_geo = bool(layers)
    text = bool(str(rec.get("llm_response") or "").strip())
    obj_count = None
    if has_obj:
        try:
            obj_count = len(layers["objects"]["features"])
        except Exception:
            obj_count = None

    if err:
        ft = classify_error(str(err))
        if ft == "timeout":
            es = "timeout"
        elif ft in ("oversized_geojson", "backend_mcp_crs", "tool_execution"):
            es = "infra_tool_failure"
        elif ft == "invalid_plan":
            es = "planning_failure"
        else:
            es = "infra_tool_failure"
        # NB: an error means the user-facing run failed even if a layer existed
        return Outcome(es, ft, False, any_geo, text, obj_count)

    # no error
    if has_obj and has_gen:
        return Outcome("full_success", None, True, any_geo, text, obj_count)
    if any_geo:
        return Outcome("partial_spatial", None, False, any_geo, text, obj_count)
    if is_clarification(rec.get("llm_response")):
        return Outcome("clarification", None, False, False, text, obj_count)
    return Outcome("empty", "empty_final", False, False, text, obj_count)


def load_results(model: str, results_dir: Path) -> list[dict]:
    p = results_dir / model / "results.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def evaluate(model: str, results_dir: Path, gold_index: dict) -> dict:
    recs = load_results(model, results_dir)
    all_out = [classify(r) for r in recs]
    # gold subset: join by (scenario_id, normalised prompt)
    seen: set = set()
    gold_out: list[tuple[Outcome, object, dict]] = []
    for r in recs:
        key = (int(r["scenario_id"]), norm(r["prompt"]))
        g = gold_index.get(key)
        if g is None or key in seen:
            continue
        seen.add(key)
        gold_out.append((classify(r), g, r))

    def pct(outs, pred):
        n = len(outs)
        return 100.0 * sum(1 for o in outs if pred(o)) / n if n else 0.0

    def dist(outs):
        n = len(outs) or 1
        return {s: 100.0 * sum(1 for o in outs if o.end_state == s) / n
                for s in END_STATES}

    def ftax(outs):
        fails = [o for o in outs if o.failure_type]
        return {t: sum(1 for o in fails if o.failure_type == t)
                for t in FAILURE_TYPES}

    gouts = [o for o, _, _ in gold_out]

    # object-count agreement (restrictions, non-% gold, objects layer present)
    matched = exact = 0
    for o, g, _ in gold_out:
        if (o.objects_count is not None and not g.expected_is_percent
                and g.expected_object_count is not None):
            matched += 1
            exact += abs(o.objects_count - g.expected_object_count) < 1e-6

    return {
        "model": model,
        "n_all": len(recs),
        "n_gold": len(gold_out),
        "task_success_all": pct(all_out, lambda o: o.task_success),
        "task_success_gold": pct(gouts, lambda o: o.task_success),
        "end_states_gold": dist(gouts),
        "end_states_all": dist(all_out),
        "failures_gold": ftax(gouts),
        "failures_all": ftax(all_out),
        "count_matched": matched,
        "count_exact": exact,
    }


# --- markdown reporting -----------------------------------------------------
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def report(evals: list[dict]) -> str:
    evals = sorted(evals, key=lambda e: -e["task_success_gold"])
    L = ["# Task-aware evaluation — restrictions pipeline (expert gold set)\n",
         f"Gold expert queries evaluated per model (join by scenario_id + prompt). "
         f"Full augmented set = {evals[0]['n_all']} runs.\n"]

    L.append("## Table 1. Task-aware restrictions success\n")
    L.append("Success = both `objects` and `generators` layers built AND no "
             "execution error (the 128 KB stream truncation counts as a failed "
             "run here — see Table 3 for the infra/model split).\n")
    L.append(md_table(
        ["Model", "Gold n", "Success @gold", "Success @full"],
        [[e["model"], e["n_gold"], f"{e['task_success_gold']:.1f}%",
          f"{e['task_success_all']:.1f}%"] for e in evals]))

    L.append("\n\n## Table 2. Mutually-exclusive end states on the gold set (sum = 100%)\n")
    L.append(md_table(
        ["Model"] + END_STATES,
        [[e["model"]] + [f"{e['end_states_gold'][s]:.1f}" for s in END_STATES]
         for e in evals]))

    L.append("\n\n## Table 3. Quantitative failure taxonomy on the gold set (run counts)\n")
    L.append(md_table(
        ["Model"] + FAILURE_TYPES,
        [[e["model"]] + [e["failures_gold"][t] for t in FAILURE_TYPES]
         for e in evals]))

    L.append("\n\n## Table 4. Object-count agreement vs gold answer (cross-check only)\n")
    L.append("Fraction of restrictions runs whose `objects` layer feature count "
             "equals the number stated in the gold answer. Weak signal (NL "
             "answers carry negation / OCR noise); authoritative object-selection "
             "and geometry are scored against the reference GeoJSON.\n")
    L.append(md_table(
        ["Model", "Comparable runs", "Exact-count match"],
        [[e["model"], e["count_matched"],
          f"{(100.0*e['count_exact']/e['count_matched']):.1f}%" if e["count_matched"] else "n/a"]
         for e in evals]))

    L.append("\n\n## Not computable from existing results (require the re-run)\n")
    L.append("- **Intent / source & target entity / buffer parameter accuracy** — "
             "the old runs logged only final layers, not the `RestrictionPlan`. "
             "`run_benchmark.py` logs the plan, so these are scored on the re-run.\n"
             "- **Object-selection P/R/F1 and geometry IoU** — require the "
             "reference GeoJSON (geometry_eval.py).\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmarks/data/results")
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--out", default="benchmarks/out/task_aware_report.md")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    gold_index = {(g.scenario_id, norm(g.question)): g for g in gold}
    results_dir = Path(args.results)

    evals = []
    for m in args.models:
        if not (results_dir / m / "results.jsonl").exists():
            print(f"skip {m}: no results.jsonl")
            continue
        evals.append(evaluate(m, results_dir, gold_index))
        e = evals[-1]
        print(f"{m:22s} gold_n={e['n_gold']:3d} "
              f"task_success@gold={e['task_success_gold']:5.1f}% "
              f"count_exact={e['count_exact']}/{e['count_matched']}")

    md = report(evals)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
