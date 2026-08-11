"""Semantic correctness of the re-run: intent + source/target entity accuracy.

The deployment does not emit the RestrictionPlan over SSE, so we derive what the
model actually chose from the layers it produced (which the pipeline canonicalises
to catalog names) and score that against the expert gold:

  intent (derived):
    needs_clarification — no layers + a clarification/"available objects" reply
    restrictions        — both `objects` and `generators` layers built
    buffers_only        — entity layers built but no objects/generators
    (error / empty otherwise)

  entities (derived, restrictions mode): the layers are {source, target(buffered),
    objects, generators}. generators == the target buffer, so the target entity is
    the named layer whose feature count equals the generators count; the other
    named entity layer is the source.

Because the gold set is ~all restrictions, a clarification on a restrictions-gold
row is an over-refusal; we report that split explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gold_parser import load_gold, norm  # noqa: E402

RESERVED = {"objects", "generators"}


def _available_from_clarification(resp: str) -> str:
    """The clarification reply lists the scenario's live catalog after
    'Доступные ...:'. Return that tail (normalised) for entity membership tests."""
    n = norm(resp)
    i = n.find("доступн")
    return n[i:] if i >= 0 else ""


def _entity_layers(layer_counts: dict) -> dict[str, int]:
    # drop the reserved backend layers and collapse case-duplicates
    # (the pipeline emits both "Аптека" and "аптека"); keep one per norm key.
    out: dict[str, int] = {}
    seen: set[str] = set()
    for k, v in layer_counts.items():
        nk = norm(k)
        if nk in RESERVED or k in RESERVED or nk in seen:
            continue
        seen.add(nk)
        out[k] = v
    return out


def derive(rec: dict) -> dict:
    lc = rec.get("layer_counts") or {}
    # some records only have `layers` (old format) — build counts
    if not lc and rec.get("layers"):
        for l in rec["layers"]:
            try:
                lc[str(l.get("name"))] = len(l["feature_collection"]["features"])
            except Exception:  # noqa: BLE001
                pass
    has_obj = any(norm(k) == "objects" or k == "objects" for k in lc)
    has_gen = any(norm(k) == "generators" or k == "generators" for k in lc)
    err = bool(rec.get("error"))
    resp = str(rec.get("llm_response") or "").lower()
    clar = (not rec.get("layers")) and any(
        k in resp for k in ("уточн", "доступны", "доступные", "переформул",
                            "отсутств", "не найден")
    )

    if err:
        intent = "error"
    elif has_obj and has_gen:
        intent = "restrictions"
    elif lc:
        intent = "buffers_only"
    elif clar:
        intent = "needs_clarification"
    else:
        intent = "empty"

    source = target = None
    if intent == "restrictions":
        gen = next((v for k, v in lc.items()
                    if norm(k) == "generators" or k == "generators"), None)
        ent = _entity_layers(lc)
        # target = entity layer whose count matches generators (the buffered one)
        for k, v in ent.items():
            if gen is not None and v == gen:
                target = k
                break
        others = [k for k in ent if k != target]
        if others:
            # source = the largest remaining entity layer (the full source set)
            source = max(others, key=lambda k: ent[k])
    return {"intent": intent, "source": source, "target": target}


def score(results_path: Path, gold_index: dict) -> dict:
    total = 0
    intent_ok = src_ok = src_n = tgt_ok = tgt_n = 0
    src_skipped = tgt_skipped = 0
    over_refusal = correct_restr = correct_clar = 0
    gold_restr = 0
    for line in results_path.open(encoding="utf-8"):
        rec = json.loads(line)
        g = gold_index.get((int(rec["scenario_id"]), norm(rec["prompt"])))
        if g is None:
            continue
        total += 1
        d = derive(rec)
        # intent accuracy (map error/empty to a wrong bucket vs gold)
        if d["intent"] == g.intent:
            intent_ok += 1
        if g.intent == "restrictions":
            gold_restr += 1
            if d["intent"] == "restrictions":
                correct_restr += 1
            elif d["intent"] == "needs_clarification":
                # correct only if a gold entity really is absent from the live
                # catalog the model listed; otherwise it refused an answerable task
                avail = _available_from_clarification(rec.get("llm_response") or "")
                gold_ents = [e for e in (g.source_entity, g.target_entity) if e]
                if avail and gold_ents and all(
                    any(w in avail for w in norm(e).split()) for e in gold_ents
                ):
                    over_refusal += 1          # both gold entities were available
                else:
                    correct_clar += 1          # a gold entity was genuinely absent
        # entity accuracy (only where the model produced a restrictions result
        # AND gold has the entity) — this measures "did it pick the right object".
        # Scored ONLY on gold whose entity resolved against the scenario catalog:
        # the parser also falls back to lexicon lemmas and to raw clause text, and
        # scoring a model layer name against "отсановками ... и них" measures the
        # gold's transcription, not the model.
        if d["intent"] == "restrictions":
            if g.source_entity and d["source"]:
                if g.conf.get("source_in_catalog"):
                    src_n += 1
                    src_ok += norm(d["source"]) == norm(g.source_entity)
                else:
                    src_skipped += 1
            if g.target_entity and d["target"]:
                if g.conf.get("target_in_catalog"):
                    tgt_n += 1
                    tgt_ok += norm(d["target"]) == norm(g.target_entity)
                else:
                    tgt_skipped += 1
    return {
        "total": total,
        "intent_acc": intent_ok / total if total else 0,
        "gold_restrictions": gold_restr,
        "produced_restrictions": correct_restr,
        "over_refusal": over_refusal,
        "correct_clarification": correct_clar,
        "src_acc": src_ok / src_n if src_n else None, "src_n": src_n,
        "tgt_acc": tgt_ok / tgt_n if tgt_n else None, "tgt_n": tgt_n,
        "src_skipped": src_skipped, "tgt_skipped": tgt_skipped,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="a model's results.jsonl")
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    args = ap.parse_args()
    gold = load_gold(args.gold)
    gi = {(g.scenario_id, norm(g.question)): g for g in gold}
    s = score(Path(args.results), gi)
    print(f"records joined to gold: {s['total']}")
    print(f"intent accuracy (derived vs gold):        {s['intent_acc']:.1%}")
    print(f"  gold restrictions rows:                 {s['gold_restrictions']}")
    print(f"  → produced restrictions (correct):      {s['produced_restrictions']} "
          f"({s['produced_restrictions']/max(s['gold_restrictions'],1):.1%})")
    print(f"  → over-refusal (both entities available, still clarified): "
          f"{s['over_refusal']} ({s['over_refusal']/max(s['gold_restrictions'],1):.1%})")
    print(f"  → correct clarification (entity truly absent): "
          f"{s['correct_clarification']} "
          f"({s['correct_clarification']/max(s['gold_restrictions'],1):.1%})")
    sa = f"{s['src_acc']:.1%}" if s['src_acc'] is not None else "n/a"
    ta = f"{s['tgt_acc']:.1%}" if s['tgt_acc'] is not None else "n/a"
    print(f"source-entity accuracy (restrictions):    {sa}  (n={s['src_n']}, "
          f"{s['src_skipped']} skipped: gold entity not resolvable to the catalog)")
    print(f"target-entity accuracy (restrictions):    {ta}  (n={s['tgt_n']}, "
          f"{s['tgt_skipped']} skipped: gold entity not resolvable to the catalog)")


if __name__ == "__main__":
    main()
