"""Check long Russian quotation spans against actual retrieved DVD text.

Typography normalization only. A mismatch is a review flag, not proof of a false
legal statement; absence of quotes does not prove grounding of the other prose.
"""

import argparse
import json
import re
from pathlib import Path

from review_answers import answers
from run_planner import save


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def normalize(text):
    text = text.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"}))
    return re.sub(
        r"\s+", "", text.replace("*", "").replace("\u00ad", "").casefold()
    ).replace("ё", "е")


def verify(trace):
    evidence = normalize(
        "\n".join(
            strings([c.get("result") for c in trace.get("tools", {}).get("dvd", [])])
        )
    )
    texts = answers(trace)["texts"]
    result = []
    for step in (trace.get("final") or {}).get("steps", []):
        if step["agent"] != "documents" or step["status"] != "completed":
            continue
        text = texts.get(step["step"], "")
        for quote in re.findall("«([^»]+)»", text, flags=re.S):
            if len(quote) < 60:
                continue
            pieces = [
                normalize(p.strip())
                for p in re.split(r"…|\.\.\.", quote)
                if len(p.strip()) >= 20
            ]
            result.append(
                {
                    "quote": quote,
                    "supported_textually": bool(pieces)
                    and all(p in evidence for p in pieces),
                }
            )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.runs.glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        checks = verify(trace)
        if checks:
            rows.append({"id": trace["id"], "checks": checks})
    checks = [c for row in rows for c in row["checks"]]
    result = {
        "cases": rows,
        "quotes": len(checks),
        "matched": sum(c["supported_textually"] for c in checks),
        "scope": "long «quoted» spans in completed DVD answers, normalized typography; mismatch needs review",
    }
    save(args.out, result)
    print(f"Quoted spans: {result['matched']}/{result['quotes']}")
