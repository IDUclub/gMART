"""Check every reference source of ``cases.json`` against the harvested corpus.

A case is trustworthy only if its quotes are the corpus text itself: each
source must name a document of the corpus, its fragment must exist, and the
quote must occur in that fragment (whitespace and the ``x``/``×`` sign
normalised). Facts of answerable cases must occur in the source fragments,
so that grading never asks for a figure the sources do not hold. Exit status is non-zero when anything fails.

    python evaluation/documents_quality/verify_cases.py \
        benchmarks/data/documents_quality/corpus
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent


def normalise(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", text).replace("×", "x").strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    args = parser.parse_args()
    index = json.loads((args.corpus / "index.json").read_text(encoding="utf-8"))
    documents = {document["name"]: document for document in index}
    fragments = {}
    problems = []
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    for case in cases:
        evidence = []
        for source in case.get("sources", []):
            document = documents.get(source["document"])
            if document is None:
                problems.append(f"{case['id']}: нет документа {source['document']!r}")
                continue
            if document["name"] not in fragments:
                path = args.corpus / document["file"]
                fragments[document["name"]] = [
                    json.loads(line) for line in path.open(encoding="utf-8")
                ]
            rows = fragments[document["name"]]
            order = source["fragment_order"]
            if order >= len(rows):
                problems.append(f"{case['id']}: нет фрагмента #{order}")
                continue
            row = rows[order]
            text = normalise((row["text"] or "") + " " + (row.get("table_html") or ""))
            evidence.append(text)
            if normalise(source["quote"]) not in text:
                problems.append(
                    f"{case['id']}: выдержка не найдена во фрагменте #{order}: "
                    f"{source['quote'][:80]!r}"
                )
        if evidence:
            for fact in case.get("facts", []):
                alternatives = fact if isinstance(fact, list) else [fact]
                if not any(normalise(a) in " ".join(evidence) for a in alternatives):
                    problems.append(f"{case['id']}: факт {fact!r} не подтверждён")
    for problem in problems:
        print(problem)
    print(f"{len(cases)} кейсов, проблем: {len(problems)}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
