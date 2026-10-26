"""Pick corpus fragments worth a benchmark question, spread over every document.

A fragment qualifies when it is a numbered clause of readable length and
falls into one of the question classes: a figure with a qualifier, an
exception or permission, a definition, or a table. At most ``--per-kind``
fragments of each kind are drawn per document (seeded, so the draw repeats),
which keeps one large code from dominating the set.

    python evaluation/documents_quality/candidates.py \
        benchmarks/data/documents_quality/corpus > candidates.md
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

FIGURE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:м|км|мм|см|%|мин|ч|лет|чел|мест|га|дБ|лк|°C|м2|м²|кв\. ?м)\b"
)
QUALIFIER = re.compile(r"не (?:менее|более|ниже|выше|превышающ)|до \d|от \d", re.I)
EXCEPTION = re.compile(
    r"допускается|за исключением|не распространя|кроме случаев|не требуется|не следует",
    re.I,
)
DEFINITION = re.compile(r"^[а-яё][^:]{2,80}:\s+[А-ЯЁ]")


def kinds(row):
    text = row["text"] or ""
    if row["kind"] == "table" or row.get("table_html"):
        yield "table"
    if FIGURE.search(text) and QUALIFIER.search(text):
        yield "figure"
    if EXCEPTION.search(text):
        yield "exception"
    if row["type"] == "definition" or DEFINITION.match(text):
        yield "definition"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--per-kind", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    index = json.loads((args.corpus / "index.json").read_text(encoding="utf-8"))
    for document in index:
        rows = [
            json.loads(line)
            for line in (args.corpus / document["file"]).open(encoding="utf-8")
        ]
        pools = {}
        for row in rows:
            size = len(row["text"] or "") + len(row.get("table_html") or "")
            if not 80 <= size <= 2500 or row["type"] in ("toc", "title_page"):
                continue
            for kind in kinds(row):
                pools.setdefault(kind, []).append(row)
        print(f"\n# {document['name']} ({document['version']}, {len(rows)} фр.)\n")
        for kind, pool in sorted(pools.items()):
            for row in rng.sample(pool, min(args.per_kind, len(pool))):
                text = re.sub(r"\s+", " ", row["text"] or "").strip()
                if row.get("table_html") and not text:
                    text = re.sub(r"<[^>]+>", " | ", row["table_html"])
                    text = re.sub(r"(\s*\|\s*)+", " | ", text)
                print(
                    f"- [{kind}] #{row['order']} п.{row['numbering'] or '—'} "
                    f"({row['type']}) {text[:700]}"
                )


if __name__ == "__main__":
    main()
