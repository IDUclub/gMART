"""Score benchmark runs against ``cases.json`` and compare them.

Every run is a JSONL file written by ``documents_latency/run_http.py``. The
scorer checks what can be checked without a person:

- ``facts`` — share of the case's required facts found in the grounded part
  of the answer (before the «Без опоры на загруженные документы» block);
  figures are compared with a decimal comma, units normalised;
- ``cited`` — the grounded part names a document the case's sources come from;
- ``refused`` — for unanswerable cases (class E), the grounded part says the
  documents hold no answer instead of stating a requirement;
- ``ok`` — the case passes: all facts and a cited source, or a refusal for E.

Forbidden claims and wrong-but-fluent answers still need a reader: the
report lists every answer whose verdict changed between runs for review.

    python evaluation/documents_quality/score.py runs/a.jsonl runs/b.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
KNOWLEDGE = "Без опоры на загруженные документы"
REFUSAL = re.compile(
    r"прямого ответа на вопрос нет|не найден|нет сведений|не содерж|отсутству|"
    r"не загружен|нет (?:такого|такой|данных|информации|требований)|не приведен|"
    r"не установлен|не регламентир|"
    r"нет (?:\w+ ){0,2}(?:положени|норм|требовани|указани|сведени|упоминани)",
    re.I,
)
# Designations that identify each source document in an answer.
DESIGNATIONS = {
    "Градостроительный кодекс Российской Федерации": ["градостроительн", "грк"],
    "Конституция Российской Федерации": ["конституци"],
    "Постановление Правительства Москвы от 23.12.2015 N 945-ПП": ["945-пп"],
}


# A readable source reference the service puts after a claim: «(СП 55…, п. 8.3)».
CITATION = re.compile(
    r"\([^()]*(?:п\.|пункт|раздел|стать|ст\.|таблиц|приложени)[^()]*\)", re.I
)


def normalise(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[   ]", " ", text)
    text = re.sub(r"[‐‑]", "-", text)
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    text = (
        text.replace("×", "x").replace("х", "x").replace("м²", "м2").replace("м³", "м3")
    )
    text = re.sub(r"кв\.\s*м", "м2", text)
    text = re.sub(r"\s+%", "%", text)
    return re.sub(r"\s+", " ", text).strip()


def designations(document: str) -> list[str]:
    if document in DESIGNATIONS:
        return DESIGNATIONS[document]
    name = normalise(document)
    code = re.match(r"(сп|санпин) [\d.,\-]+\d", name)
    if code:
        # «СП 55.13330.2016» is also cited as «СП 55.13330».
        short = re.match(r"(сп|санпин) \d+(?:,\d+)?", name)
        return [code.group(0), short.group(0)] if short else [code.group(0)]
    number = re.search(r"n (\d+)", name)
    return [f"n {number.group(1)}", f"№ {number.group(1)}"] if number else [name]


def fact_pattern(fact: str) -> re.Pattern:
    """Figures must match exactly; words match by stem, so case endings vary."""
    parts = []
    for token in normalise(fact).split():
        if re.search(r"\d", token) or len(token) <= 3:
            parts.append(re.escape(token))
        else:
            parts.append(re.escape(token[: max(3, len(token) - 3)]) + r"\w*")
    # Up to two extra words between the parts of one fact.
    return re.compile(r"\W+(?:\w+\W+){0,2}?".join(parts))


def has_fact(fact, text: str) -> bool:
    alternatives = fact if isinstance(fact, list) else [fact]
    return any(fact_pattern(a).search(text) for a in alternatives)


def grounded_part(answer: str) -> str:
    return (answer or "").split(KNOWLEDGE, 1)[0]


def score_case(case: dict, record: dict) -> dict:
    answer = record.get("answer") or ""
    grounded = normalise(grounded_part(answer))
    result = {
        "id": case["id"],
        "class": case["class"],
        "error": bool(record.get("error")) or not answer.strip(),
        "total_ms": record.get("total_ms"),
        "knowledge": KNOWLEDGE in answer,
    }
    if not case.get("answerable", True):
        result["refused"] = bool(REFUSAL.search(grounded))
        # A refusal next to sourced requirements is a partial answer that
        # passes something else off as the answer.
        result["claims"] = len(CITATION.findall(grounded_part(answer)))
        result["ok"] = (
            result["refused"] and not result["claims"] and not result["error"]
        )
        return result
    facts = case.get("facts", [])
    hits = [fact for fact in facts if has_fact(fact, grounded)]
    result["facts"] = len(hits) / len(facts) if facts else 1.0
    result["missed"] = [fact for fact in facts if fact not in hits]
    documents = {source["document"] for source in case.get("sources", [])}
    result["cited"] = (
        any(any(d in grounded for d in designations(doc)) for doc in documents)
        if documents
        else True
    )
    result["ok"] = not result["error"] and result["facts"] == 1.0 and result["cited"]
    return result


def load(path: Path) -> dict:
    records = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        if not record.get("skipped"):
            records[record["id"]] = record
    return records


def percentile(values, share):
    values = sorted(values)
    if not values:
        return None
    index = min(len(values) - 1, round(share * (len(values) - 1)))
    return values[index]


def summary(scores: list[dict]) -> dict:
    times = [s["total_ms"] / 1000 for s in scores if s["total_ms"]]
    answerable = [s for s in scores if "facts" in s]
    return {
        "n": len(scores),
        "ok": sum(s["ok"] for s in scores),
        "facts": (
            statistics.mean(s["facts"] for s in answerable) if answerable else None
        ),
        "cited": sum(s.get("cited", False) for s in answerable),
        "errors": sum(s["error"] for s in scores),
        "p50": percentile(times, 0.5),
        "p95": percentile(times, 0.95),
        "mean": statistics.mean(times) if times else None,
    }


def fmt(value, digits=1):
    if value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    runs = []
    for path in args.runs:
        records = load(path)
        scores = [
            score_case(case, records[case["id"]])
            for case in cases
            if case["id"] in records
        ]
        runs.append((path.stem, {s["id"]: s for s in scores}))

    print(
        "| Прогон | n | Верно | Факты | Источник | Ошибки | p50, с | p95, с | Среднее, с |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for name, scores in runs:
        s = summary(list(scores.values()))
        print(
            f"| {name} | {s['n']} | {s['ok']} | {fmt(s['facts'] and s['facts'] * 100)} % "
            f"| {s['cited']} | {s['errors']} | {fmt(s['p50'])} | {fmt(s['p95'])} | {fmt(s['mean'])} |"
        )
    classes = sorted({case["class"] for case in cases})
    print(
        "\n| Класс | " + " | ".join(f"{name}: верно / p50" for name, _ in runs) + " |"
    )
    print("|---|" + "---|" * len(runs))
    for cls in classes:
        cells = []
        for _, scores in runs:
            s = summary([v for v in scores.values() if v["class"] == cls])
            cells.append(f"{s['ok']}/{s['n']} · {fmt(s['p50'])} с")
        print(f"| {cls} | " + " | ".join(cells) + " |")
    if len(runs) > 1:
        (base_name, base), (new_name, new) = runs[0], runs[-1]
        print(f"\nИзменения {base_name} → {new_name}:")
        for case in cases:
            a, b = base.get(case["id"]), new.get(case["id"])
            if a and b and a["ok"] != b["ok"]:
                mark = "стало верно" if b["ok"] else "СТАЛО НЕВЕРНО"
                print(f"- {case['id']}: {mark}; пропущено {b.get('missed', [])}")
    else:
        name, scores = runs[0]
        print(f"\nНеверные в {name}:")
        for case in cases:
            s = scores.get(case["id"])
            if s and not s["ok"]:
                detail = (
                    f"факты {s.get('missed')}, источник {s.get('cited')}"
                    if "facts" in s
                    else f"отказ {s['refused']}, ссылок {s['claims']}"
                )
                print(
                    f"- {case['id']} ({fmt(s['total_ms'] and s['total_ms'] / 1000)} с): {detail}"
                )


if __name__ == "__main__":
    main()
