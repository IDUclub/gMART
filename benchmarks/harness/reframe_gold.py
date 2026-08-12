#!/usr/bin/env python3
"""Reframe the accessibility-worded gold questions as restriction checks.

Roughly half the expert gold asks about service accessibility ("сколько жилых
домов в 500 м от школы") while the pipeline under evaluation is a construction-
restrictions pipeline. The underlying spatial task is identical — buffer of R
around B, count/show A inside — so only the WORDING is changed here. Scenario,
entities, radius, expected answer, expected layers and the reference GeoJSON
link are carried over untouched, which is what keeps the expert ground truth
valid for the rewritten question.

Rewriting is done by an LLM because the source wording is irregular (only ~half
the corpus matches any regex), and every rewrite is then checked mechanically:

  * the radius must survive verbatim;
  * both entity phrases must survive as written in the original (canonical
    catalog names are NOT substituted — that would leak the catalog and remove
    the entity-resolution part of the task);
  * accessibility vocabulary must be gone and restriction vocabulary present;
  * no new digits may appear, and the question stays a single sentence.

A row that fails any check keeps its original wording and is reported, so the
output never contains a silently mangled question.

    python3 benchmarks/harness/reframe_gold.py --model gpt-oss:20b
    -> benchmarks/data/gold/exp_data_restrictions.csv   (+ column "Промт (исходный)")
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))
from gold_parser import load_gold, norm  # noqa: E402

COL_Q = "Промт (вопрос)"
COL_ORIG = "Промт (исходный)"

ACCESS_WORDS = (
    "доступн", "пешей", "пешком", "добира", "удобн", "обеспеченност",
    "обespech", "комфорт", "качеств жизни", "привлекательн", "оснащен",
)
RESTRICTION_WORDS = ("ограничен",)  # the rewrite must name the restriction zone, not merely a "zone"

SYSTEM = """Ты переписываешь формулировки задач для геоинформационной системы.

Тебе дают вопрос о территории. Его надо переформулировать как проверку зоны
ограничений, сохранив ровно ту же пространственную задачу.

ЖЁСТКИЕ ПРАВИЛА:
1. Сохрани названия объектов ДОСЛОВНО так, как они написаны в исходном вопросе.
   Не заменяй их синонимами, не приводи к единственному числу, не уточняй.
2. Сохрани число метров без изменений. Других чисел не добавляй.
3. Убери всё, что говорит о доступности, удобстве, обеспеченности, пешей
   доступности и качестве жизни.
4. Сформулируй как проверку попадания объектов в ЗОНУ ОГРАНИЧЕНИЙ заданного
   радиуса вокруг других объектов. Словосочетание "зона ограничений" (в любом
   падеже) обязано присутствовать в ответе.
5. Сохрани тип ответа: если спрашивали количество — спрашивай количество, если
   долю или процент — спрашивай долю или процент.
6. НАПРАВЛЕНИЕ ЗАДАЧИ. Тебе укажут, вокруг каких объектов строится зона. Именно
   они должны стоять после "вокруг"/"от" в твоей формулировке, а вторые объекты —
   те, что проверяются внутри зоны. Не меняй эти роли местами. Называй объекты
   словами исходного вопроса, а не теми, что даны в подсказке.
7. Одно предложение. Без пояснений, без кавычек, без префиксов.

Верни ТОЛЬКО переформулированный вопрос."""


def _phrases(question: str, gold) -> list[str]:
    """Entity phrases to preserve: the longest word of each gold entity as it
    appears in the question (so inflection in the original text is respected)."""
    out = []
    for ent in (gold.source_entity, gold.target_entity):
        if not ent:
            continue
        words = sorted((w for w in norm(ent).split() if len(w) > 3), key=len, reverse=True)
        for w in words:
            stem = w[: max(4, len(w) - 3)]
            if stem in norm(question):
                out.append(stem)
                break
    return out


# Vocabulary the reframing itself is allowed to introduce. Anything else must
# already occur in the original question: the direction hint names entities by
# their canonical catalog form, and without this check the model copies those
# names in ("объектов Детские товары" for "магазинов детских товаров"), which
# both mangles the wording and leaks the catalog into the benchmark.
FRAMING_STEMS = {
    "зон", "ограничен", "радиус", "вокруг", "наход", "попад", "располож",
    "скольк", "как", "процент", "доля", "объект", "метр", "провер",
    "определ", "территор", "проект", "все", "внутр", "вне", "слое", "данн",
    "предел", "количеств", "числ", "имеет", "имеют", "являет",
}


def _new_words(original: str, rewritten: str) -> list[str]:
    """Content words the rewrite introduced that the original never had.

    Comparison is fuzzy on purpose: suffix stripping cannot unify Russian forms
    with a fleeting vowel ("площадками" / "площадок"), so a word counts as known
    when it is close enough to any original word. A swapped entity is nowhere
    near its replacement ("магазин" -> "супермаркет"), so leaks are still caught.
    """
    orig = [w for w in re.split(r"[^а-яa-z0-9]+", norm(original)) if len(w) > 3]
    out = []
    for w in re.split(r"[^а-яa-z0-9]+", norm(rewritten)):
        if len(w) <= 4 or w.isdigit():
            continue
        st = ru_stem(w)
        if any(st.startswith(f) or f.startswith(st) for f in FRAMING_STEMS):
            continue
        if any(
            difflib.SequenceMatcher(None, w, o).ratio() >= 0.72
            or ru_stem(o) == st
            for o in orig
        ):
            continue
        out.append(w)
    return out


def validate(original: str, rewritten: str, gold, keep: list[str]) -> str | None:
    """Return None when the rewrite is acceptable, else the reason it is not."""
    r, o = norm(rewritten), norm(original)
    if not rewritten.strip():
        return "пустой ответ"
    if len(rewritten) > 2 * len(original) + 80:
        return "ответ слишком длинный"
    if rewritten.count("?") > 1 or "\n" in rewritten.strip():
        return "не одно предложение"
    orig_nums = re.findall(r"\d+", o)
    new_nums = re.findall(r"\d+", r)
    if gold.distance_m is not None and str(gold.distance_m) not in new_nums:
        return f"потеряна дистанция {gold.distance_m}"
    if set(new_nums) - set(orig_nums):
        return f"появились новые числа: {sorted(set(new_nums) - set(orig_nums))}"
    for stem in keep:
        if stem not in r:
            return f"потеряна сущность '{stem}'"
    if any(w in r for w in ACCESS_WORDS):
        return "осталась лексика доступности"
    if not any(w in r for w in RESTRICTION_WORDS):
        return "нет формулировки зоны ограничений"
    new = _new_words(original, rewritten)
    if new:
        return f"введены слова, которых не было в вопросе: {new[:3]}"
    return None


# Crude Russian suffix stripping — enough to tell entity mentions apart across
# cases ("жилых"/"жилой" -> "жил"), without pulling in a morphology dependency.
_ENDINGS = (
    "ского", "ские", "ский", "ская", "ское", "ыми", "ими", "ого", "его", "ому",
    "ему", "ах", "ях", "ам", "ям", "ов", "ев", "ей", "ые", "ых", "их", "ая",
    "яя", "ое", "ее", "ой", "ый", "ий", "ии", "ия", "ие", "ью", "а", "я", "о",
    "е", "у", "ю", "ы", "и", "й", "ь",
)


def ru_stem(word: str) -> str:
    for end in _ENDINGS:
        if len(word) - len(end) >= 3 and word.endswith(end):
            return word[: -len(end)]
    return word


def _stem_set(text: str) -> set[str]:
    return {ru_stem(w) for w in re.split(r"[^а-яa-z0-9]+", norm(text)) if len(w) > 3}


def role_inverted(rewritten: str, gold) -> bool | None:
    """True when the rewrite buffers the wrong entity.

    Gold's convention: `target_entity` is the BUFFERED one. In the rewrite the
    buffered entity is whatever follows "вокруг"/"от", so if that phrase names
    the gold SOURCE instead, the question now asks for the mirrored geometry and
    the expected layers no longer describe it. None = cannot tell.
    """
    m = re.search(r"(?:вокруг|от)\s+([^,.?!]{0,45})", norm(rewritten))
    if not m:
        return None
    after = _stem_set(m.group(1))
    tgt, src = _stem_set(gold.target_entity or ""), _stem_set(gold.source_entity or "")
    hit_t, hit_s = bool(tgt & after), bool(src & after)
    if hit_t == hit_s:
        return None
    return hit_s


def is_metric_buffer(question: str) -> bool:
    """Walk-time questions ("10 минут со скоростью 80 метров в минуту") are not a
    metric buffer task; reframing them would misstate the geometry, so skip."""
    return "минут" not in norm(question)


def ask(host: str, model: str, question: str, timeout: float, hint: str = "") -> str:
    resp = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": (f"{hint}\n\n" if hint else "") + question},
            ],
            "stream": False,
            "keep_alive": "2h",
            "options": {"temperature": 0.2},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip().strip('"').strip()


# A question already stating a separation requirement, a ban or a norm is a
# restriction task as written ("Соблюдается ли удаленность детских площадок от
# остановок на 50 и более метров") — the entities and the distance are there and
# the pipeline can build the zone from it. Such rows are left alone.
COMPLIANCE_WORDS = (
    "соблюда", "удаленност", "удалённост", "не ближе", "не менее", "и более метр",
    "запрещ", "нарушен", "санитарн", "охранн", "защитн", "допустим", "предельн",
    "требовани",
)


def already_restriction_shaped(gold) -> bool:
    q = norm(gold.question)
    if any(w in q for w in ACCESS_WORDS):
        return False        # "доступными ... с учётом норматива" is still accessibility
    return any(w in q for w in COMPLIANCE_WORDS)


def is_accessibility(gold) -> bool:
    """Same split as benchmarks/eval: judged by the buffered (target) entity."""
    access = (
        "школ", "детск", "поликлин", "больниц", "аптек", "парк", "сквер", "зелен",
        "супермаркет", "магазин", "булочн", "кафе", "ресторан", "библиотек", "музе",
        "театр", "стадион", "спортивн", "останов", "пункт выдачи", "банкомат",
        "рынок", "стоматолог", "кинотеатр", "бар", "столов", "парковк",
    )
    return any(k in norm(gold.target_entity or "") for k in access)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--out", default="benchmarks/data/gold/exp_data_restrictions.csv")
    ap.add_argument("--report", default="benchmarks/out/reframe_report.csv")
    ap.add_argument("--ollama-host", default="http://a.dgx:11434")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    df = pd.read_csv(args.gold, sep=";", engine="python")
    df[COL_ORIG] = df[COL_Q]

    rows = []
    targets = [
        i for i, g in enumerate(gold)
        if is_accessibility(g) and not already_restriction_shaped(g)
    ]
    if args.limit:
        targets = targets[: args.limit]
    print(f"accessibility-worded rows to reframe: {len(targets)}/{len(gold)}")

    changed = kept = 0
    for n, i in enumerate(targets, 1):
        g = gold[i]
        original = str(df.at[i, COL_Q])
        keep = _phrases(original, g)
        best, reason, last = None, "не пытались", ""
        if not is_metric_buffer(original):
            reason = "радиус задан не в метрах (время ходьбы)"
        else:
            for _ in range(args.attempts):
                try:
                    hint = (
                        f"Зона ограничений строится вокруг: {g.target_entity}. "
                        f"Внутри зоны проверяются: {g.source_entity}."
                        if g.target_entity and g.source_entity else ""
                    )
                    cand = ask(
                        args.ollama_host, args.model, original, args.timeout, hint
                    )
                except Exception as e:  # noqa: BLE001
                    reason = f"ошибка запроса: {str(e)[:60]}"
                    continue
                last = cand
                reason = validate(original, cand, g, keep)
                if reason is None and role_inverted(cand, g) is True:
                    reason = "перевёрнуты роли: буфер вокруг не того объекта"
                if reason is None:
                    best = cand
                    break
        if best:
            df.at[i, COL_Q] = best
            changed += 1
            status = "ok"
        else:
            kept += 1
            status = f"оставлен оригинал ({reason})"
        rows.append({"base_index": i, "status": status,
                     "original": original, "rewritten": best or "",
                     "last_candidate": "" if best else last})
        print(f"  [{n}/{len(targets)}] base={i} {status}")

    df.to_csv(args.out, sep=";", index=False)
    pd.DataFrame(rows).to_csv(args.report, index=False)
    print(f"\nreframed {changed}, kept original {kept} -> {args.out}")
    print(f"per-row report -> {args.report}")


if __name__ == "__main__":
    main()
