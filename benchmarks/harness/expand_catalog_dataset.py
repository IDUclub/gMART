#!/usr/bin/env python3
"""Grow the gold set with questions written *from* each scenario's own catalog.

``expand_dataset.py`` paraphrases: same entities, same distance, new wording. It
therefore inherits whatever the base question asked for — including entities the
scenario does not actually expose, which the pipeline can only answer with a
clarification. On the expert set that is most of the run.

This script goes the other way. For every base record it draws nine
``(buffered entity, counted entity, distance)`` triples from the entities the
scenario's Urban API catalog really offers, and asks a local model to write the
question a planner would have written for each. Three consequences:

* every generated question is answerable in its scenario, so an end state of
  ``needs_clarification`` is a finding about the model rather than about the
  dataset;
* the triple is chosen before the question is written, so the buffered entity,
  the counted entity and the buffer parameter are **known** — the extended set
  carries the same ground truth ``inproc_eval.py`` scores the expert set on,
  without new expert annotation;
* entities the Urban API cannot currently serve (see the gap report) are
  excluded up front, so an extended row never fails for a reason the models had
  no part in.

What the extended set is *not*: it has no expert answer text, no expected object
count and no reference GeoJSON. It scores intent, entity roles, the parameter,
plan validity, end states and the failure taxonomy — not object-selection P/R/F1
and not geometry. Keep the two sets in separate tables; the expert set stays the
one that carries the spatial ground truth.

The output is a semicolon CSV in exactly the column shape of ``exp_data.csv``,
so ``gold_parser.load_gold`` reads it and ``inproc_eval.py --gold`` scores it
with no changes:

    python benchmarks/harness/expand_catalog_dataset.py \\
        --model gpt-oss:20b --ollama-host http://localhost:11434 \\
        --n-questions 9 --out benchmarks/data/gold/expanded_catalog.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from gold_parser import norm  # noqa: E402

COL_IDX = "Unnamed: 0"
COL_NO = "№ п/п"
COL_PROJ = "Наименование проекта"
COL_PROJ_ID = "ID проекта"
COL_SCEN_NAME = "Название сценария"
COL_Q = "Промт (вопрос)"
COL_A = "Промт (ответ)"
COL_CLOUD = "Ссылка на облако"
COL_SID = "scenario_id"
COL_SN = "service_names"
COL_PN = "phys_names"

SERVICE = "service"
PHYSICAL = "physical_object"

# The distances the expert set actually uses. Sampling from the gold
# distribution keeps the extended questions in the range the tools were built
# for — a 5 km buffer on a city scenario is a different (and easier) problem.
DISTANCES_M = [50, 100, 150, 200, 300, 500]

SYS_PROMPT = (
    "Ты составляешь тестовые запросы к градостроительной ГИС на русском языке. "
    "Тебе дают список заданий: в каждом указаны объект, вокруг которого строится "
    "буферная зона, объект, который надо посчитать внутри этой зоны, и радиус в "
    "метрах. Для КАЖДОГО задания напиши ОДИН запрос от лица градостроителя.\n"
    "Правила:\n"
    "- ровно {n} строк, по одной на задание, в том же порядке, без нумерации и "
    "пояснений;\n"
    "- в строке обязательно должны встретиться названия обоих объектов и число "
    "радиуса с единицей измерения;\n"
    "- отношение всегда «внутри радиуса / в пределах / не далее чем»; НИКОГДА не "
    "пиши «за пределами», «вне», «дальше чем»;\n"
    "- названия объектов бери из задания, можно склонять и ставить во "
    "множественное число, но не заменяй их синонимами;\n"
    "- добавь короткое обоснование в стиле примера, если оно уместно;\n"
    "- не добавляй других условий, дат, площадей и этажности;\n"
    "- ВСЕ {n} строк должны быть сформулированы ПО-РАЗНОМУ: меняй глагол "
    "(покажи / найди / определи / сколько / проверь / выведи), порядок частей "
    "и стиль — от делового до разговорного. Не повторяй одну конструкцию "
    "дважды; шаблонный список из одинаковых фраз не годится."
)

# The same polarity guard expand_dataset.py uses: a question generated as
# "within the radius" that comes back as "outside the radius" describes a
# different task and would be scored against the wrong ground truth.
_INVERSION = re.compile(
    r"за предел|вне\b|снаружи|дальше|за границ|более чем|не ближе|"
    r"превыша\w* \d|больше \d|дальше чем"
)


# --------------------------------------------------------------------------- #
# Catalogs, read from the offline store
# --------------------------------------------------------------------------- #
def _entry_files(root: Path, scenario_id: int) -> list[Path]:
    return sorted((root / str(scenario_id)).glob("*.json.gz"))


def catalog_from_store(root: Path, scenario_id: int) -> dict[str, list[str]]:
    """The scenario's two catalogs, out of the store ``prefetch_scenarios`` filled.

    Reading the store rather than Urban API keeps this script offline and, more
    to the point, keeps it honest: an entity is only offered to the generator if
    the run that will execute the questions can actually fetch it.
    """

    catalogs: dict[str, list[str]] = {SERVICE: [], PHYSICAL: []}
    for path in _entry_files(root, scenario_id):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, json.JSONDecodeError, EOFError):
            continue
        endpoint = str(entry.get("endpoint", ""))
        if endpoint.endswith("/service_types"):
            key = SERVICE
        elif endpoint.endswith("/physical_object_types"):
            key = PHYSICAL
        else:
            continue
        response = entry.get("response")
        if not isinstance(response, list):
            continue
        catalogs[key] = [
            str(item["name"])
            for item in response
            if isinstance(item, dict) and item.get("name")
        ]
    return catalogs


def broken_entities(report_paths: list[Path]) -> dict[int, set[str]]:
    """Entity names the Urban API could not serve, per scenario.

    A question built on one of these can only end in ``data_unavailable``: the
    row would be dropped from every model-facing denominator, so generating it
    at all just burns a slot in the extended set.
    """

    broken: dict[int, set[str]] = {}
    for path in report_paths:
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for scenario in report.get("scenarios", []):
            try:
                scenario_id = int(scenario["scenario_id"])
            except (KeyError, TypeError, ValueError):
                continue
            for failure in scenario.get("failed", []):
                name = failure.get("name")
                if name:
                    broken.setdefault(scenario_id, set()).add(norm(name))
                else:
                    # A catalog-level failure means the whole scenario is out.
                    broken.setdefault(scenario_id, set()).add("*")
    return broken


# --------------------------------------------------------------------------- #
# Triples
# --------------------------------------------------------------------------- #
def draw_triples(
    catalogs: dict[str, list[str]],
    blocked: set[str],
    count: int,
    rng: random.Random,
    used: set[tuple[str, str, int]] | None = None,
) -> list[dict]:
    """``count`` distinct (buffered, counted, distance) triples for one scenario.

    Both roles are drawn from the union of the two catalogs — the pipeline
    buffers services and physical objects alike — but never the same entity in
    both roles, which would ask how many schools are within 200 m of a school.

    ``used`` carries the triples already taken **for this scenario**, across every
    base record, and is added to in place. Several base records share a scenario
    (ten of them share 1747), and drawing each one independently produced the
    same triple twice often enough to matter — near-duplicate questions inflate
    the row count without adding coverage.
    """

    used = set() if used is None else used

    pool = [
        (name, kind)
        for kind, names in catalogs.items()
        for name in names
        if norm(name) not in blocked
    ]
    # De-duplicate on the normalised name: the catalogs carry case variants.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for name, kind in pool:
        if norm(name) in seen:
            continue
        seen.add(norm(name))
        unique.append((name, kind))
    if len(unique) < 2:
        return []

    # Candidates are (pair × radius), not pairs alone. The radius is part of the
    # task, so the same two entities at 100 m and at 500 m are two questions —
    # and counting them separately is what keeps a small catalog usable: scenario
    # 846 offers 132 ordered pairs and carries 8 base records, so at 18 questions
    # each the pairs alone run out, while pairs × 6 radii do not.
    combos = [
        (buffered, counted, distance)
        for buffered in unique
        for counted in unique
        if norm(buffered[0]) != norm(counted[0])
        for distance in DISTANCES_M
    ]
    rng.shuffle(combos)
    # Prefer an unused *pair* before reusing one at a different radius, so the
    # set spreads over the catalog instead of clustering on a few entities.
    used_pairs = {(norm(b), norm(c)) for b, c, _ in used}
    combos.sort(key=lambda combo: (norm(combo[0][0]), norm(combo[1][0])) in used_pairs)

    triples: list[dict] = []
    for buffered, counted, distance in combos:
        if len(triples) >= count:
            break
        key = (norm(buffered[0]), norm(counted[0]), distance)
        if key in used:
            continue
        used.add(key)
        triples.append(
            {
                "buffered_name": buffered[0],
                "buffered_kind": buffered[1],
                "counted_name": counted[0],
                "counted_kind": counted[1],
                "distance_m": distance,
            }
        )
    return triples


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def build_prompt(triples: list[dict], example: str) -> str:
    tasks = "\n".join(
        f"{i}. буфер вокруг: {t['buffered_name']}; считать внутри: "
        f"{t['counted_name']}; радиус: {t['distance_m']} м"
        for i, t in enumerate(triples, start=1)
    )
    return (
        SYS_PROMPT.format(n=len(triples))
        + f"\n\nПример стиля (другой сценарий):\n{example.strip()}"
        + f"\n\nЗадания:\n{tasks}"
    )


def generate(
    host: str, model: str, prompt: str, temperature: float, timeout: float
) -> list[str]:
    """Ask the model once and split the reply into lines.

    Ollama's native ``/api/generate`` is used rather than the OpenAI route so the
    model can be pinned resident with ``keep_alive``; a 20B model reloaded per
    request costs minutes per call.
    """

    base = host.rstrip("/")
    if base.endswith("/v1"):
        response = requests.post(
            f"{base}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"] or ""
    else:
        response = requests.post(
            f"{base}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        text = response.json().get("response", "")
    lines = [
        re.sub(r"^\s*\d+[.)]\s*", "", line).strip(" -–—\t")
        for line in text.splitlines()
        if line.strip()
    ]
    return [line for line in lines if len(line) > 15]


def _stem(word: str) -> str:
    """Crude Russian stem: enough to match «жилой» against «жилых»."""

    stripped = norm(word)
    return stripped[: max(3, len(stripped) - 2)]


def _stems(name: str) -> set[str]:
    words = [w for w in re.findall(r"[а-яёa-z]+", norm(name)) if len(w) > 2]
    return {_stem(word) for word in words}


def mentions(question: str, name: str) -> bool:
    """Whether every word of the entity name survives, inflection allowed."""

    haystack = norm(question)
    stems = _stems(name)
    if not stems:
        return norm(name) in haystack
    return all(stem in haystack for stem in stems)


def valid(question: str, triple: dict, competitors: list[str] | None = None) -> bool:
    """Whether the line can be accepted as the question for this triple.

    ``competitors`` are the scenario's other catalog names, and they are checked
    because the stem match is deliberately loose: a question written for «Река»
    that says «малых рек» matches «Река» on the stem while actually naming
    «Малая река», a different entity in the same catalog. Accepting it would
    file the row under the wrong ground truth — the one failure mode of this
    script that produces plausible, wrong numbers rather than a visible gap.
    """

    if _INVERSION.search(norm(question)):
        return False
    if not re.search(rf"\b{triple['distance_m']}\b", question):
        return False
    targets = (triple["buffered_name"], triple["counted_name"])
    if not all(mentions(question, name) for name in targets):
        return False
    target_stems = [_stems(name) for name in targets]
    for other in competitors or ():
        if norm(other) in {norm(name) for name in targets}:
            continue
        other_stems = _stems(other)
        if not other_stems or not mentions(question, other):
            continue
        # Only a name that *extends* a target is ambiguous evidence; an unrelated
        # entity mentioned in passing does not put the target's identity in doubt.
        if any(other_stems > stems for stems in target_stems):
            return False
    return True


# --------------------------------------------------------------------------- #
# Output rows
# --------------------------------------------------------------------------- #
def answer_text(project: str, triple: dict) -> str:
    """The expected-answer prose ``gold_parser`` reads the entities out of.

    Deliberately carries **no object count**: the count is unknown for a
    generated question, and inventing one would put a fabricated number into a
    field the evaluation scores against.
    """

    return (
        f"В отображаемом слое для проекта {project} показаны объекты типа "
        f"«{triple['counted_name']}», расположенные в радиусе "
        f"{triple['distance_m']} м от объектов типа «{triple['buffered_name']}»."
    )


def layers_text(triple: dict) -> str:
    return (
        f"1 - слой с объектами «{triple['counted_name']}», "
        f"2 - слой с объектами «{triple['buffered_name']}», учитывающий буфер "
        f"{triple['distance_m']} м вокруг каждого объекта."
    )


def make_row(base: pd.Series, triple: dict, question: str, variant: int) -> dict:
    project = str(base.get(COL_PROJ, ""))
    service_names = [
        t["buffered_name"] for t in (triple,) if t["buffered_kind"] == SERVICE
    ] + [t["counted_name"] for t in (triple,) if t["counted_kind"] == SERVICE]
    phys_names = [
        t["buffered_name"] for t in (triple,) if t["buffered_kind"] == PHYSICAL
    ] + [t["counted_name"] for t in (triple,) if t["counted_kind"] == PHYSICAL]
    return {
        COL_NO: "",
        COL_PROJ: project,
        COL_PROJ_ID: base.get(COL_PROJ_ID, ""),
        COL_SCEN_NAME: base.get(COL_SCEN_NAME, ""),
        COL_Q: question,
        COL_A: answer_text(project, triple),
        "layers": layers_text(triple),
        COL_CLOUD: "",
        COL_SID: int(base[COL_SID]),
        COL_SN: ", ".join(service_names),
        COL_PN: ", ".join(phys_names),
        # provenance + the ground truth the triple fixes
        "base_index": int(base.name),
        "variant": variant,
        "gen_buffered_entity": triple["buffered_name"],
        "gen_counted_entity": triple["counted_name"],
        "gen_distance_m": triple["distance_m"],
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", default="benchmarks/data/gold/exp_data.csv")
    parser.add_argument("--out", default="benchmarks/data/gold/expanded_catalog.csv")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    )
    parser.add_argument(
        "--urban-data-dir", default=os.getenv("URBAN_DATA_DIR", "runtime/urban_data")
    )
    parser.add_argument(
        "--gaps",
        nargs="*",
        default=["runtime/urban_data/gaps_token.json", "runtime/urban_data/gaps.json"],
        help="prefetch gap reports; entities listed in them are not offered to "
        "the generator",
    )
    parser.add_argument("--n-questions", type=int, default=9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.base, sep=";", engine="python")
    frame = frame[frame[COL_Q].notna() & frame[COL_SID].notna()].reset_index(drop=True)
    if args.limit:
        frame = frame.head(args.limit)

    store_root = Path(args.urban_data_dir)
    blocked = broken_entities([Path(p) for p in args.gaps])

    # Resumable: one JSONL row per generated question, appended as it is accepted.
    jsonl_path = Path(args.out).with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # Resumable *and* toppable-up: an existing file is not "done", it is a
    # starting point. Raising --n-questions from 9 to 18 therefore generates the
    # nine that are missing rather than the eighteen that would replace them —
    # the first nine cost four hours of GPU, and regenerating them would also
    # throw away the questions the run has already been scored on.
    have: dict[int, int] = {}
    highest: dict[int, int] = {}
    used_by_scenario: dict[int, set[tuple[str, str, int]]] = {}
    if jsonl_path.exists():
        for line in jsonl_path.open(encoding="utf-8"):
            try:
                row = json.loads(line)
                index = int(row["base_index"])
                have[index] = have.get(index, 0) + 1
                # Numbering continues from the highest variant, not from the row
                # count: a record whose earlier batch lost a question to
                # validation has fewer rows than its top variant, and offsetting
                # by the count then reissues a number already in the file.
                highest[index] = max(highest.get(index, 0), int(row["variant"]))
                used_by_scenario.setdefault(int(row[COL_SID]), set()).add(
                    (
                        norm(row["gen_buffered_entity"]),
                        norm(row["gen_counted_entity"]),
                        int(row["gen_distance_m"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        print(
            f"resuming: {sum(have.values())} questions over {len(have)} base "
            f"records already generated",
            flush=True,
        )
    handle = jsonl_path.open("a", encoding="utf-8")

    kept = skipped = 0
    for index, base in frame.iterrows():
        missing = args.n_questions - have.get(index, 0)
        if missing <= 0:
            continue
        scenario_id = int(base[COL_SID])
        scenario_blocked = blocked.get(scenario_id, set())
        if "*" in scenario_blocked:
            print(f"  [{index}] scenario {scenario_id}: no catalog offline, skipped")
            skipped += 1
            continue
        catalogs = catalog_from_store(store_root, scenario_id)
        rng = random.Random(args.seed + index + 1000 * have.get(index, 0))
        triples = draw_triples(
            catalogs,
            scenario_blocked,
            missing,
            rng,
            used_by_scenario.setdefault(scenario_id, set()),
        )
        if len(triples) < missing:
            print(
                f"  [{index}] scenario {scenario_id}: catalog offers only "
                f"{len(triples)} unused triples, generating those"
            )
        if not triples:
            skipped += 1
            continue

        competitors = catalogs[SERVICE] + catalogs[PHYSICAL]
        prompt = build_prompt(triples, str(base[COL_Q]))
        accepted: dict[int, str] = {}
        for attempt in range(1, args.attempts + 1):
            try:
                lines = generate(
                    args.ollama_host,
                    args.model,
                    prompt,
                    args.temperature,
                    args.timeout,
                )
            except (requests.RequestException, KeyError, ValueError) as exc:
                print(f"  [{index}] attempt {attempt}: {type(exc).__name__}: {exc}")
                continue
            for position, triple in enumerate(triples):
                if position in accepted or position >= len(lines):
                    continue
                candidate = lines[position]
                if valid(candidate, triple, competitors):
                    accepted[position] = candidate
            if len(accepted) == len(triples):
                break

        for position, triple in enumerate(triples):
            question = accepted.get(position)
            if not question:
                continue
            # Variants continue from what the file already holds, so a top-up
            # does not restart numbering and collide with the earlier batch.
            row = make_row(base, triple, question, highest.get(index, 0) + position + 1)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
        handle.flush()
        print(
            f"  [{index}] scenario {scenario_id}: "
            f"{len(accepted)}/{len(triples)} questions kept",
            flush=True,
        )
    handle.close()

    rows = [json.loads(line) for line in jsonl_path.open(encoding="utf-8")]
    if not rows:
        raise SystemExit("nothing generated")
    out = pd.DataFrame(rows)
    # gold_parser finds the layers column by prefix, and load_gold reads ';'.
    layers_column = next(
        column
        for column in pd.read_csv(args.base, sep=";", engine="python", nrows=1).columns
        if column.startswith("Слои")
    )
    out = out.rename(columns={"layers": layers_column})
    out.insert(0, COL_IDX, range(len(out)))
    out.to_csv(args.out, sep=";", index=False, encoding="utf-8")
    print(
        f"\nwrote {len(out)} questions for "
        f"{out['base_index'].nunique()} base records -> {args.out}"
        f"  (kept this run: {kept}, scenarios skipped: {skipped})"
    )


if __name__ == "__main__":
    main()
