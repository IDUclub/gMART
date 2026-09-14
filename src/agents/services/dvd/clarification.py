"""Round-trip human-readable retrieval choices without asking an LLM to reparse them."""

from __future__ import annotations

import re
import unicodedata

CLARIFICATION = "Нашлось несколько подходящих элементов. Уточните документ, редакцию или структурный путь:"
_RUSSIAN_ENDING = re.compile(
    r"(?:иями|ями|ами|ого|его|ому|ему|ией|ов|ев|ей|ия|ие|ии|ую|юю|ая|яя|"
    r"ое|ее|ые|ый|ий|ой|ом|ем|ым|им|ам|ям|ах|ях|а|я|ы|и|у|ю|е|о)$"
)


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def candidate_label(candidate: dict) -> str:
    path = (
        candidate.get("selection_path")
        or candidate.get("structure_path")
        or [
            " ".join(
                filter(
                    None, [candidate.get("numbering"), candidate.get("fragment_name")]
                )
            )
        ]
    )
    label = (
        f"{candidate.get('name')}, редакция {candidate.get('version')}: "
        + " / ".join(path)
    )
    if candidate.get("block") == "amendment":
        label += " [изменения]"
    return label


def choice_groups(candidates: list[dict]) -> list[tuple[str, list[dict]]]:
    """Only proven copies may share a choice; a display path is not an identity."""
    groups = {}
    for index, candidate in enumerate(candidates):
        base = candidate_label(candidate)
        digest = candidate.get("content_digest")
        if digest and candidate.get("parent_id"):
            key = (
                normalized(base),
                candidate.get("doc_id"),
                candidate["parent_id"],
                digest,
            )
        else:
            key = ("node", candidate.get("id") or (normalized(base), index))
        groups.setdefault(key, []).append(candidate)
    labeled = []
    for members in groups.values():
        candidate = members[0]
        label = candidate_label(candidate)
        excerpt = " ".join((candidate.get("excerpt") or "").split())[:200]
        if excerpt:
            label += " — «" + excerpt.replace("«", '"').replace("»", '"') + "»"
        labeled.append((label, members))
    counts = {}
    for label, _ in labeled:
        key = normalized(label)
        counts[key] = counts.get(key, 0) + 1
    seen = {}
    out = []
    for label, members in labeled:
        key = normalized(label)
        if counts[key] > 1:
            seen[key] = seen.get(key, 0) + 1
            label += f" (совпадение {seen[key]})"
        out.append((label, members))
    return out


def ranked_choices(candidates: list[dict], question: str) -> list[str]:
    def terms(text):
        result = set()
        for word in re.findall(r"[^\W\d_]{5,}", normalized(text)):
            # A ranking heuristic, not linguistic analysis: keep common case/number
            # variants together ("этапов" / "этапы") without reducing short roots.
            stem = _RUSSIAN_ENDING.sub("", word)
            result.add((stem if len(stem) >= 4 else word)[:6])
        return result

    query_terms = terms(question)

    def relevance(choice):
        label, members = choice
        candidate = members[0]
        return (
            -len(terms(candidate.get("name") or "") & query_terms),
            -len(terms(label) & query_terms),
            candidate.get("block") == "amendment",
            -len(candidate.get("structure_path") or []),
        )

    return [label for label, _ in sorted(choice_groups(candidates), key=relevance)]


def matching_choices(candidates: list[dict], choice: str) -> list[dict]:
    groups = choice_groups(candidates)
    exact = [
        members for label, members in groups if normalized(label) == normalized(choice)
    ]
    if len(exact) == 1:
        return exact[0]
    # Previously sent labels did not contain excerpts. Accept them only if they
    # still identify one proven group, never union unrelated same-label nodes.
    legacy = [
        members
        for _, members in groups
        if normalized(candidate_label(members[0])) == normalized(choice)
    ]
    return legacy[0] if len(legacy) == 1 else []


def parse_choice(label: str) -> dict | None:
    label = label.strip().removeprefix("- ")
    label = re.sub(r" \(совпадение \d+\)$", "", label)
    label = re.sub(r" — «.*»$", "", label)
    block = "amendment" if label.endswith(" [изменения]") else None
    if block:
        label = label.removesuffix(" [изменения]")
    match = re.fullmatch(r"(.+),\s*редакция\s+(.+):\s+(.+)", label)
    if not match:
        return None
    name, version, path = match.groups()
    if not path.strip() or len(path) > 256:
        return None
    return {
        "retrieval_mode": "structure",
        "pattern": path.strip(),
        "document_names": [name.strip()],
        "version": version.strip() if version.strip() != "None" else None,
        "block": block,
        "include_children": True,
        "allow_multiple": False,
        "name_query": None,
        "doc_id": None,
        "types": None,
        "kind": "all",
    }


def selected_choice(user_query: str, history: list[dict]) -> str | None:
    query = normalized(user_query.strip().removeprefix("- "))
    # Only the latest assistant response can be an outstanding clarification.
    previous = next((m for m in reversed(history) if m.get("role") == "assistant"), {})
    content = previous.get("content", "")
    options = []
    if CLARIFICATION in content:
        options = list(
            dict.fromkeys(
                line[2:]
                for line in content.splitlines()
                if line.startswith("- ") and parse_choice(line[2:])
            )
        )
    for label in options:
        if normalized(label) == query:
            return label  # Retain the exact edition spelling returned by DVD.
    ordinal = re.fullmatch(r"(?:вариант\s*)?(\d+)(?:[.)]|\s+вариант)?", query)
    words = {"первый": 1, "второй": 2, "третий": 3, "четвертый": 4, "пятый": 5}
    index = int(ordinal[1]) if ordinal else words.get(query.removesuffix(" вариант"))
    if index and 0 < index <= len(options):
        return options[index - 1]
    return user_query.strip().removeprefix("- ") if parse_choice(user_query) else None
