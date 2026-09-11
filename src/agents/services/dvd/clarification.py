"""Round-trip human-readable retrieval choices without asking an LLM to reparse them."""

from __future__ import annotations

import re
import unicodedata

CLARIFICATION = "Нашлось несколько подходящих элементов. Уточните документ, редакцию или структурный путь:"


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def candidate_label(candidate: dict) -> str:
    path = candidate.get("structure_path") or [
        " ".join(
            filter(None, [candidate.get("numbering"), candidate.get("fragment_name")])
        )
    ]
    label = (
        f"{candidate.get('name')}, редакция {candidate.get('version')}: "
        + " / ".join(path)
    )
    if candidate.get("block") == "amendment":
        label += " [изменения]"
    return label


def ranked_choices(candidates: list[dict], question: str) -> list[str]:
    # Physical fragment IDs are not choices: repeated ingestion can produce
    # several records for the same document, edition and structural address.
    unique = {}
    for candidate in candidates:
        label = candidate_label(candidate)
        unique.setdefault(normalized(label), label)

    def terms(text):
        # Prefix overlap handles Russian inflection without a morphology dependency.
        return {word[:6] for word in re.findall(r"[^\W\d_]{5,}", normalized(text))}

    query_terms = terms(question)

    def relevance(label):
        name, _, path = label.partition(", редакция ")
        path = path.rsplit(": ", 1)[-1]
        return (
            -len(terms(name) & query_terms),
            -len(terms(path) & query_terms),
            path.count(" / "),
        )

    return sorted(unique.values(), key=relevance)


def parse_choice(label: str) -> dict | None:
    label = label.strip().removeprefix("- ")
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
