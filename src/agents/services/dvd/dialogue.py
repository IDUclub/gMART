"""A pending structural question survives document/edition clarification turns."""

import re

from .clarification import (
    _RUSSIAN_ENDING,
    CLARIFICATION,
    candidate_label,
    choice_groups,
    normalized,
    parse_choice,
    ranked_choices,
)


def _words(text):
    return {
        _RUSSIAN_ENDING.sub("", w)
        for w in re.findall(r"[^\W\d_]{4,}", normalized(text))
        if w
        not in {
            "российской",
            "федерации",
            "редакция",
            "редакции",
            "номер",
            "пункт",
            "статья",
        }
    }


def narrow_candidates(query, candidates):
    words = _words(query)
    scored = [
        (
            sum(
                any(a.startswith(b) or b.startswith(a) for b in words)
                for a in _words(c.get("name", ""))
            ),
            c,
        )
        for c in candidates
    ]
    best = max((score for score, _ in scored), default=0)
    result = [c for score, c in scored if score == best] if best >= 2 else candidates
    dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", query)
    if dates:
        result = [c for c in result if all(d in c.get("version", "") for d in dates)]
    article = re.search(r"(?:стать[яиюе]|ст\.)\s*(\d+(?:\.\d+)*)", query, re.I)
    if article:
        result = [
            c
            for c in result
            if any(
                re.match(
                    r"^(?:статья\s+)?" + re.escape(article[1]) + r"(?:\s|$)",
                    normalized(p),
                )
                for p in c.get("selection_path", c.get("structure_path", []))[:-1]
            )
        ]
    return result, best >= 2 or bool(dates) or bool(article)


def pending_question(plan, candidates, query):
    narrowed, relevant = narrow_candidates(query, candidates)
    # Never discard the full candidate set on an unrecognised/unsatisfied selector.
    candidates = narrowed if relevant and narrowed else candidates
    groups = choice_groups(candidates)
    ranking = {label: i for i, label in enumerate(ranked_choices(candidates, query))}
    groups.sort(key=lambda group: ranking.get(group[0], len(ranking)))
    # Keep each document heading together even when relevance interleaves editions.
    documents = {}
    for group in groups:
        candidate = group[1][0]
        documents.setdefault(
            (candidate.get("name"), candidate.get("version")), []
        ).append(group)
    groups = [group for entries in documents.values() for group in entries]
    return {
        "plan": plan.model_dump(),
        "options": [{"label": label, "members": members} for label, members in groups][
            :20
        ],
    }


def render_question(pending):
    lines = [CLARIFICATION, ""]
    previous = None
    for number, option in enumerate(pending["options"], 1):
        c = option["members"][0]
        document = (c.get("name", "Документ"), c.get("version"))
        if document != previous:
            lines += [f"{document[0]}, редакция {document[1] or 'не указана'}:"]
            previous = document
        path = " / ".join(
            c.get("selection_path")
            or c.get("structure_path")
            or [c.get("numbering") or "Путь не восстановлен"]
        )
        excerpt = " ".join((c.get("excerpt") or "").split())[:180]
        lines.append(f"{number}. {path}" + (f" — «{excerpt}»" if excerpt else ""))
    lines += [
        "",
        "Укажите номер варианта или статью. Можно уточнить документ и редакцию.",
    ]
    return "\n".join(lines)


def resolve_reply(query, pending):
    """Return a constrained plan and selected root IDs; None means a new question."""
    if re.match(
        r"^(?:новый вопрос|другой вопрос|теперь|расскажи|объясни|какие|что|как|почему|зачем)\b",
        normalized(query),
    ):
        return None
    options = pending["options"]
    ordinal = re.fullmatch(
        r"(?:вариант\s*)?(\d+)(?:[.)]|\s+вариант)?", normalized(query)
    )
    index = (
        int(ordinal[1])
        if ordinal
        else {"первый": 1, "второй": 2, "третий": 3}.get(
            normalized(query).removesuffix(" вариант")
        )
    )
    if index:
        if not 0 < index <= len(options):
            return {"unresolved": True}
        selected = options[index - 1]["members"]
    else:
        candidates = [c for option in options for c in option["members"]]
        selected, relevant = narrow_candidates(query, candidates)
        if not relevant:
            return None
        if not selected:
            return {"unresolved": True}
    plan = dict(pending["plan"])
    # A new explicit structural address starts its own question.
    address = re.search(r"(?:пункт[а-я]*|п\.)\s*(\d+(?:\.\d+)*)", query, re.I)
    if address and address[1] != plan.get("pattern"):
        return None
    docs = {c.get("doc_id") for c in selected}
    names = {c.get("name") for c in selected}
    editions = {c.get("version") for c in selected}
    if len(docs) == 1 and None not in docs:
        plan["doc_id"] = next(iter(docs))
    if len(names) == 1:
        plan["document_names"] = list(names)
    if len(editions) == 1:
        plan["version"] = next(iter(editions))
    result = {"plan": plan}
    if len(choice_groups(selected)) == 1:
        parsed = parse_choice(candidate_label(selected[0]))
        if parsed is None:
            return {"unresolved": True}
        plan.update(parsed)
        plan["doc_id"] = selected[0].get("doc_id")
        result["selected_ids"] = [c["id"] for c in selected if c.get("id")]
    return result
