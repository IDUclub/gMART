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
from .document_reference import parse_reference
from .retrieval_scope import document_scope, resets_scope


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
    names = parse_reference(query).document_names
    if names:

        def matches(candidate):
            actual = re.sub(r"\s+", "", normalized(candidate.get("name", "")))
            return any(
                re.match(
                    re.escape(re.sub(r"\s+", "", normalized(name))) + r"(?!\d)", actual
                )
                for name in names
            )

        result = [c for c in result if matches(c)]
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
    return result, best >= 2 or bool(dates) or bool(article) or bool(names)


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
            (candidate.get("doc_id"), candidate.get("name"), candidate.get("version")),
            [],
        ).append(group)
    for entries in documents.values():
        if not all(g[1][0].get("hierarchy") for g in entries):
            continue
        # Rank branches by first relevance, then traverse each branch together.
        # A parent heading must appear once even when leaf scores interleave.
        positions = {}

        def tree_order(group):
            path, order = (), []
            for node in group[1][0]["hierarchy"]:
                path += (node["id"],)
                order.append(positions.setdefault(path, len(positions)))
            return tuple(order)

        entries.sort(key=tree_order)
    groups = [group for entries in documents.values() for group in entries]
    return {
        "plan": plan.model_dump(),
        "question": query,
        "options": [{"label": label, "members": members} for label, members in groups][
            :20
        ],
    }


def render_question(pending):
    lines = [CLARIFICATION, ""]
    previous, parents = None, ()
    labels = {
        "section": "Раздел",
        "chapter": "Глава",
        "article": "Статья",
        "appendix": "Приложение",
    }
    for number, option in enumerate(pending["options"], 1):
        c = option["members"][0]
        document = (c.get("doc_id"), c.get("name", "Документ"), c.get("version"))
        if document != previous:
            lines += [
                "",
                f"**{document[1]} · редакция {document[2] or 'не указана'}**",
                "",
            ]
            previous, parents = document, ()
        if c.get("entity_kind") == "document":
            lines.append(f"- **Вариант {number}: этот документ**")
            continue
        hierarchy = c.get("hierarchy") or []
        if hierarchy:
            keys = tuple(n.get("id") for n in hierarchy[:-1])
            common = 0
            while (
                common < min(len(keys), len(parents))
                and keys[common] == parents[common]
            ):
                common += 1

            def title(n):
                kind = labels.get(
                    n.get("type"), "Пункт" if n.get("numbering") else "Элемент"
                )
                address = " ".join(filter(None, [kind, n.get("numbering")]))
                return address + (". " + n["name"] if n.get("name") else "")

            for depth, node in enumerate(hierarchy[:-1]):
                if depth >= common:
                    lines.append("  " * depth + "- **" + title(node) + "**")
            leaf = hierarchy[-1]
            text = title(leaf)
            # Named entities need no arbitrary text slice. For unnamed nodes show
            # a readable preview, with an explicit ellipsis at a word boundary.
            if not leaf.get("name") and c.get("excerpt"):
                excerpt = " ".join(c["excerpt"].split())
                if len(excerpt) > 160:
                    excerpt = excerpt[:161].rsplit(" ", 1)[0] + "…"
                text += " — «" + excerpt + "»"
            lines.append(
                "  " * (len(hierarchy) - 1) + f"- **Вариант {number}:** " + text
            )
            parents = keys
        else:
            # Compatibility with DVD versions predating typed hierarchy metadata.
            path = " / ".join(
                c.get("selection_path")
                or c.get("structure_path")
                or [c.get("numbering") or "Путь не восстановлен"]
            )
            excerpt = " ".join((c.get("excerpt") or "").split())
            if len(excerpt) > 160:
                excerpt = excerpt[:161].rsplit(" ", 1)[0] + "…"
            lines.append(f"{number}. {path}" + (f" — «{excerpt}»" if excerpt else ""))
    lines += [
        "",
        "Укажите номер варианта или уточните документ, редакцию, раздел или пункт.",
    ]
    return "\n".join(lines)


def resolve_reply(query, pending):
    """Return a constrained plan and selected root IDs; None means a new question."""
    if resets_scope(query):
        return None
    reference = parse_reference(query)
    candidates = [c for option in pending["options"] for c in option["members"]]
    scope = document_scope(candidates)
    if reference.document_names:
        narrowed, _ = narrow_candidates(query, candidates)
        if not narrowed:
            return None
        scope = document_scope(narrowed)
    if reference.pattern and reference.pattern != pending["plan"].get("pattern"):
        # An article-only reply qualifies the existing leaf; a new clause replaces it.
        if re.search(r"(?:пункт[а-я]*|п\.)\s*\d", query, re.I) and scope:
            plan = {
                **pending["plan"],
                **scope,
                "pattern": reference.pattern,
                "retrieval_mode": "structure",
                "name_query": None,
                "rank_by_relevance": False,
                "include_children": True,
                "search_query": query,
            }
            return {"plan": plan}
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
    if (
        len(choice_groups(selected)) == 1
        and selected[0].get("entity_kind") == "document"
    ):
        plan.update(document_scope(selected))
        return {"plan": plan}
    if len(choice_groups(selected)) == 1:
        parsed = parse_choice(candidate_label(selected[0]))
        if parsed is None:
            return {"unresolved": True}
        plan.update(parsed)
        plan["doc_id"] = selected[0].get("doc_id")
        result["selected_ids"] = [c["id"] for c in selected if c.get("id")]
    return result
