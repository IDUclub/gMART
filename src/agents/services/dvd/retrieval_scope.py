"""Document scope survives address changes, but explicit new selectors always win."""

import re

from .document_reference import parse_reference


def resets_scope(query):
    return bool(
        re.search(
            r"(?:новый вопрос|другой вопрос|по всей базе|во всех документах|во всех сп|сбрось.*документ)",
            query,
            re.I,
        )
    )


def continues_document(query):
    """An address («пункт 3.3») or anaphora («в нём», «там») refers to the
    document already under discussion; a new topic does not."""
    return bool(
        parse_reference(query).pattern
        or re.search(r"\b(?:н[её]м|него|этом|этого|там|тот же|тому же)\b", query, re.I)
    )


def document_scope(candidates):
    """Persist only an unambiguous document/edition, never a guessed identity."""
    if not candidates:
        return {}
    result = {}
    for key in ("doc_id", "version", "name"):
        values = {c.get(key) for c in candidates}
        if len(values) == 1 and next(iter(values)):
            value = next(iter(values))
            result["document_names" if key == "name" else key] = (
                [value] if key == "name" else value
            )
    if all(c.get("user_id") for c in candidates):
        result["include_shared"] = False
    return result if result.get("doc_id") or result.get("document_names") else {}


def apply_scope(plan, query, scope=None, history=None):
    from src.agents.services.service_entities.dvd_plan import validate_retrieval_plan

    if resets_scope(query):
        if parse_reference(query).document_names:
            return plan
        return validate_retrieval_plan(
            {
                **plan.model_dump(),
                "doc_id": None,
                "document_names": None,
                "version": None,
            }
        )
    explicit = parse_reference(query).document_names
    if explicit:
        # A new named document must not carry the previous document ID or edition.
        return plan
    scope = dict(scope or {})
    if not scope:
        for message in reversed(history or []):
            if message.get("role") != "user":
                continue
            text = message.get("content", "")
            if resets_scope(text):
                break
            names = parse_reference(text).document_names
            if names:
                scope = {"document_names": names}
                break
    # Do not overwrite an explicit document name supplied in free text and resolved
    # by the planner. An address/anaphoric continuation uses the established scope.
    if plan.document_names and not continues_document(query):
        return plan
    updates = {
        k: v
        for k, v in scope.items()
        if k in {"doc_id", "document_names", "version", "include_shared"}
    }
    if plan.version:
        updates["version"] = plan.version
    return validate_retrieval_plan({**plan.model_dump(), **updates})
