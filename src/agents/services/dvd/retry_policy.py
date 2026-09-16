"""Request-local retrieval identity and server-side review failure categories."""

import json


class CriticResponseError(ValueError):
    reason = "critic_invalid_response"


class ReviewExhaustedError(ValueError):
    reason = "review_exhausted"


def normalized_query(query):
    return " ".join((query or "").split())


def retrieval_key(plan, scenario_id, selected_ids):
    values = plan.model_dump(mode="json")
    values["search_query"] = normalized_query(plan.search_query)
    if plan.retrieval_mode != "semantic" and not plan.rank_by_relevance:
        # Exact retrieval ignores the ranking query and always fetches the full
        # target with its descendants, using pages of 100 and context_height=0.
        for key in ("search_query", "limit", "kind", "types", "context_height"):
            values.pop(key, None)
    return json.dumps(
        [values, scenario_id, selected_ids], sort_keys=True, ensure_ascii=False
    )
