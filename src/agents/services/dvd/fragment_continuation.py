"""Complete search hits that a document parse cut in the middle of a sentence.

Word wraps a document code after its prefix ("СП\\n17.13330"). IDU_DVD parsers
before ``dvd-parser-6`` took the wrapped code for a clause number and stored the
rest of the requirement as the next fragment, so a hit could end with "с учетом
СП". Such a hit is joined with the fragments that continue it in reading order
(IDU_DVD ``get_node``), and the model reads the whole requirement under one
source label. Documents parsed later are not cut, and nothing is fetched for them.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from loguru import logger

# A sentence, list item or quotation that ends here needs no continuation.
_TERMINAL = re.compile(r"[.!?;:…»)\]\"]\s*$")
# A line ending in a document prefix is continued by that document's code.
_DESIGNATION_TAIL = re.compile(
    r"(?:^|[\s(«\"])(?:ГОСТ(?:\s+Р)?|ТР\s+(?:ТС|ЕАЭС)|СНиП|СанПиН|СП|СН|ГН|РД|ВСН"
    r"|ТСН|НПБ|ППБ|ПУЭ|ОДМ|МДС|СТО|ФЗ|№|N|п\.|пп\.)\s*$"
)
# Fragments joined to one hit; a wrapped code needs one, a long sentence two.
_MAX_STEPS = 2
# Cut hits completed per search; each costs one MCP call per step.
_MAX_HITS = 12


def _body(fragment: dict[str, Any]) -> str:
    return (fragment.get("source_text") or fragment.get("text") or "").strip()


def is_cut(hit: dict[str, Any]) -> bool:
    """A shared-corpus text hit that ends mid-sentence and has a next fragment."""
    body = _body(hit)
    return bool(
        body
        and hit.get("id")
        and hit.get("next_id")
        and not hit.get("table_html")
        and hit.get("kind", "text") == "text"
        and not hit.get("user_id")
        and not _TERMINAL.search(body)
    )


def continues(hit: dict[str, Any], following: dict[str, Any]) -> bool:
    """Whether ``following`` is the rest of the sentence ``hit`` ends with."""
    body, rest = _body(hit), _body(following)
    if not rest or following.get("table_html"):
        return False
    start, end = following.get("char_start"), hit.get("char_end")
    if isinstance(start, int) and isinstance(end, int) and start < end:
        # Both were cut from one source paragraph.
        return True
    if _DESIGNATION_TAIL.search(body):
        return rest[:1].isdigit()
    return rest[:1].islower()


def _join(hit: dict[str, Any], following: dict[str, Any]) -> None:
    for key in ("text", "source_text"):
        if hit.get(key):
            rest = following.get(key) or following.get("text") or ""
            hit[key] = hit[key].rstrip() + " " + rest.strip()
    if isinstance(following.get("char_end"), int):
        hit["char_end"] = max(hit.get("char_end") or 0, following["char_end"])
    hit["next_id"] = following.get("next_id")
    hit.setdefault("continued_by", []).append(following["id"])


async def complete_cut_fragments(client, hits: list[dict]) -> list[dict]:
    """``hits`` with cut fragments completed; joined fragments leave the list."""
    get_node = getattr(client, "get_node", None)
    if get_node is None or not any(is_cut(hit) for hit in hits):
        return hits
    hits = [dict(hit) for hit in hits]

    async def complete(hit: dict[str, Any]) -> None:
        node_id = hit["id"]
        for _ in range(_MAX_STEPS):
            try:
                following = (await get_node(node_id)).get("next")
            # A completion only adds text; the hit stays usable without it.
            except Exception as exc:
                logger.warning("DVD fragment {} not completed: {}", node_id, exc)
                return
            if (
                not following
                or not following.get("id")
                or not continues(hit, following)
            ):
                return
            _join(hit, following)
            if _TERMINAL.search(_body(hit)):
                return
            node_id = following["id"]

    await asyncio.gather(
        *(complete(hit) for hit in [h for h in hits if is_cut(h)][:_MAX_HITS])
    )
    joined = {node for hit in hits for node in hit.get("continued_by", [])}
    if joined:
        logger.info("DVD completed cut fragments with {}", sorted(joined))
    return [hit for hit in hits if hit.get("id") not in joined]
