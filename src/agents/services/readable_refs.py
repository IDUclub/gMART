"""Human-readable names for what the agents mention in their answers.

System identifiers (Urban API entity ids, composite object refs, restriction hashes,
fragment UUIDs) identify records for code and for the map. The user reads names:
documents, clauses, objects, scenarios. Keep identifiers out of every text the model
drafts from and out of every message the program writes itself.
"""

from __future__ import annotations

import re
from typing import Any

NO_SYSTEM_IDS_RULE = (
    "Называй документы, пункты, объекты и сценарии их названиями. Не пиши системные "
    "идентификаторы (id, UUID, хеши, коды записей и шагов), даже если они встретились "
    "в данных или в истории диалога."
)

# Urban API names an object without a name "(Безымянный физический объект)".
_PLACEHOLDER_NAME = re.compile(r"^\(?\s*безымянн", re.IGNORECASE)


def readable_name(value: Any) -> str | None:
    """A name worth showing, or ``None`` for an empty or placeholder name."""
    name = " ".join(str(value or "").split())
    if not name or _PLACEHOLDER_NAME.match(name):
        return None
    return name


def object_label(ref: dict[str, Any] | None) -> str:
    """Name an object from its ``object_ref`` without its identifiers.

    An unnamed object is called by its layer («Жилой дом (без названия)»); the row
    number and the map tell such objects apart, not a database key.
    """
    ref = ref or {}
    layer = readable_name(ref.get("layer"))
    name = readable_name(ref.get("name"))
    identifier = str(ref.get("id") or "")
    entity_id = ref.get("entity_id")
    # Refs from before idu_mcp stopped naming unnamed objects «<layer> #<id>».
    numbered = layer and entity_id is not None and name == f"{layer} #{entity_id}"
    if name and name != identifier and not numbered:
        return name
    return f"{layer} (без названия)" if layer else "Объект без названия"
