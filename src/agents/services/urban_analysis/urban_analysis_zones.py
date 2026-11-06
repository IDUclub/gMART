"""Select scenario functional zones, preserving trusted API constraints."""

from dataclasses import dataclass, field

from loguru import logger

from src.agents.dto.pzz_request_dto import PzzInputs

_SOURCES = {"pzz": "PZZ", "user": "User", "osm": "OSM"}
_PRIORITY = {"PZZ": 3, "User": 2, "OSM": 1}


@dataclass
class ZoneSelection:
    inputs: PzzInputs | None = None
    status: str = "completed"
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    def report_data(self) -> dict:
        return {
            "year": self.inputs.year if self.inputs else None,
            "source": self.inputs.source if self.inputs else None,
            "notes": self.notes,
        }


async def resolve_zones(
    urban_api_client,
    pzz_service,
    *,
    token,
    scenario_id,
    model,
    user_query,
    pzz_inputs=None,
) -> ZoneSelection:
    supplied = PzzInputs.model_validate(pzz_inputs or {})
    # Preserve scenario options; uploaded layers cannot change the mode.
    trusted = PzzInputs(
        mode="scenario",
        year=supplied.year,
        source=supplied.source,
        physical_object_type_id=supplied.physical_object_type_id,
        group_by=supplied.group_by,
        priority=supplied.priority,
        force_recompute=supplied.force_recompute,
    )
    if trusted.year is not None and trusted.source is not None:
        return ZoneSelection(inputs=trusted)
    try:
        data = await urban_api_client.get_scenario_functional_zone_sources(
            token, scenario_id
        )
    except Exception:
        logger.exception("Urban analysis functional zone source discovery failed")
        return ZoneSelection(
            status="failed",
            reason="Не удалось получить источники функциональных зон сценария из Urban API.",
        )
    pairs = set()
    for item in data or []:
        source = _SOURCES.get(
            str(item.get("source") or item.get("name") or "").casefold()
        )
        try:
            year = int(item["year"])
        except (KeyError, ValueError, TypeError):
            continue
        if source:
            pairs.add((source, year))
    notes = []
    inferred = trusted
    if pairs:
        try:
            inferred = await pzz_service.resolve_inputs(
                model, user_query, scenario_id, trusted
            )
        except Exception:
            logger.exception("Urban analysis functional zone input resolution failed")
            notes.append(
                "Год и источник не удалось извлечь из запроса; использован автовыбор зон."
            )
    candidates = [
        (source, year)
        for source, year in pairs
        if (trusted.source is None or source == trusted.source)
        and (trusted.year is None or year == trusted.year)
    ]
    from_text = [
        (source, year)
        for source, year in candidates
        if (inferred.source is None or source == inferred.source)
        and (inferred.year is None or year == inferred.year)
    ]
    if from_text:
        candidates = from_text
    elif inferred.year != trusted.year or inferred.source != trusted.source:
        notes.append(
            "Год или источник из текста не соответствует доступным зонам; использован автовыбор с учётом параметров API."
        )
    if not candidates:
        return ZoneSelection(
            status="needs_clarification",
            notes=notes,
            reason=(
                "У сценария нет подходящих функциональных зон. Загрузите их и повторите проверку."
            ),
        )
    source, year = max(candidates, key=lambda pair: (_PRIORITY[pair[0]], pair[1]))
    return ZoneSelection(
        inputs=trusted.model_copy(update={"year": year, "source": source}), notes=notes
    )
