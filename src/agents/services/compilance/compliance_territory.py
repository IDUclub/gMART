"""Which NormGraph documents are in force on the scenario's territory.

IDU_DVD knows where a document applies: ``list_documents(scenario_id)`` returns the
shared documents in force under the scenario's project boundary — its municipal,
regional and federal documents. NormGraph documents are synced from IDU_DVD and
carry its ``doc_id``, so a NormGraph document is in force when its ``doc_id`` (or,
for documents synced before the id was kept, its name) is in that list. Nothing
is assumed in force when IDU_DVD cannot say: the filter is strict.

A check of user files without a scenario has no project boundary in Urban API; the
boundary from the files names the territories instead (``resolve_geometry``): IDU
MCP finds the deepest Urban API territories it lies in, and IDU_DVD lists the
documents of those territories and of every territory above them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from src.agents.services.compilance.compliance_scope import (
    DOCUMENT_POOL_LIMIT,
    normalized,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
    from src.agents.mcp_clients.idu_mcp_client import IduMcpClient
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient

Place = Literal["scenario", "files"]
_PLACES = {"scenario": "сценария", "files": "проекта из загруженных файлов"}
_UNAVAILABLE = (
    "Не удалось определить документы, действующие на территории {place}: "
    "{reason}. Без этого нельзя отобрать применимые нормы, поэтому нормы не "
    "применялись."
)
_NO_DOCUMENTS = (
    "На территории {place} не найдено действующих документов с нормами из графа "
    "норм. Нормы не применялись."
)


@dataclass(frozen=True)
class TerritoryDocuments:
    """``ok`` narrows norms to ``allowed``; any other status stops the run."""

    status: Literal["ok", "unavailable", "empty"]
    allowed: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    message: str | None = None
    place: Place = "scenario"
    # Names of the Urban API territories found from the files' boundary.
    territories: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "allowed": list(self.allowed),
            "excluded": list(self.excluded),
            "message": self.message,
            "place": self.place,
            "territories": list(self.territories),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerritoryDocuments":
        return cls(
            status=data["status"],
            allowed=tuple(data.get("allowed") or ()),
            excluded=tuple(data.get("excluded") or ()),
            message=data.get("message"),
            place="files" if data.get("place") == "files" else "scenario",
            territories=tuple(data.get("territories") or ()),
        )

    def note(self) -> str:
        """One line for the answer: how many graph documents the territory kept."""
        where = _PLACES[self.place]
        if self.territories:
            where += " (" + ", ".join(self.territories) + ")"
        return (
            f"Учтены только документы, действующие на территории {where}: "
            f"{len(self.allowed)}; не действуют на ней: {len(self.excluded)}."
        )


class ComplianceTerritoryFilter:
    async def resolve(
        self,
        dvd_client: "DvdMcpClient | None",
        normgraph_client: "NormGraphMcpClient",
        scenario_id: int,
    ) -> TerritoryDocuments:
        if dvd_client is None:
            return _unavailable("сервис документов IDU_DVD не подключён")
        try:
            in_force = await dvd_client.list_documents(scenario_id=scenario_id)
        except Exception as exc:  # the run stops with the reason instead of failing
            logger.exception("IDU_DVD territory documents lookup failed")
            return _unavailable(f"ошибка сервиса документов IDU_DVD ({exc})")
        return await self._match(
            in_force, normgraph_client, f"scenario_id={scenario_id}"
        )

    async def resolve_geometry(
        self,
        idu_client: "IduMcpClient",
        dvd_client: "DvdMcpClient | None",
        normgraph_client: "NormGraphMcpClient",
        geometry: dict[str, Any] | None,
    ) -> TerritoryDocuments:
        """Documents in force where a boundary from the user's files lies."""
        if geometry is None:
            return _unavailable(
                "в файлах нет ни границы проекта, ни объектов проекта", place="files"
            )
        if dvd_client is None:
            return _unavailable("сервис документов IDU_DVD не подключён", place="files")
        try:
            found = await idu_client.territories_by_geometry(geometry)
        except Exception as exc:
            logger.exception("Urban API territories by geometry lookup failed")
            return _unavailable(
                f"не удалось найти территорию Urban API по границе ({exc})",
                place="files",
            )
        territories = (found or {}).get("territories") or []
        territory_ids = [int(item["territory_id"]) for item in territories]
        if not territory_ids:
            return _unavailable(
                "граница проекта не попадает ни в одну территорию Urban API",
                place="files",
            )
        try:
            in_force = await dvd_client.list_documents(territory_ids=territory_ids)
        except Exception as exc:
            logger.exception("IDU_DVD territory documents lookup failed")
            return _unavailable(
                f"ошибка сервиса документов IDU_DVD ({exc})", place="files"
            )
        result = await self._match(
            in_force,
            normgraph_client,
            f"territory_ids={territory_ids}",
            place="files",
        )
        return replace(
            result,
            territories=tuple(
                str(item["name"]) for item in territories if item.get("name")
            ),
        )

    @staticmethod
    async def _match(
        in_force: list[dict[str, Any]],
        normgraph_client: "NormGraphMcpClient",
        where: str,
        *,
        place: Place = "scenario",
    ) -> TerritoryDocuments:
        doc_ids = {str(item["doc_id"]) for item in in_force if item.get("doc_id")}
        names = {normalized(item["name"]) for item in in_force if item.get("name")}
        graph_documents = await normgraph_client.list_restriction_documents(
            limit=DOCUMENT_POOL_LIMIT
        )
        allowed: list[str] = []
        excluded: list[str] = []
        for item in graph_documents:
            name = item.get("name")
            if not name:
                continue
            bucket = (
                allowed
                if str(item.get("doc_id") or "") in doc_ids or normalized(name) in names
                else excluded
            )
            if name not in bucket:
                bucket.append(name)
        logger.info(
            "Compliance territory {} dvd_documents={} "
            "graph_documents={} allowed={} excluded={}",
            where,
            len(in_force),
            len(graph_documents),
            len(allowed),
            len(excluded),
        )
        if not allowed:
            return TerritoryDocuments(
                status="empty",
                excluded=tuple(excluded),
                message=_NO_DOCUMENTS.format(place=_PLACES[place]),
                place=place,
            )
        return TerritoryDocuments(
            status="ok", allowed=tuple(allowed), excluded=tuple(excluded), place=place
        )


def _unavailable(reason: str, *, place: Place = "scenario") -> TerritoryDocuments:
    return TerritoryDocuments(
        status="unavailable",
        message=_UNAVAILABLE.format(place=_PLACES[place], reason=reason),
        place=place,
    )
