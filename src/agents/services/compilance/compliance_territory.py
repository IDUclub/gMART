"""Which NormGraph documents are in force on the scenario's territory.

IDU_DVD knows where a document applies: ``list_documents(scenario_id)`` returns the
shared documents in force under the scenario's project boundary — its municipal,
regional and federal documents. NormGraph documents are synced from IDU_DVD and
carry its ``doc_id``, so a NormGraph document is in force when its ``doc_id`` (or,
for documents synced before the id was kept, its name) is in that list. Nothing
is assumed in force when IDU_DVD cannot say: the filter is strict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from src.agents.services.compilance.compliance_scope import (
    DOCUMENT_POOL_LIMIT,
    normalized,
)

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient
    from src.agents.mcp_clients.normgraph_mcp_client import NormGraphMcpClient

_UNAVAILABLE = (
    "Не удалось определить документы, действующие на территории сценария: "
    "{reason}. Без этого нельзя отобрать применимые нормы, поэтому нормы не "
    "применялись."
)
_NO_DOCUMENTS = (
    "На территории сценария не найдено действующих документов с нормами из графа "
    "норм. Нормы не применялись."
)


@dataclass(frozen=True)
class TerritoryDocuments:
    """``ok`` narrows norms to ``allowed``; any other status stops the run."""

    status: Literal["ok", "unavailable", "empty"]
    allowed: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "allowed": list(self.allowed),
            "excluded": list(self.excluded),
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TerritoryDocuments":
        return cls(
            status=data["status"],
            allowed=tuple(data.get("allowed") or ()),
            excluded=tuple(data.get("excluded") or ()),
            message=data.get("message"),
        )

    def note(self) -> str:
        """One line for the answer: how many graph documents the territory kept."""
        return (
            "Учтены только документы, действующие на территории сценария: "
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
            "Compliance territory scenario_id={} dvd_documents={} "
            "graph_documents={} allowed={} excluded={}",
            scenario_id,
            len(in_force),
            len(graph_documents),
            len(allowed),
            len(excluded),
        )
        if not allowed:
            return TerritoryDocuments(
                status="empty", excluded=tuple(excluded), message=_NO_DOCUMENTS
            )
        return TerritoryDocuments(
            status="ok", allowed=tuple(allowed), excluded=tuple(excluded)
        )


def _unavailable(reason: str) -> TerritoryDocuments:
    return TerritoryDocuments(
        status="unavailable", message=_UNAVAILABLE.format(reason=reason)
    )
