"""Shared event envelopes and per-step observation for orchestration streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.agents.api_clients.chat_storage_client.request_models import (
    TablePartRequest,
    TablePayload,
)
from src.agents.services.orchestrator.orchestrator_catalog import AGENT_CATALOG
from src.agents.services.service_entities.orchestrator_plan import (
    OrchestratorAgent,
    OrchestratorPlan,
    OrchestratorStep,
)

SUPPRESSED_INNER_EVENTS = {"pipeline_started", "service_event"}
TERMINAL_STATUSES = {
    "error": "failed",
    "pipeline_failed": "failed",
    "clarification": "needs_clarification",
    "pipeline_suspended": "suspended",
}


def agent_title(agent: OrchestratorAgent | str) -> str:
    if agent == OrchestratorAgent.URBAN_ANALYSIS_REPORT:
        return "Итоговый отчёт градостроительного анализа"
    entry = AGENT_CATALOG.get(OrchestratorAgent(agent))
    return entry.title if entry else str(agent)


def pipeline_started_event(request_id: str) -> dict:
    return {"type": "pipeline_started", "content": {"request_id": request_id}}


def status_event(status: str, text: str) -> dict:
    return {"type": "status", "content": {"status": status, "text": text}}


def plan_event(plan: OrchestratorPlan) -> dict:
    return {
        "type": "plan",
        "content": {
            "steps": [
                {
                    "step": number,
                    "agent": step.agent.value,
                    "agent_title": agent_title(step.agent),
                    "task": step.task,
                }
                for number, step in enumerate(plan.steps, start=1)
            ]
        },
    }


def step_started_event(
    number: int, step: OrchestratorStep, request_id: str, task: str
) -> dict:
    return {
        "type": "step_started",
        "content": {
            "step": number,
            "agent": step.agent.value,
            "step_request_id": request_id,
            "task": task,
        },
    }


def step_event(number: int, step: OrchestratorStep, item: dict) -> dict:
    return {
        "type": "step_event",
        "content": {
            "step": number,
            "agent": step.agent.value,
            "event": item,
        },
    }


def step_finished_event(
    number: int, step: OrchestratorStep, status: str, summary: str
) -> dict:
    return {
        "type": "step_finished",
        "content": {
            "step": number,
            "agent": step.agent.value,
            "status": status,
            "summary": summary,
        },
    }


def clarification_event(question: str) -> dict:
    return {"type": "clarification", "content": {"question": question}}


def summary_step(
    number: int, step: OrchestratorStep, status: str, summary: str
) -> dict:
    return {
        "step": number,
        "agent": step.agent.value,
        "task": step.task,
        "status": status,
        "summary": summary,
    }


def final_event(steps: list[dict]) -> dict:
    return {"type": "orchestrator_final", "content": {"steps": steps}}


def chat_created_event(chat_id: str, chat_title: str) -> dict:
    return {
        "type": "service_event",
        "content": {
            "event_type": "storage_event",
            "event": {
                "storage_event_type": "chat_created",
                "chat_id": chat_id,
                "chat_title": chat_title,
            },
        },
    }


def collect_digest(collected: dict[str, Any], item: dict) -> None:
    content = item.get("content") or {}
    if not isinstance(content, dict):
        return
    event_id = content.get("event_id")
    if event_id is not None:
        seen = collected.setdefault("event_ids", set())
        if event_id in seen:
            return
        seen.add(event_id)
    if item.get("type") == "feature_collection":
        collected["feature_collection"] = {
            key: value for key, value in content.items() if key != "event_id"
        }
    elif item.get("type") == "status" and content.get("status") == "economy":
        collected["parks_economy"] = content.get("economy") or {}
    if item.get("type") == "chunk":
        iteration = int(content.get("iteration") or 0)
        collected["chunks"].setdefault(iteration, []).append(content.get("text") or "")
    elif item.get("type") == "feature_collection" and content.get("name"):
        collected["notes"].append(f"Построен слой «{content['name']}».")
    elif item.get("type") == "table" and (content.get("title") or content.get("name")):
        title = content.get("title") or content.get("name")
        collected["notes"].append(f"Сформирована таблица «{title}».")


def digest_from_collected(collected: dict[str, Any], limit: int = 1500) -> str:
    chunks = collected["chunks"]
    text = "".join(chunks[max(chunks)]).strip() if chunks else ""
    digest = "\n".join(part for part in (text, " ".join(collected["notes"])) if part)
    return digest if len(digest) <= limit else digest[: limit - 1].rstrip() + "…"


def table_part(item: dict) -> TablePartRequest | None:
    if item.get("type") != "table" or not isinstance(item.get("content"), dict):
        return None
    try:
        return TablePartRequest(
            kind="table", payload=TablePayload.model_validate(item["content"])
        )
    except Exception as exc:
        logger.warning(f"Orchestrator: could not persist table event: {exc}")
        return None


@dataclass
class StepObservation:
    """Collect evidence and terminal state; drafts never survive failed steps."""

    collected: dict = field(default_factory=lambda: {"chunks": {}, "notes": []})
    tables: list[TablePartRequest] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    status: str = "completed"
    reason: str = ""

    def observe(self, item: dict) -> bool:
        """Return whether the item should be forwarded as a nested event."""
        kind = item.get("type")
        if kind in SUPPRESSED_INNER_EVENTS:
            return False
        if kind == "file":
            self.files.append(item)
            return False
        collect_digest(self.collected, item)
        part = table_part(item)
        if part is not None:
            self.tables.append(part)
        if kind in TERMINAL_STATUSES:
            self.status = TERMINAL_STATUSES[kind]
            content = item.get("content") or {}
            self.reason = next(
                (
                    content[key]
                    for key in (
                        "question",
                        "text",
                        "message",
                        "detail",
                        "error_message",
                        "reason",
                    )
                    if isinstance(content.get(key), str) and content[key]
                ),
                "",
            )
            if not self.reason and isinstance(content.get("reasons"), list):
                self.reason = "; ".join(str(reason) for reason in content["reasons"])
            if self.status == "needs_clarification":
                self.reason = (
                    content.get("question")
                    or content.get("text")
                    or "Для выполнения шага требуется уточнение пользователя."
                )
                self.collected = {
                    "chunks": {},
                    "notes": [self.reason],
                }
        return True

    def summary(self, limit: int = 1500) -> str:
        if self.status in {"failed", "suspended"}:
            default = (
                "Шаг не выполнен: агент сообщил об ошибке. Результат не подтверждён."
                if self.status == "failed"
                else "Шаг приостановлен. Результат ещё не подтверждён."
            )
            text = self.reason or default
            return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
        return digest_from_collected(self.collected, limit)
