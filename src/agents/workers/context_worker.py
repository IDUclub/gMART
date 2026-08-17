"""Asynchronously summarize completed chats without blocking the request pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import socket
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from src.agents.common.api_handlers.json_api_handler import JsonApiHandler
from src.agents.model_clients.factory import build_llm_adapter
from src.agents.services.restriction_catalog import strip_json_fence


class ContextContent(BaseModel):
    summary: str = Field(max_length=14000)
    structured: dict[str, Any] = Field(default_factory=dict)


class ContextWorker:
    def __init__(
        self,
        chat_storage_url: str,
        internal_api_key: str,
        llm_host: str,
        *,
        worker_id: str | None = None,
    ) -> None:
        self.api = JsonApiHandler(chat_storage_url, max_retries=3)
        self.headers = {"X-Internal-API-Key": internal_api_key}
        self.llm = build_llm_adapter(llm_host)
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"

    async def run_forever(self) -> None:
        logger.info(f"Context worker {self.worker_id} started")
        while True:
            try:
                worked = await self.run_once()
            except Exception as exc:  # worker loop must survive a downstream outage
                logger.exception(f"Context worker loop failed: {exc}")
                worked = False
            if not worked:
                await asyncio.sleep(5)

    async def run_once(self) -> bool:
        job = await self.api.post(
            "/api/v1/internal/chat_context/jobs/claim",
            headers=self.headers,
            data={"worker_id": self.worker_id, "lease_seconds": 600},
        )
        if not job:
            return False
        job_id = job["job_id"]
        try:
            content = await self._summarize_job(job)
            await self.api.post(
                f"/api/v1/internal/chat_context/jobs/{job_id}/complete",
                headers=self.headers,
                data={"worker_id": self.worker_id, "content": content.model_dump()},
            )
        except Exception as exc:
            logger.warning(f"Context job {job_id} failed: {exc}")
            await self.api.post(
                f"/api/v1/internal/chat_context/jobs/{job_id}/fail",
                headers=self.headers,
                data={"worker_id": self.worker_id, "error": str(exc)[:2000]},
            )
        return True

    async def _summarize_job(self, job: dict) -> ContextContent:
        """Fold an arbitrary message tail into the context in bounded chunks."""

        after_seq: int | None = None
        current: ContextContent | None = None
        while True:
            params: dict[str, Any] = {
                "worker_id": self.worker_id,
                "tail_limit": 100,
            }
            if after_seq is not None:
                params["after_seq"] = after_seq
            source = (
                await self.api.get(
                    f"/api/v1/internal/chat_context/jobs/{job['job_id']}/source",
                    headers=self.headers,
                    params=params,
                )
                or {}
            )
            if current is None:
                current = ContextContent.model_validate(source.get("content") or {})
            tail = source.get("tail") or []
            if tail:
                current = await self._summarize(job, current, tail)
            if not source.get("tail_has_more"):
                return current
            next_after = source.get("tail_next_after_seq")
            if not isinstance(next_after, int) or next_after == after_seq:
                raise ValueError(
                    "ChatStorage вернул некорректный курсор хвоста контекста"
                )
            after_seq = next_after

    async def _summarize(
        self,
        job: dict,
        previous: ContextContent,
        messages: list[dict],
    ) -> ContextContent:
        tail = self._compact_tail(messages)
        prompt = f"""Обнови контекст диалога городского аналитического агента.
Сохраняй только подтверждённые факты, решения пользователя, актуальные маппинги,
метаданные наборов, выполненные результаты и открытые вопросы. Не выдавай неподтверждённые
предположения за факты. Не сохраняй JWT, секреты, геометрии, полные таблицы и внутренние
рассуждения модели. Неудачные попытки сведи к короткому списку отдельно.

Предыдущий контекст:
{previous.model_dump_json()[:18000]}

Новые сообщения до seq={job['target_seq']}:
{json.dumps(tail, ensure_ascii=False)[:24000]}

Верни JSON {{summary, structured}}. summary — компактный русский текст; structured содержит
массивы verified_facts, user_decisions, mappings, datasets, completed_tasks, open_questions,
failed_attempts. Общий ответ не должен превышать примерно 6000 токенов."""
        response = await self.llm.chat(
            model=job["model"],
            messages=[{"role": "system", "content": prompt}],
            think=False,
            format=ContextContent.model_json_schema(),
            options={"temperature": 0, "num_predict": 6000},
            reasoning_effort="medium",
        )
        raw = (response.get("message") or {}).get("content") or ""
        try:
            return ContextContent.model_validate(json.loads(strip_json_fence(raw)))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid context summary: {exc}") from exc

    @staticmethod
    def _compact_tail(messages: list[dict]) -> list[dict]:
        compact = []
        for message in messages:
            parts = []
            for part in message.get("parts") or []:
                kind = part.get("kind")
                payload = part.get("payload") or {}
                if kind == "text":
                    parts.append(
                        {"kind": kind, "text": str(payload.get("text") or "")[:8000]}
                    )
                elif kind in {
                    "plan",
                    "plan_revision",
                    "artifact_ref",
                    "validation",
                    "failure",
                    "table",
                }:
                    parts.append({"kind": kind, "payload": payload})
            if parts:
                compact.append(
                    {
                        "seq": message.get("seq"),
                        "role": message.get("role"),
                        "parts": parts,
                    }
                )
        return compact


def main() -> None:
    url = os.getenv("CHAT_STORAGE")
    key = os.getenv("CONTEXT_INTERNAL_API_KEY")
    llm_host = os.getenv("OLLAMA_API_URL")
    if not url or not key or not llm_host:
        raise RuntimeError(
            "CHAT_STORAGE, CONTEXT_INTERNAL_API_KEY and OLLAMA_API_URL are required"
        )
    asyncio.run(ContextWorker(url, key, llm_host).run_forever())


if __name__ == "__main__":
    main()
