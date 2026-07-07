from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

from loguru import logger
from ollama import ChatResponse

from src.agents.api_clients.chat_storage_client.chat_storage_client import (
    ChatStorageApiClient,
)
from src.agents.api_clients.chat_storage_client.entities import RoleEnum
from src.agents.api_clients.chat_storage_client.request_models import (
    TextPartRequest,
    TextPayload,
    ToolCall,
    ToolCallPartRequest,
    ToolCallPayload,
)
from src.agents.api_clients.urban_api_client.urban_api_client import UrbanApiClient
from src.agents.services.base_llm_service import BaseLlmService
from src.agents.services.dvd_context import DvdContextBuilder
from src.agents.services.dvd_reasoning import AnswerCritic, RetrievalPlanner
from src.agents.services.pipeline_state import PipelineStateStore, PipelineStatus

if TYPE_CHECKING:
    from src.agents.mcp_clients.dvd_mcp_client import DvdMcpClient

_MCP_SOURCE = "DVD_MCP_URL"
_EXECUTION_MODE = "rag_search"
# Checkpoint key holding the iterative loop progress (so a reconnect can resume).
_QA_PROGRESS = "qa_progress"


class DvdRagService(BaseLlmService):
    """
    Iterative RAG agent over regulatory documents (IDU_DVD).

    For each round:
        1. RETRIEVAL_PLANNING — an LLM picks the search query, surface (text/table/all),
           number of fragments and neighbour-context width.
        2. SEARCHING — the chosen IDU_DVD search tool is called deterministically.
        3. ANSWER_DRAFTING — the answer is streamed to the client, grounded in the fragments.
        4. SELF_REVIEW — a critic LLM checks the draft against the fragments. If rejected,
           a new (refined) search query is planned and the answer is rewritten on fresh
           context. The loop repeats up to ``MAX_ITERATIONS`` rounds.

    Every draft is streamed (each chunk tagged with its ``iteration``); status events report
    the self-correction. With full integration the final answer is persisted to ChatStorage.

    Reconnect: every emitted event is buffered in Redis (``PipelineStateStore``) keyed by a
    ``request_id`` that is announced via the first ``pipeline_started`` event. If the SSE
    connection drops, the client re-requests with the same ``request_id``: buffered events are
    replayed and the loop resumes from the last completed iteration (checkpointed in Redis).
    """

    MAX_ITERATIONS = 3

    def __init__(
        self,
        ollama_host: str,
        chat_storage_client: ChatStorageApiClient,
        urban_api_client: UrbanApiClient,
        state_store: PipelineStateStore,
    ) -> None:
        super().__init__(ollama_host, chat_storage_client, urban_api_client)
        self.planner = RetrievalPlanner(self.llm_client)
        self.critic = AnswerCritic(self.llm_client)
        self.context_builder = DvdContextBuilder()
        self.state_store = state_store

    # ------------------------------------------------------------------
    # Public entry point (reconnect handling + chat storage + history)
    # ------------------------------------------------------------------

    async def run_document_qa_pipeline(
        self,
        dvd_mcp_client: "DvdMcpClient",
        token: str,
        model: str,
        temperature: float,
        user_query: str,
        scenario_id: int | None = None,
        chat_id: str | None = None,
        request_id: str | None = None,
        persist_history: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        collected: dict[str, Any] = {
            "final_answer": "",
            "tool_calls": [],
            "newly_completed": False,
        }
        is_reconnect = request_id is not None and await self.state_store.exists(
            request_id
        )

        if is_reconnect:
            logger.info(
                f"DVD QA reconnect request_id={request_id}, replaying buffered events"
            )
            for event in await self.state_store.get_buffered_events(request_id):
                yield event
            stored = await self.state_store.get_state(request_id) or {}
            if not chat_id and stored.get("chat_id"):
                chat_id = stored["chat_id"]
            model = stored.get("model") or model
            if stored.get("temperature") is not None:
                temperature = stored["temperature"]
            user_query = stored.get("user_query") or user_query
            if stored.get("scenario_id") is not None:
                scenario_id = stored["scenario_id"]
        else:
            request_id = request_id or self.state_store.new_request_id()

        original_chat_id = chat_id

        if not is_reconnect:
            yield self._buf(request_id, self._pipeline_started_event(request_id))

            # No chat_id supplied → create a new chat tagged with scenario_id. The
            # project_id is resolved from scenario_id; if that lookup fails we warn the
            # client, drop the project filter, and keep going (the chat is still created).
            # A2A runs pass persist_history=False: no chat is created and nothing is
            # written to ChatStorage (history stays read-only).
            if not chat_id and persist_history:
                project_id: int | None = None
                if scenario_id is not None:
                    try:
                        project_id = (
                            await self.urban_api_client.get_project_by_scenario(
                                token, scenario_id
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            f"DVD QA: failed to resolve project_id for "
                            f"scenario_id={scenario_id}: {exc}"
                        )
                        yield self._buf(
                            request_id,
                            self._project_lookup_failed_event(scenario_id),
                        )
                try:
                    chat_id, title = await self.create_chat(
                        token,
                        model,
                        user_query,
                        additional_instructions=(
                            "Запрос направлен агенту вопросов по нормативной "
                            "документации (RAG по базе IDU_DVD)."
                        ),
                        scenario_id=scenario_id,
                        project_id=project_id,
                        resolve_project_id=False,
                    )
                    yield self._buf(
                        request_id, self._chat_created_event(chat_id, title)
                    )
                except Exception as exc:  # chat storage must not break the stream
                    logger.warning(f"DVD QA: failed to create chat: {exc}")
                    chat_id = None

            await self.state_store.create(
                request_id,
                chat_id=chat_id,
                user_query=user_query,
                scenario_id=scenario_id,
                model=model,
                temperature=temperature,
            )

        history: list[dict] = []
        if original_chat_id:
            try:
                chat_info = await self.get_chat_messages(token, original_chat_id)
                history = self.build_llm_history(
                    chat_info.messages, current_user_query=user_query
                )
            except Exception as exc:
                logger.warning(f"DVD QA: failed to fetch chat history: {exc}")

        # A follow-up question in an existing chat is persisted here — create_chat
        # stores only the first one. Runs after the history fetch so the current
        # question doesn't also enter the LLM context from storage, and is skipped
        # on reconnect (the original run already stored it). Chat storage failures
        # must not break the stream.
        if persist_history and not is_reconnect and original_chat_id:
            try:
                await self.add_single_message(
                    token,
                    original_chat_id,
                    RoleEnum.USER,
                    user_query,
                    scenario_id=scenario_id,
                )
            except Exception as exc:
                logger.warning(f"DVD QA: failed to persist user question: {exc}")

        async for event in self._run_qa_loop(
            dvd_mcp_client,
            model,
            temperature,
            user_query,
            history,
            collected,
            request_id,
        ):
            yield event

        # Persist only when this run actually produced the answer — never on a reconnect
        # that merely replayed an already-completed pipeline (avoids duplicate messages).
        if persist_history and collected.get("newly_completed"):
            self._schedule_persist_answer(token, chat_id, collected, scenario_id)

    # ------------------------------------------------------------------
    # Inner iterative loop (retrieve -> draft -> critique -> refine)
    # ------------------------------------------------------------------

    async def _run_qa_loop(
        self,
        dvd_mcp_client: "DvdMcpClient",
        model: str,
        temperature: float,
        user_query: str,
        history: list[dict],
        collected: dict[str, Any],
        request_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        checkpoint = await self.state_store.get_checkpoint(request_id)
        progress = checkpoint.get(_QA_PROGRESS) or {}
        collected["tool_calls"] = list(progress.get("tool_calls", []))

        if progress.get("accepted"):
            # The pipeline already produced the final answer before the disconnect; its
            # terminal events were buffered and have just been replayed — nothing to redo.
            collected["final_answer"] = progress.get("final_answer", "")
            await self.state_store.set_status(request_id, PipelineStatus.DONE)
            return

        prev_critique: str | None = progress.get("prev_critique")
        prev_query: str | None = progress.get("prev_query")
        start_iteration = int(progress.get("completed_iterations", 0)) + 1
        final_iteration = start_iteration

        for iteration in range(start_iteration, self.MAX_ITERATIONS + 1):
            final_iteration = iteration
            is_last = iteration == self.MAX_ITERATIONS

            # ── Step 1: plan retrieval (LLM chooses query + context size) ──
            yield self._buf(
                request_id,
                self._status(
                    "retrieval_planning",
                    f"Подбираю параметры поиска (попытка {iteration})…",
                ),
            )
            plan = await self.planner.build_plan(
                model, user_query, history, prev_critique, prev_query
            )

            # ── Step 2: search IDU_DVD ────────────────────────────────────
            yield self._buf(
                request_id,
                self._status(
                    "searching",
                    f"Ищу в нормативной базе: «{plan.search_query}» "
                    f"(тип: {plan.kind}, фрагментов: {plan.limit}, "
                    f"контекст: ±{plan.context_height})…",
                ),
            )
            search_args = {
                "query": plan.search_query,
                "limit": plan.limit,
                "context_height": plan.context_height,
            }
            search_result = await dvd_mcp_client.search(
                plan.search_query,
                kind=plan.kind,
                limit=plan.limit,
                context_height=plan.context_height,
            )
            hits = search_result.get("hits") or []

            tool_call = self._search_tool_call(
                dvd_mcp_client.tool_name_for_kind(plan.kind), search_args
            )
            collected["tool_calls"].append(tool_call)
            yield self._buf(
                request_id,
                self._tool_call(_EXECUTION_MODE, [tool_call], mcp_source=_MCP_SOURCE),
            )

            if not hits and not is_last:
                yield self._buf(
                    request_id,
                    self._status(
                        "searching",
                        "Релевантных фрагментов не найдено, переформулирую запрос…",
                    ),
                )
                prev_critique = (
                    "Поиск не дал результатов. Переформулируй поисковый запрос: "
                    "используй синонимы, официальную терминологию, более общие "
                    "или более узкие формулировки."
                )
                prev_query = plan.search_query
                await self._save_progress(
                    request_id,
                    collected,
                    completed_iterations=iteration,
                    accepted=False,
                    prev_critique=prev_critique,
                    prev_query=prev_query,
                )
                continue

            context = self.context_builder.build_context(hits)

            # ── Step 3: draft the answer (streamed) ───────────────────────
            yield self._buf(
                request_id,
                self._status(
                    "answer_drafting", f"Формирую ответ (попытка {iteration})…"
                ),
            )
            revision_note = prev_critique if iteration > 1 else None
            draft_parts: list[str] = []
            async for chunk_event in self._generate_answer(
                model,
                user_query,
                context,
                temperature,
                history,
                iteration,
                revision_note,
            ):
                if text := chunk_event["content"]["text"]:
                    draft_parts.append(text)
                yield self._buf(request_id, chunk_event)
            draft = "".join(draft_parts).strip()

            # ── Step 4: critique (last round is always accepted) ──────────
            if not is_last:
                yield self._buf(
                    request_id,
                    self._status(
                        "self_review",
                        "Проверяю ответ на полноту и соответствие источникам…",
                    ),
                )
                verdict = await self.critic.review(model, user_query, context, draft)
            else:
                verdict = None

            if is_last or (verdict is not None and verdict.satisfied):
                collected["final_answer"] = draft
                collected["newly_completed"] = True
                # Emit terminal events BEFORE checkpointing "accepted" so a reconnect that
                # sees accepted=True always has the done chunk in the replay buffer.
                yield self._buf(
                    request_id, self._status("finalizing", "Ответ сформирован")
                )
                yield self._buf(
                    request_id, self._chunk("", done=True, iteration=iteration)
                )
                await self._save_progress(
                    request_id,
                    collected,
                    completed_iterations=iteration,
                    accepted=True,
                    final_answer=draft,
                    final_iteration=iteration,
                )
                await self.state_store.set_status(request_id, PipelineStatus.DONE)
                return

            critique_text = (verdict.critique or "ответ недостаточно обоснован").strip()
            yield self._buf(
                request_id,
                self._status(
                    "self_review",
                    f"Модель не удовлетворена ответом: {critique_text} "
                    "Переформулирую запрос и переписываю ответ…",
                ),
            )
            prev_critique = critique_text
            prev_query = verdict.refined_search_query or plan.search_query
            await self._save_progress(
                request_id,
                collected,
                completed_iterations=iteration,
                accepted=False,
                prev_critique=prev_critique,
                prev_query=prev_query,
            )

        # Defensive: the last iteration always accepts above, so this is normally unreachable
        # (covers an empty resume range).
        collected["newly_completed"] = True
        yield self._buf(request_id, self._status("finalizing", "Ответ сформирован"))
        yield self._buf(
            request_id, self._chunk("", done=True, iteration=final_iteration)
        )
        await self.state_store.set_status(request_id, PipelineStatus.DONE)

    # ------------------------------------------------------------------
    # LLM answer generation (streaming)
    # ------------------------------------------------------------------

    async def _generate_answer(
        self,
        model: str,
        user_query: str,
        context: str,
        temperature: float,
        history: list[dict],
        iteration: int,
        revision_note: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        system = (
            "Ты — ассистент-эксперт по нормативной документации в сфере градостроительства "
            "и городского планирования. Отвечай на вопрос пользователя СТРОГО на основании "
            "приведённых фрагментов нормативных документов. Правила:\n"
            "- Не выдумывай нормы, цифры и положения, которых нет во фрагментах.\n"
            "- Если данных во фрагментах недостаточно — прямо сообщи об этом.\n"
            "- Ссылайся на источники: название документа, редакцию и номер пункта "
            "(можно через номера [1], [2]… из фрагментов).\n"
            "- Отвечай на русском языке, ясно и по существу.\n\n"
            f"Фрагменты нормативных документов:\n"
            f"{context or '(релевантные фрагменты не найдены)'}"
        )
        if revision_note:
            system += (
                "\n\nУчти замечание к предыдущей версии ответа и исправь его: "
                f"{revision_note}"
            )
        messages = [
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        response_buffer: list[str] = []
        async for part in await self.llm_client.chat(
            model,
            messages,
            options={"temperature": temperature},
            stream=True,
        ):
            part: ChatResponse
            if part.message.content:
                response_buffer.append(part.message.content)
            # ``done`` is forced False here; finality is decided by the loop after the
            # critic accepts a draft (a single done=True chunk is emitted at the end).
            yield self._chunk(
                part.message.content or "", done=False, iteration=iteration
            )
        logger.debug(
            f"DVD answer draft {iteration} [{model}]: {''.join(response_buffer)}"
        )

    # ------------------------------------------------------------------
    # Redis state helpers (event buffering + resume checkpoint)
    # ------------------------------------------------------------------

    def _buf(self, request_id: str, event: dict) -> dict:
        """Fire-and-forget buffer the event for reconnect replay, then return it."""
        asyncio.create_task(self.state_store.buffer_event(request_id, event))
        return event

    async def _save_progress(
        self,
        request_id: str,
        collected: dict[str, Any],
        *,
        completed_iterations: int,
        accepted: bool,
        final_answer: str | None = None,
        final_iteration: int | None = None,
        prev_critique: str | None = None,
        prev_query: str | None = None,
    ) -> None:
        await self.state_store.save_checkpoint(
            request_id,
            _QA_PROGRESS,
            {
                "completed_iterations": completed_iterations,
                "tool_calls": collected["tool_calls"],
                "accepted": accepted,
                "final_answer": final_answer,
                "final_iteration": final_iteration,
                "prev_critique": prev_critique,
                "prev_query": prev_query,
            },
        )

    # ------------------------------------------------------------------
    # Chat storage persistence (final answer only — drafts are not saved)
    # ------------------------------------------------------------------

    def _schedule_persist_answer(
        self,
        token: str,
        chat_id: str | None,
        collected: dict[str, Any],
        scenario_id: int | None,
    ) -> None:
        if not chat_id or not collected.get("final_answer"):
            return
        task = asyncio.create_task(
            self._persist_answer(token, chat_id, dict(collected), scenario_id)
        )
        task.add_done_callback(self._log_persist_result)

    async def _persist_answer(
        self,
        token: str,
        chat_id: str,
        collected: dict[str, Any],
        scenario_id: int | None,
    ) -> None:
        parts: list[TextPartRequest | ToolCallPartRequest] = []
        if collected.get("tool_calls"):
            calls = [
                self._tool_call_to_storage(step, tc)
                for step, tc in enumerate(collected["tool_calls"], start=1)
            ]
            parts.append(
                ToolCallPartRequest(
                    kind="tool_call",
                    payload=ToolCallPayload(
                        execution_mode=_EXECUTION_MODE, calls=calls
                    ),
                    mcp_source=_MCP_SOURCE,
                )
            )
        parts.append(
            TextPartRequest(
                kind="text", payload=TextPayload(text=collected["final_answer"])
            )
        )
        await self.add_complex_message(
            token, chat_id, RoleEnum.ASSISTANT, parts, scenario_id=scenario_id
        )

    @staticmethod
    def _log_persist_result(task: asyncio.Task) -> None:
        try:
            task.result()
        except Exception as exc:
            logger.exception(f"DVD QA: failed to persist answer: {exc}")

    # ------------------------------------------------------------------
    # Event / tool-call helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _search_tool_call(tool_name: str, arguments: dict) -> dict:
        return {"function": {"name": tool_name, "arguments": arguments}}

    @staticmethod
    def _tool_call_to_storage(step: int, tool_call: dict) -> ToolCall:
        function_call = tool_call.get("function") or {}
        tool_name = function_call.get("name") or tool_call.get("name")
        arguments = function_call.get("arguments") or tool_call.get("arguments") or {}
        if not tool_name:
            raise ValueError(f"Tool call without tool name: {tool_call}")
        return ToolCall(step=step, tool_name=tool_name, arguments=arguments)

    @staticmethod
    def _pipeline_started_event(request_id: str) -> dict:
        return {"type": "pipeline_started", "content": {"request_id": request_id}}

    @staticmethod
    def _project_lookup_failed_event(scenario_id: int | None) -> dict:
        return {
            "type": "warning",
            "content": {
                "code": "project_id_unavailable",
                "scenario_id": scenario_id,
                "message": (
                    f"Не удалось получить идентификатор проекта (project_id) по "
                    f"scenario_id={scenario_id}. Фильтр проекта не будет сохранён, "
                    "выполнение запроса продолжается."
                ),
            },
        }

    @staticmethod
    def _status(status: str, text: str) -> dict:
        return {"type": "status", "content": {"status": status, "text": text}}

    @staticmethod
    def _chunk(text: str, done: bool, iteration: int) -> dict:
        return {
            "type": "chunk",
            "content": {"text": text, "done": done, "iteration": iteration},
        }

    @staticmethod
    def _tool_call(
        execution_mode: str, tool_calls: list[dict], mcp_source: str | None = None
    ) -> dict:
        content: dict = {"execution_mode": execution_mode, "tool_calls": tool_calls}
        if mcp_source is not None:
            content["mcp_source"] = mcp_source
        return {"type": "tool_call", "content": content}

    @staticmethod
    def _chat_created_event(chat_id: str, chat_title: str) -> dict:
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
