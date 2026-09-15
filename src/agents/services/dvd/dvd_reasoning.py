from __future__ import annotations

import json
import os
import re
from typing import Any, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.dvd.document_reference import parse_reference, wants_full_quote
from src.agents.services.dvd.retrieval_scope import apply_scope
from src.agents.services.restriction.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.dvd_plan import (
    CriticVerdict,
    RetrievalPlan,
    SearchKind,
    validate_retrieval_plan,
)

from .clarification import parse_choice, selected_choice
from .context_reducer import cost, current_context_window
from .dvd_context import source_records

T = TypeVar("T", bound=BaseModel)


class EvidenceAudit(BaseModel):
    """List evidence defects before deciding acceptance, avoiding an early verdict."""

    model_config = ConfigDict(
        json_schema_extra={
            "required": [
                "unsupported_claims",
                "missing_requirements",
                "satisfied",
                "critique",
                "refined_search_query",
            ]
        }
    )
    # Defaults also accept older adapter/test payloads; the generation schema
    # requires these fields and places the evidence audit before the verdict.
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    satisfied: bool
    critique: str = ""
    refined_search_query: str | None = None


_LIMIT_MIN, _LIMIT_MAX = 1, 20
_CONTEXT_HEIGHT_MIN, _CONTEXT_HEIGHT_MAX = 0, 5
# IDU_DVD ``block`` filter accepts only these two values (see IDU_DVD SearchRequest).
_VALID_BLOCKS = {"main", "amendment"}


def _clean_str_list(
    values: list[str] | None, *, lower: bool = False
) -> list[str] | None:
    """Drop empties/non-strings from an LLM-produced list filter; ``None`` if nothing remains."""
    if not values:
        return None
    cleaned = [
        (v.strip().lower() if lower else v.strip())
        for v in values
        if isinstance(v, str) and v.strip()
    ]
    return cleaned or None


async def _request_json(
    llm_client,
    model: str,
    messages: list[dict],
    model_cls: Any,
    retries: int = 2,
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
) -> T:
    """
    Ask the LLM for a JSON object and parse it into ``model_cls``.

    Mirrors the structured-output convention used by ProvisionPlanBuilder: temperature 0,
    strip markdown fences, retry by feeding the invalid response back to the model.
    """
    adapter = TypeAdapter(model_cls)
    model_name = (
        "RetrievalPlan"
        if model_cls is RetrievalPlan
        else getattr(model_cls, "__name__", "structured response")
    )
    schema = adapter.json_schema()
    for attempt in range(retries + 1):
        # The schema is a decoding constraint, not another message. Reserving its
        # serialized UTF-8 size rejected the existing planner even with no history.
        available = (
            current_context_window() - sum(cost(m["content"]) for m in messages) - 256
        )
        if available < 128:
            raise ValueError("structured request exceeds configured context window")
        response = await llm_client.chat(
            model=model,
            think=False,
            format=schema,
            options={
                "temperature": 0,
                "num_predict": min(max_tokens, available),
                "num_ctx": current_context_window(),
            },
            messages=messages,
            **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
        )
        content = response["message"]["content"]
        logger.debug(f"LLM {model_name} response [{model}]: {content}")
        try:
            return adapter.validate_json(strip_json_fence(content))
        except (ValidationError, json.JSONDecodeError) as exc:
            validation_details = (
                json.dumps(
                    exc.errors(include_url=False), ensure_ascii=False, default=str
                )
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            if attempt < retries:
                logger.warning(
                    f"LLM returned invalid {model_name} JSON "
                    f"(retries left: {retries - attempt - 1}): {exc}"
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий JSON нарушает схему: "
                            f"{validation_details}. Исправь указанные поля и верни "
                            "только валидный JSON нужной структуры без markdown и пояснений."
                        ),
                    },
                ]
            else:
                raise ValueError(f"Model returned invalid {model_name} JSON") from exc
    raise AssertionError("unreachable")


class RetrievalPlanner:
    """Builds a :class:`RetrievalPlan` for a RAG round via structured LLM output."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def build_plan(
        self,
        model: str,
        user_query: str,
        history: list[dict] | None = None,
        prev_critique: str | None = None,
        prev_query: str | None = None,
    ) -> RetrievalPlan:
        choice = selected_choice(user_query, history or [])
        if choice:
            return validate_retrieval_plan(
                {"search_query": user_query, **parse_choice(choice)}
            )
        messages: list[dict] = [
            {"role": "system", "content": self._prompt(prev_critique, prev_query)},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        plan = await _request_json(self.llm_client, model, messages, RetrievalPlan)
        plan = self._clamp(plan, user_query)
        plan = apply_scope(plan, user_query, history=history)
        logger.info(f"DVD retrieval plan: {plan.model_dump_json(ensure_ascii=False)}")
        return plan

    @staticmethod
    def _clamp(plan: RetrievalPlan, user_query: str) -> RetrievalPlan:
        if choice := parse_choice(user_query):
            return validate_retrieval_plan({**plan.model_dump(), **choice})
        block = (plan.block or "").strip().lower() or None
        if block not in _VALID_BLOCKS:
            block = None
        # A concrete address is an identifier, not a semantic search phrase. The
        # explicit user token wins over an LLM paraphrase (and over later critiques).
        reference = parse_reference(user_query)
        designations = reference.document_names
        pattern = reference.pattern or (plan.pattern or "").strip() or None
        # Keep a supplied complete path when the literal query only names its leaf.
        if (
            reference.pattern
            and "/" not in reference.pattern
            and plan.pattern
            and "/" in plan.pattern
        ):
            if plan.pattern.rsplit("/", 1)[-1].strip() == reference.pattern:
                pattern = plan.pattern.strip()
        mode = (
            "structure"
            if pattern
            else "name" if plan.name_query else plan.retrieval_mode
        )
        updates = {
            "pattern": pattern,
            "retrieval_mode": mode,
            "types": (
                None if mode != "semantic" else _clean_str_list(plan.types, lower=True)
            ),
            "kind": SearchKind.ALL if mode != "semantic" else plan.kind,
        }
        if reference.pattern:
            updates.update(include_children=True, name_query=None, types=None)
            # A specific provision is already the requested text. Ranking is useful
            # for a topic inside a section, but never for truncating an exact quote.
            if wants_full_quote(user_query) or not reference.pattern.rsplit("/", 1)[
                -1
            ].strip().startswith(("раздел ", "глава ", "приложение ")):
                updates["rank_by_relevance"] = False
        if designations:
            updates["document_names"] = list(dict.fromkeys(designations))
            updates["doc_id"] = None
        if re.search(
            r"\b(?:мо[её]м|моего|моих|мой|загруженн[а-я]+\s+мной)\s+документ",
            user_query,
            re.I,
        ):
            updates["include_shared"] = False
        return validate_retrieval_plan(
            {
                **plan.model_dump(),
                "search_query": (plan.search_query or "").strip() or user_query,
                "limit": min(max(plan.limit, _LIMIT_MIN), _LIMIT_MAX),
                "context_height": min(
                    max(plan.context_height, _CONTEXT_HEIGHT_MIN), _CONTEXT_HEIGHT_MAX
                ),
                "document_names": _clean_str_list(plan.document_names),
                "block": block,
                "types": _clean_str_list(plan.types, lower=True),
                **updates,
            }
        )

    @staticmethod
    def _prompt(prev_critique: str | None, prev_query: str | None) -> str:
        structure = {
            "retrieval_mode": "semantic | structure | name",
            "pattern": 'null | "3.3" | "3.*" | "3.3–3.5" | "А / 2"',
            "name_query": "null | собственное наименование фрагмента",
            "name_mode": "strict | expanded",
            "name_scope": "self | path",
            "doc_id": "null | известный ID документа",
            "version": "null | явно запрошенная редакция",
            "include_children": True,
            "allow_multiple": False,
            "rank_by_relevance": False,
            "include_shared": True,
            "search_query": "строка для векторного поиска",
            "kind": "text | table | all",
            "limit": 10,
            "context_height": 1,
            "document_names": 'null | ["название документа", ...]',
            "block": "null | main | amendment",
            "types": 'null | ["clause", "table", ...]',
        }
        prompt = f"""Ты планируешь поиск по векторной базе нормативных документов \
(градостроительство и городское планирование).
По вопросу пользователя сформируй параметры поиска. \
Верни только валидный JSON без markdown и пояснений:
{json.dumps(structure, ensure_ascii=False)}

Правила:
1. Сначала заполни фильтры из вопроса и диалога. Новый номер пункта, «в нём», «в СП»
   продолжают выбранный документ; явно другой документ заменяет его. Не придумывай
   ID, редакцию или адрес. Сохраняй ограничения при повторных поисках.
2. Точный пункт/цитата/полный раздел: structure, pattern — адрес, rank_by_relevance=false,
   include_children=true. «3.3» точно, «3.*» потомки, «3.3–3.5» диапазон, «А / 2» путь.
   Номер не задаёт types: definition с номером тоже пункт. Для structure kind=all, types=null.
3. Тема внутри раздела: structure, pattern="раздел 5", rank_by_relevance=true,
   search_query — тема. Документ и тема без адреса: semantic с document_names/doc_id.
   Только тема: semantic по доступной базе. Точные тексты не заменяются похожими.
4. Наименование элемента: name, name_query — заголовок/термин/подпись.
   name_mode=strict (часть/маска), expanded (словоформы/опечатки/смысл).
   name_scope=self либо path для названия предка. pattern и name_query совместимы (AND).
   Название документа помещай в document_names, не name_query.
5. allow_multiple=true только для явного обзора/сравнения/маски/диапазона.
   Иначе неоднозначный документ или адрес требует выбора уникальной сущности.
6. «В моём документе», «загруженных мной документах»: include_shared=false (индекс
   текущего проекта). Иначе true. document_names, version, block, types по умолчанию
   null, но сохраняй выбранный документ из контекста. block=main для основной части,
   amendment для изменений. types задавай только по явно запрошенному виду элемента.
7. search_query — краткая тема поиска на русском. kind=text/table/all; не дублируй
   kind=table фильтром types. limit=1..20, context_height=0..5, для точечных вопросов 0..1.
Пример «что в пункте 3.3 СП 55»: structure, pattern="3.3", document_names=["СП 55"]."""
        if prev_critique:
            prompt += f"""

Предыдущая попытка ответа не прошла самопроверку.
Замечание критика: {prev_critique}
Предыдущий поисковый запрос: «{prev_query}».
Сформируй ИНОЙ, улучшенный поисковый запрос (синонимы, иные формулировки, \
официальная терминология); при необходимости измени kind, limit, context_height. \
Сохрани явно заданные документ, редакцию, структуру и наименование: ограничения \
нельзя снимать ради получения совпадений."""
        return prompt


class AnswerCritic:
    """Reviews a drafted answer against the retrieved context via structured LLM output."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def review(
        self,
        model: str,
        user_query: str,
        context: str,
        answer: str,
    ) -> CriticVerdict:
        if defects := self._literal_defects(context, answer):
            return CriticVerdict(satisfied=False, critique="; ".join(defects))
        messages: list[dict] = [
            {"role": "system", "content": self._prompt()},
            {"role": "user", "content": self._payload(user_query, context, answer)},
        ]
        try:
            audit = await _request_json(
                self.llm_client,
                model,
                messages,
                EvidenceAudit,
                max_tokens=int(os.getenv("DVD_REVIEW_MAX_TOKENS", "4096")),
                reasoning_effort=(
                    "medium"
                    if isinstance(self.llm_client, OpenAiCompatAdapter)
                    and "gpt-oss" in model.lower()
                    else None
                ),
            )
            defects = audit.unsupported_claims + audit.missing_requirements
            verdict = CriticVerdict(
                satisfied=audit.satisfied and not defects,
                critique=audit.critique or "; ".join(defects),
                refined_search_query=audit.refined_search_query,
            )
        except ValueError:
            logger.warning("Critic produced invalid JSON; draft remains unverified")
            return CriticVerdict(
                satisfied=False, critique="Не удалось проверить обоснованность ответа."
            )
        logger.info(
            f"DVD critic verdict: {verdict.model_dump_json(ensure_ascii=False)}"
        )
        return verdict

    @staticmethod
    def _literal_defects(context: str, answer: str) -> list[str]:
        """Verify explicit expansions and table identifiers against literal evidence.

        These narrow, checkable assertions were repeatedly approved incorrectly by
        the live critic. Other semantic claims still require the model's audit.
        """

        def normalize(text):
            return " ".join(re.sub(r"[‐‑–—]", "-", text).lower().split())

        source = normalize(context)
        defects = []
        # Application labels must point at retrieved sources. Do not reinterpret
        # bibliography markers inside the verbatim quotation as generated links.
        explanation = answer.split("Полная цитата:", 1)[0]
        generated = "\n".join(
            line
            for line in explanation.splitlines()
            if not line.lstrip().startswith(">")
        )
        known_labels = set(source_records(context)) - {"unlabelled"}
        for label in re.findall(r"\[(?:N|\d+)\]", generated):
            if label == "[N]" or (known_labels and label not in known_labels):
                defects.append(
                    f"Ссылка {label} отсутствует среди источников. Используй конкретные доступные метки, например [1], вместо шаблона [N]."
                )
        if "Полная цитата:" in answer and re.search(
            r"(?:\b(?:нет|отсутствует|не\s+содерж[а-яё]*)\s+(?:\w+\s+){0,2}(?:текст[а-я]*|содержани[а-я]*)\b|\b(?:сам\s+)?текст\b[^!?\n]{0,100}(?:не\s+(?:привед[её]н|предоставлен|представлен)|отсутствует))",
            explanation,
            re.I,
        ):
            defects.append(
                "Полный текст уже приведён в цитате. Не утверждай, что сам текст отсутствует; объясни имеющуюся формулировку без выдуманных требований."
            )
        for acronym, expansion in re.findall(
            r"\b([А-ЯЁA-Z]{2,})\s*\(([^()\n]+)\)", answer
        ):
            # Numeric editions/units and simple cross references are not expansions.
            if (
                len(re.findall(r"[а-яёa-z]{3,}", expansion)) >= 2
                and normalize(expansion) not in source
            ):
                defects.append(
                    f"Расшифровка {acronym} «{expansion}» не подтверждена исходным текстом. Убери её или приведи дословно из источника."
                )
        table_pattern = r"(?:\bтабл(?:иц[а-яё]*|\.)|\bт\s+а\s+б\s+л\s+и\s+ц\s+а)\s*([а-яa-z]?\.?\s*\d+(?:\.\d+)*)"
        known_tables = {m.replace(" ", "") for m in re.findall(table_pattern, source)}
        # Keep decimal identifiers intact while separating sentences/paragraphs.
        for sentence in re.split(r"\n|(?<=[.!?])\s+(?=[А-ЯЁA-Z])", answer):
            lowered = normalize(sentence)
            # An explicit statement that a referenced table is absent is not a citation.
            if re.search(
                r"не (?:найден\w*|приведен\w*|приведён\w*|представлен\w*)|отсутств\w*|нет (?:текста |данных .*?о )?таблиц",
                lowered,
            ):
                continue
            for table in re.findall(table_pattern, lowered):
                if table.replace(" ", "") not in known_tables:
                    defects.append(
                        f"Таблица {table} не подтверждена фрагментами. Не подменяй номер пункта номером таблицы; исправь ссылку или убери её."
                    )
        return defects

    @staticmethod
    def _prompt() -> str:
        structure = {
            "unsupported_claims": [
                "unsupported statements, including definitions and citation metadata; [] if none"
            ],
            "missing_requirements": [
                "directly relevant requirements omitted from the answer; [] if none"
            ],
            "satisfied": "true | false",
            "critique": "кратко: что не так с ответом (пусто, если всё хорошо)",
            "refined_search_query": "улучшенный поисковый запрос или null",
        }
        return f"""Audit a Russian answer against the supplied document EXCERPTS, not your prior knowledge.
Return JSON only: {json.dumps(structure, ensure_ascii=False)}
First inspect every assertion and list evidence defects; only then decide satisfied.
Do not approve a mostly correct answer that contains even one unsupported assertion.
For example, if a source only uses an acronym, an invented parenthetical expansion
in the answer is an unsupported claim even when its main conclusion is correct.
If the source says clause 27.3 and table 31.3, citing TABLE 27.3 is unsupported.

Distinguish application citation labels in excerpt HEADERS from bibliography
references inside source TEXT. For example, if excerpt [1] contains a reference
[6], an answer may say "позиция 6 библиографии документа" and cite excerpt [1].
This correctly attributes the cross-reference; do not require application label
[6] or confuse the bibliography position with a clause number. The referenced
external document's actual requirements are not available from that reference.

Hard rejection rules:
1. A rule for one building type MUST NOT be transferred to another type. A house,
   hotel, prison or prison school rule does NOT establish a rule for an ordinary
   school. Calling it a general principle, analogy or useful guideline is STILL
   an unsupported claim. References to other standards do not establish their text.
2. Every statement, including acronym expansions, definitions, source attribution,
   clause/table numbers, obligations, conditions and exceptions must be supported by
   the supplied text and retain its explicit scope. Do not fill gaps from memory.
3. A table of contents or section TITLE only proves that the topic is mentioned;
   it does NOT provide the requirements inside that section.
4. Reject invented applicability, invented facts, or omissions of directly relevant
   requirements actually present in the excerpts. Request a refined search.

Accept an honest statement that THESE EXCERPTS do not contain enough applicable
information when this is true. Lack of evidence is not a reason to force an answer.
An honest limited answer must not claim the entire document or corpus has no rules.
Do not reject it merely because a contents page mentions schools or because other
building types have placement requirements. If only special-scope rules are present,
accept a clearly scoped quotation or explanation that ordinary schools need other sources.

When rejecting, write a short Russian critique identifying the unsupported claim
or the specific omitted passage. When accepting, satisfied=true, critique="",
refined_search_query=null. Never reward an answer just because it sounds helpful."""

    @staticmethod
    def _payload(user_query: str, context: str, answer: str) -> str:
        ctx = context or "(релевантные фрагменты не найдены)"
        return (
            f"Вопрос пользователя:\n{user_query}\n\n"
            f"Доступные фрагменты:\n{ctx}\n\n"
            f"Ответ ассистента для проверки:\n{answer}"
        )
