from __future__ import annotations

import json
import os
import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.model_clients.openai_adapter import OpenAiCompatAdapter
from src.agents.services.restriction.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.dvd_plan import (
    CriticVerdict,
    RetrievalPlan,
    SearchKind,
)

from .context_reducer import cost, current_context_window

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
    model_cls: type[T],
    retries: int = 2,
    max_tokens: int = 1024,
    reasoning_effort: str | None = None,
) -> T:
    """
    Ask the LLM for a JSON object and parse it into ``model_cls``.

    Mirrors the structured-output convention used by ProvisionPlanBuilder: temperature 0,
    strip markdown fences, retry by feeding the invalid response back to the model.
    """
    for attempt in range(retries + 1):
        schema = model_cls.model_json_schema()
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
        logger.debug(f"LLM {model_cls.__name__} response [{model}]: {content}")
        try:
            return model_cls.model_validate_json(strip_json_fence(content))
        except (ValidationError, json.JSONDecodeError) as exc:
            if attempt < retries:
                logger.warning(
                    f"LLM returned invalid {model_cls.__name__} JSON "
                    f"(retries left: {retries - attempt - 1}): {exc}"
                )
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Твой предыдущий ответ содержит невалидный JSON. "
                            "Верни только валидный JSON нужной структуры без markdown и пояснений."
                        ),
                    },
                ]
            else:
                raise ValueError(
                    f"Model returned invalid {model_cls.__name__} JSON"
                ) from exc
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
        messages: list[dict] = [
            {"role": "system", "content": self._prompt(prev_critique, prev_query)},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        plan = await _request_json(self.llm_client, model, messages, RetrievalPlan)
        plan = self._clamp(plan, user_query)
        logger.info(f"DVD retrieval plan: {plan.model_dump_json(ensure_ascii=False)}")
        return plan

    @staticmethod
    def _clamp(plan: RetrievalPlan, user_query: str) -> RetrievalPlan:
        block = (plan.block or "").strip().lower() or None
        if block not in _VALID_BLOCKS:
            block = None
        # A concrete address is an identifier, not a semantic search phrase. The
        # explicit user token wins over an LLM paraphrase (and over later critiques).
        address = re.search(
            r"(?:пункт[а-я]*|подпункт[а-я]*|п\.|раздел[а-я]*|section|clause)\s*"
            r"([А-ЯA-Zа-яa-z]?\d+(?:\.[\d*?]+)*(?:\s*[–—-]\s*\d+(?:\.\d+)*)?)",
            user_query,
            re.I,
        )
        designations = [
            m[1]
            for m in re.finditer(
                r"\b([A-ZА-ЯЁ]{2,10}\s*\d+(?:[.\-]\d+){1,5})", user_query, re.I
            )
            if not (address and m.start() < address.end() and m.end() > address.start())
        ]
        pattern = address[1] if address else (plan.pattern or "").strip() or None
        if address and plan.pattern and "/" in plan.pattern:
            if plan.pattern.rsplit("/", 1)[-1].strip() == address[1].strip():
                pattern = plan.pattern.strip()
        mode = (
            "structure"
            if pattern
            else "name" if plan.name_query else plan.retrieval_mode
        )
        if mode == "structure" and not pattern:
            raise ValueError("structural retrieval requires pattern")
        if mode == "name" and not plan.name_query:
            raise ValueError("name retrieval requires name_query")
        updates = {
            "pattern": pattern,
            "retrieval_mode": mode,
            "types": (
                None if mode != "semantic" else _clean_str_list(plan.types, lower=True)
            ),
            "kind": SearchKind.ALL if mode != "semantic" else plan.kind,
        }
        if designations:
            updates["document_names"] = list(dict.fromkeys(designations))
        return plan.model_copy(
            update={
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
- Выбери retrieval_mode="structure", если пользователь указал структурную ссылку.
  Сохрани её в pattern: "3.3" точно, "3.*" все уровни ниже 3, "3.3–3.5" диапазон
  соседних элементов, "А / 2" элемент 2 внутри А. Тип документа не важен.
  Никогда не преобразуй номер пункта в семантический запрос и не ставь types по
  слову «пункт»: номер может принадлежать definition, section, table и любому типу.
- retrieval_mode="name" для поиска по наименованию фрагмента. name_query — заголовок,
  определяемый термин или подпись, а document_names — название исходного документа.
  name_mode="strict" ищет совпадение/часть/маску; expanded дополнительно словоформы,
  опечатки и смысл названия. name_scope="path" для «в разделе с названием ...»;
  self для собственного названия. pattern и name_query можно совмещать (AND).
- include_children=true: получаем также дочерние пункты. allow_multiple=true только
  для явно множественного/обзорного/сравнительного вопроса или маски/диапазона.
  Иначе несколько кандидатов требуют уточнения документа, редакции или пути.
- Пример «что в пункте 3.3 СП 2.13130.2020»: structure, pattern="3.3",
  document_names=["СП 2.13130.2020"], types=null, include_children=true.
  Пример «покажи определения огнезащитного покрытия»: name,
  name_query="огнезащитное покрытие", name_mode="expanded", name_scope="self".
- Для обычного смыслового вопроса без адреса/наименования используй semantic.
- Никогда не снимай явно названные документ, редакцию, структуру или наименование
  ради получения непустого ответа. Идентификаторы не придумывай.
- search_query — краткий поисковый запрос на русском, отражающий суть вопроса \
(ключевые термины, нормативная лексика). Не копируй вопрос дословно — выдели суть.
- kind = "table" если вопрос про числовые нормативы, показатели или таблицы; \
"text" для текстовых формулировок, определений и требований; "all" если неясно.
- limit — сколько фрагментов извлечь (целое 1–20). Больше для широких/обзорных \
вопросов, меньше для точечных.
- context_height — сколько соседних фрагментов прикреплять к каждому найденному \
(целое 0–5). Больше (2–3), когда важен контекст вокруг (определения, процедуры, \
перечни, ссылки на смежные пункты); 0–1 для точечных фактов.
- document_names — null по умолчанию (искать по всей базе). Заполняй списком названий \
документов ТОЛЬКО если пользователь явно назвал конкретный документ (например \
«СП 42.13330», «по ГОСТ 21.501»).
- block — null по умолчанию (искать везде). "amendment" — если вопрос про изменения/\
поправки к документу; "main" — если явно про основную (действующую) редакцию без учёта \
поправок.
- types — null по умолчанию (все уровни). Список структурных уровней для сужения: \
"table" (таблицы), "clause"/"subclause" (пункты/подпункты), "chapter"/"section" \
(главы/разделы), "definition" (определения/термины), "appendix" (приложения), \
"note" (примечания). Заполняй, только когда вопрос явно нацелен на определённый вид \
элемента («дай определение…» → ["definition"], «что в таблице…» → ["table"]). \
Не дублируй kind: при kind="table" не указывай types=["table"].

Все фильтры (document_names, block, types) по умолчанию null — не сужай поиск без явной \
необходимости, лишние фильтры отсекают релевантные фрагменты."""
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
