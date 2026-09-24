from __future__ import annotations

import json
import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel, ValidationError

from src.agents.services.restriction.restriction_catalog import strip_json_fence
from src.agents.services.service_entities.normgraph_plan import (
    NormGraphCriticVerdict,
    NormGraphPlan,
    PrimaryTool,
)

T = TypeVar("T", bound=BaseModel)

_LIMIT_MIN, _LIMIT_MAX = 1, 20
_NEIGHBORS_DEPTH_MIN, _NEIGHBORS_DEPTH_MAX = 0, 2

# Seed kinds of the NormGraph vocabulary that describe where a facility may stand.
PLACEMENT_KINDS = (
    "минимальное_расстояние",
    "требование_размещения",
    "запрет_размещения",
)
_PLACEMENT_QUESTION = re.compile(
    r"размещ|располож|расстояни|удален|отступ|разрыв|рядом|вблизи|вокруг|возле|около|"
    r"доступност|санитарно-защитн|\bсзз\b",
    re.I,
)


def is_placement_question(query: str) -> bool:
    """Where a facility may stand, not what it must contain."""
    return bool(_PLACEMENT_QUESTION.search(query or ""))


# A quantity is a number followed by a unit; clause numbers and [N] labels are not.
_QUANTITY = re.compile(
    r"(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+(?:[.,]\d+)?)\s*"
    r"(?:км|километр\w*|метр\w*|мест\w*|м²|м2|м|га|%|эт\w*|чел\w*)(?![а-яё])",
    re.I,
)
_NUMBER = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+(?:[.,]\d+)?")


def _normalize_number(raw: str) -> str:
    value = float(re.sub(r"[ \u00a0\u202f]", "", raw).replace(",", "."))
    return f"{value:g}"


def ungrounded_quantities(answer: str, context: str) -> list[str]:
    """Quantities in the answer whose number does not occur anywhere in the context."""
    known = {_normalize_number(raw) for raw in _NUMBER.findall(context or "")}
    missing: list[str] = []
    for match in _QUANTITY.finditer(answer or ""):
        if _normalize_number(match.group(1)) not in known:
            quantity = " ".join(match.group(0).split())
            if quantity not in missing:
                missing.append(quantity)
    return missing


def _clean_str_list(values: list[str] | None) -> list[str] | None:
    """Drop empties/non-strings from an LLM-produced list filter; ``None`` if nothing remains."""
    if not values:
        return None
    cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
    return cleaned or None


async def _request_json(
    llm_client,
    model: str,
    messages: list[dict],
    model_cls: type[T],
    retries: int = 2,
) -> T:
    """
    Ask the LLM for a JSON object and parse it into ``model_cls``.

    Mirrors the structured-output convention used by RetrievalPlanner/AnswerCritic (DVD RAG
    agent): temperature 0, strip markdown fences, retry by feeding the invalid response back
    to the model.
    """
    for attempt in range(retries + 1):
        response = await llm_client.chat(
            model=model,
            messages=messages,
            # Reasoning models such as gpt-oss may spend the entire prediction
            # budget in ``message.thinking`` and leave ``message.content`` empty.
            # Planning/critique need deterministic machine-readable output, not
            # a reasoning trace, so explicitly disable thinking and let Ollama
            # constrain the response with the Pydantic JSON schema.
            think=False,
            format=model_cls.model_json_schema(),
            options={"temperature": 0, "num_predict": 1024},
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


class NormGraphRetrievalPlanner:
    """Builds a :class:`NormGraphPlan` for a QA round via structured LLM output."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def build_plan(
        self,
        model: str,
        user_query: str,
        history: list[dict] | None = None,
        prev_critique: str | None = None,
        prev_query: str | None = None,
        prev_object: str | None = None,
    ) -> NormGraphPlan:
        messages: list[dict] = [
            {
                "role": "system",
                "content": self._prompt(prev_critique, prev_query, prev_object),
            },
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        plan = await _request_json(self.llm_client, model, messages, NormGraphPlan)
        plan = self._clamp(plan, user_query)
        logger.info(f"NormGraph plan: {plan.model_dump_json(ensure_ascii=False)}")
        return plan

    @staticmethod
    def _clamp(plan: NormGraphPlan, user_query: str) -> NormGraphPlan:
        updates: dict = {
            "search_query": (plan.search_query or "").strip() or user_query,
            "object": (plan.object or "").strip() or None,
            "subject": (plan.subject or "").strip() or None,
            "kind": (plan.kind or "").strip() or None,
            # Kind lists are chosen by code from the question, not by the model.
            "kinds": None,
            "document_names": _clean_str_list(plan.document_names),
            "tags": _clean_str_list(plan.tags),
            "limit": min(max(plan.limit, _LIMIT_MIN), _LIMIT_MAX),
            "neighbors_depth": min(
                max(plan.neighbors_depth, _NEIGHBORS_DEPTH_MIN), _NEIGHBORS_DEPTH_MAX
            ),
        }
        # ``applicable`` requires an object; fall back to search when the model omitted it.
        if plan.primary_tool == PrimaryTool.APPLICABLE and not updates["object"]:
            updates["primary_tool"] = PrimaryTool.SEARCH
        return plan.model_copy(update=updates)

    @staticmethod
    def _prompt(
        prev_critique: str | None, prev_query: str | None, prev_object: str | None
    ) -> str:
        structure = {
            "primary_tool": "search | applicable",
            "search_query": "строка для поиска по search_restrictions",
            "object": 'null | "объект, на который накладываются ограничения"',
            "subject": "null | строка",
            "kind": "null | строка (вид ограничения из контролируемого словаря)",
            "document_names": 'null | ["название документа", ...]',
            "tags": 'null | ["тег", ...]',
            "limit": 10,
            "neighbors_depth": 0,
            "check_conflicts": "true | false",
        }
        prompt = f"""Ты планируешь запрос к графу нормативных ограничений NormGraph \
(градостроительство: СП/СНиП/ГОСТ/СанПиН). По вопросу пользователя сформируй параметры запроса. \
Верни только валидный JSON без markdown и пояснений:
{json.dumps(structure, ensure_ascii=False)}

Правила:
- primary_tool = "applicable", если вопрос имеет вид «какие ограничения действуют на <объект>», \
«что нельзя размещать рядом с <объект>», т.е. про объект, к которому применяются ограничения. \
В этом случае обязательно заполни "object".
- primary_tool = "search" для всех остальных (открытых, текстовых, определительных) вопросов.
- search_query — краткий поисковый запрос на русском, отражающий суть вопроса (ключевые термины, \
нормативная лексика), даже в режиме "applicable" (используется как вторичный сигнал).
- object — сущность, к которой применяется ограничение (например «объекты пищевой промышленности»); \
null, если не применимо.
- subject — сущность, которая накладывает ограничение (например «санитарно-защитная зона»); \
null, если не сужает запрос.
- kind — конкретный вид ограничения, ТОЛЬКО если пользователь явно его называет; иначе null.
- document_names — null по умолчанию (искать по всей базе); заполняй, только если пользователь \
явно назвал документ («по СП 42.13330», «согласно СанПиН…»).
- tags — null, если не сужает запрос.
- Если вопрос о размещении объекта (где его можно или нельзя размещать, на каком расстоянии, \
что допустимо рядом или вокруг), пиши search_query лексикой размещения: «расстояние от <объекта> \
до …», «пешеходная доступность <объекта>», «запрет размещения <объекта> в зонах», \
«санитарно-защитная зона». Не пиши общее «ограничения размещения <объекта>»: оно находит \
требования к помещениям и оборудованию вместо норм размещения.
- limit — сколько ограничений извлечь (целое 1–20). Больше для широких/обзорных вопросов.
- neighbors_depth — 0 по умолчанию; 1–2, если вопрос требует понять связанные/смежные ограничения \
(например «а что ещё с этим связано», «какие ещё нормы это затрагивает»).
- check_conflicts = true, если пользователь явно спрашивает про противоречия/конфликты между \
нормами, или если вопрос сравнивает несколько источников/редакций и расхождение вероятно; \
иначе false."""
        if prev_critique:
            prompt += f"""

Предыдущая попытка ответа не прошла самопроверку.
Замечание критика: {prev_critique}
Предыдущий поисковый запрос: «{prev_query}». Предыдущий object-фильтр: «{prev_object}».
Сформируй ИНОЙ, улучшенный запрос (синонимы, иные формулировки, официальная терминология); \
при необходимости смени primary_tool, ослабь фильтры или включи check_conflicts, если противоречие \
могло остаться незамеченным."""
        return prompt


class NormGraphAnswerCritic:
    """Reviews a drafted answer against the retrieved context via structured LLM output."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def review(
        self,
        model: str,
        user_query: str,
        context: str,
        answer: str,
    ) -> NormGraphCriticVerdict:
        if missing := ungrounded_quantities(answer, context):
            # A number the context does not contain is invented; no model is needed
            # to see that, and a model reviewer has repeatedly let such values pass.
            verdict = NormGraphCriticVerdict(
                satisfied=False,
                critique=(
                    "В ответе есть значения, которых нет в приведённых ограничениях: "
                    + ", ".join(missing)
                    + ". Приводи только значения из контекста."
                ),
            )
            logger.info(
                f"NormGraph critic verdict: {verdict.model_dump_json(ensure_ascii=False)}"
            )
            return verdict
        messages: list[dict] = [
            {"role": "system", "content": self._prompt()},
            {"role": "user", "content": self._payload(user_query, context, answer)},
        ]
        try:
            verdict = await _request_json(
                self.llm_client, model, messages, NormGraphCriticVerdict
            )
        except ValueError:
            logger.warning("Critic produced invalid JSON; draft remains unverified")
            return NormGraphCriticVerdict(
                satisfied=False, critique="Не удалось проверить обоснованность ответа."
            )
        logger.info(
            f"NormGraph critic verdict: {verdict.model_dump_json(ensure_ascii=False)}"
        )
        return verdict

    @staticmethod
    def _prompt() -> str:
        structure = {
            "satisfied": "true | false",
            "critique": "кратко: что не так с ответом (пусто, если всё хорошо)",
            "refined_search_query": "улучшенный поисковый запрос или null",
            "refined_object": "улучшенный object-фильтр или null",
        }
        return f"""Ты — строгий рецензент ответов ассистента по нормативным ограничениям \
(градостроительство). Оцени, полностью ли ответ обоснован приведёнными ограничениями и отвечает \
ли на вопрос. Верни только валидный JSON без markdown:
{json.dumps(structure, ensure_ascii=False)}

Критерии отказа (satisfied = false):
- В ответе есть утверждения, не подтверждённые приведёнными ограничениями (галлюцинации).
- В ответе есть толкования, выводы или расшифровки сокращений, которых нет в контексте \
(например, сокращение объяснено по догадке).
- Ответ неполный или не отвечает на вопрос пользователя.
- В ответ включены ограничения не по предмету вопроса: например, спрашивают о размещении \
объекта (расстояния, зоны, доступность), а в ответе требования к его помещениям, оборудованию \
или внутренней территории; или спрашивают о школах, а приведены нормы для других учреждений.
- В кавычках «» приведён текст, которого нет дословно в строках «Текст пункта» контекста \
(строка «Структура» — служебная, её нельзя выдавать за цитату документа).
- Не указаны источники (документ, редакция, номер пункта или restriction_id для каждого \
приведённого ограничения), хотя они есть в контексте.
- В контексте приведены противоречащие друг другу ограничения (раздел «Обнаруженные \
противоречия»), но ответ их не упоминает.
- В контексте недостаточно данных — тогда обязательно предложи refined_search_query и/или \
refined_object для нового запроса.

Если ответ корректен, полон, обоснован и содержит ссылки на источники — satisfied = true, \
critique пустой, refined_search_query и refined_object = null. Будь требователен по сути, но не \
придирайся к стилю."""

    @staticmethod
    def _payload(user_query: str, context: str, answer: str) -> str:
        ctx = context or "(релевантные ограничения не найдены)"
        return (
            f"Вопрос пользователя:\n{user_query}\n\n"
            f"Доступный контекст:\n{ctx}\n\n"
            f"Ответ ассистента для проверки:\n{answer}"
        )
