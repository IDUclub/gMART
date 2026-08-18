"""Judge a scenario-data draft answer, and say what a second attempt should fix.

The agent can finish a run having called the wrong tool, or having ignored numbers that were
right there in the observations, and still produce fluent prose — "types are not specified,
so the exact distribution cannot be reported" while the breakdown sat in the context. Nothing
downstream noticed, because the pipeline had no notion of a *bad but well-formed* answer.

Two layers, cheapest first:

* deterministic checks — no LLM, no tokens, and they cannot themselves be fooled by fluent
  prose: an answer that pleads ignorance while aggregates exist, an answer with no numbers
  when counts were computed, a request for map layers that produced none;
* an LLM judge for everything a rule cannot see — whether the answer actually addresses the
  question that was asked.

A verdict carries a ``hint``: the planner receives it on the retry, so the second pass is
steered rather than merely repeated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.agents.services.restriction_catalog import strip_json_fence
from src.agents.services.scenario_data_aggregate import bounded_observation_context

#: Extra tool-loop passes allowed after a rejected answer.
MAX_ANSWER_ATTEMPTS = 2

_IGNORANCE_MARKERS = (
    "неизвестн",
    "не указан",
    "не удалось определить",
    "нет данных о типах",
    "невозможно сообщить",
    "не могу сообщить",
    "unknown",
    "not specified",
)

_LAYER_REQUEST_MARKERS = (
    "слой",
    "слои",
    "слоя",
    "слоёв",
    "слоев",
    "на карте",
    "на карту",
    "geojson",
    "featurecollection",
    "границ",
    "геометри",
)

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "missing": {"type": "string"},
    },
    "required": ["sufficient", "missing"],
}


@dataclass
class Verdict:
    """Outcome of one evaluation."""

    sufficient: bool
    #: What a retry must do differently; empty when the answer was accepted.
    hint: str = ""
    #: Which checks rejected it, for logs and the SSE status text.
    reasons: list[str] = field(default_factory=list)


def wants_layers(user_query: str) -> bool:
    """True when the question asks for something to be drawn on the map."""

    lowered = user_query.lower()
    return any(marker in lowered for marker in _LAYER_REQUEST_MARKERS)


def _has_aggregates(observations: list[dict[str, Any]]) -> bool:
    return any(observation.get("aggregate") for observation in observations)


def _layer_count(observations: list[dict[str, Any]]) -> int:
    return sum(int(observation.get("layer_count") or 0) for observation in observations)


def deterministic_checks(
    user_query: str, observations: list[dict[str, Any]], answer: str
) -> list[str]:
    """Reasons to reject ``answer``, or an empty list when the rules see nothing wrong."""

    reasons: list[str] = []
    lowered = answer.lower()

    if not answer.strip():
        reasons.append("Ответ пустой.")
        return reasons

    if _has_aggregates(observations):
        if any(marker in lowered for marker in _IGNORANCE_MARKERS):
            reasons.append(
                "В ответе сказано, что данных о типах нет, хотя точная разбивка по "
                "полям уже посчитана и лежит в наблюдениях (поле aggregate)."
            )
        if not re.search(r"\d", answer):
            reasons.append(
                "В наблюдениях есть посчитанные количества, но в ответе нет ни одного "
                "числа — распределение не приведено."
            )

    pending_set: set[str] = set()
    for index, observation in enumerate(observations):
        for reference in observation.get("unresolved_references") or []:
            if not _reference_resolved(reference, observations[index + 1 :]):
                pending_set.add(reference)
    pending = sorted(pending_set)
    if pending:
        reasons.append(
            "Записи ссылаются на справочник полями "
            f"{', '.join(pending)}, но их названия так и не получены — ответ по номерам "
            "вместо названий не отвечает на вопрос. Нужен вызов справочника и "
            "сопоставление идентификаторов с названиями."
        )

    if wants_layers(user_query) and _layer_count(observations) == 0:
        reasons.append(
            "Пользователь просил показать объекты на карте, но ни один слой "
            "(FeatureCollection) не был получен: нужен инструмент, возвращающий "
            "геометрию — с GeoJSON или WithGeometry в названии."
        )

    return reasons


def _reference_resolved(reference: str, later: list[dict[str, Any]]) -> bool:
    """Recognize a later dictionary result instead of keeping stale pending state."""

    tail = reference.rsplit(".", 1)[-1]
    stem = tail.removesuffix("_ids").removesuffix("_id").lower()
    stem_tokens = {token for token in stem.split("_") if len(token) > 2}
    for observation in later:
        mapping = observation.get("mapping") or {}
        domain = str(mapping.get("domain") or "").lower()
        if stem_tokens and any(token in domain for token in stem_tokens):
            return True
        aggregate = observation.get("aggregate") or {}
        fields = set((aggregate.get("breakdown") or {}).keys())
        has_id = any(
            stem in field.lower() and field.lower().endswith("id") for field in fields
        )
        has_name = any(
            stem in field.lower() and "name" in field.lower() for field in fields
        )
        if has_id and has_name:
            return True
    return False


class ScenarioDataEvaluator:
    """Deterministic checks plus an LLM judge over the draft answer."""

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    async def evaluate(
        self,
        model: str,
        user_query: str,
        observations: list[dict[str, Any]],
        answer: str,
    ) -> Verdict:
        reasons = deterministic_checks(user_query, observations, answer)
        if reasons:
            # A rule already found something concrete; spending a judge call to confirm it
            # would only add latency.
            return Verdict(sufficient=False, hint=" ".join(reasons), reasons=reasons)

        judge = await self._judge(model, user_query, observations, answer)
        if judge is None:
            # The judge is an improvement, not a gate: if it fails, keep the answer.
            return Verdict(sufficient=True)
        sufficient, missing = judge
        if sufficient:
            return Verdict(sufficient=True)
        return Verdict(
            sufficient=False,
            hint=missing,
            reasons=[missing or "Ответ не отвечает на вопрос по существу."],
        )

    async def _judge(
        self,
        model: str,
        user_query: str,
        observations: list[dict[str, Any]],
        answer: str,
    ) -> tuple[bool, str] | None:
        context = bounded_observation_context(observations, max_chars=12000)
        prompt = (
            "Ты проверяешь ответ агента по городским данным. Верни строгий JSON "
            '{"sufficient": bool, "missing": str}.\n'
            "sufficient=false, если ответ не отвечает на заданный вопрос, игнорирует "
            "посчитанные количества из наблюдений, подменяет конкретику общими словами "
            "или обещает данные, которых не привёл.\n"
            "sufficient=true, если наблюдения действительно пусты и ответ честно об этом "
            "говорит.\n"
            "Если наблюдение содержит table_count > 0, полная таблица уже показана "
            "пользователю отдельной частью ответа. Не требуй перепечатывать её строки "
            "в тексте: точного количества и ссылки на полную таблицу достаточно. При "
            "этом проверь, что таблица действительно отфильтрована по запросу.\n"
            "В missing — что именно должен сделать следующий проход: какой инструмент "
            "вызвать или какие числа привести. Пиши по-русски, одной-двумя фразами.\n\n"
            f"Наблюдения:\n{context}"
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Вопрос пользователя:\n{user_query}\n\nОтвет агента:\n{answer}",
            },
        ]
        try:
            response = await self.llm_client.chat(
                model=model,
                messages=messages,
                think=False,
                format=_JUDGE_SCHEMA,
                options={"temperature": 0, "num_predict": 400},
            )
            payload = json.loads(strip_json_fence(response["message"]["content"]))
            verdict = payload.get("sufficient")
            if not isinstance(verdict, bool):
                # No usable opinion. Treating that as a rejection would let a malformed judge
                # reply burn the retry budget and tack a shortfall note onto a fine answer.
                logger.warning(
                    "Scenario data: answer judge returned no boolean verdict; accepting"
                )
                return None
            return verdict, str(payload.get("missing") or "")
        except Exception as exc:  # noqa: BLE001 - never fail the run over the judge
            logger.warning(f"Scenario data: answer judge failed: {exc}")
            return None
