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

from src.agents.services.restriction.restriction_catalog import strip_json_fence
from src.agents.services.scenario_data.scenario_data_aggregate import (
    bounded_public_observation_context,
)

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
    "карта",
    "карту",
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
        "missing_code": {
            "type": "string",
            "enum": [
                "none",
                "answer_incomplete",
                "counts_missing",
                "wrong_scope",
                "required_table_not_emitted",
                "required_layer_not_emitted",
                "other",
            ],
        },
        "details": {"type": "string"},
    },
    "required": ["sufficient", "missing_code", "details"],
}

_JUDGE_REASONS = {
    "answer_incomplete": "Ответ не раскрывает запрошенные данные.",
    "counts_missing": "В ответе отсутствуют рассчитанные количества.",
    "wrong_scope": "Ответ относится не к выбранному сценарию.",
    "required_table_not_emitted": "Требуемая таблица не была сформирована.",
    "required_layer_not_emitted": "Требуемый географический слой не был сформирован.",
    "other": "Ответ не прошёл проверку полноты.",
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


def _required_items(required_output: Any, key: str) -> list[Any]:
    if required_output is None:
        return []
    if isinstance(required_output, dict):
        value = required_output.get(key)
    else:
        value = getattr(required_output, key, None)
    return value if isinstance(value, list) else []


def _has_complete_table(observations: list[dict[str, Any]]) -> bool:
    return any(
        int(observation.get("table_count") or 0) > 0
        and observation.get("table_complete", True) is not False
        for observation in observations
    )


def required_output_checks(
    required_output: Any, observations: list[dict[str, Any]]
) -> list[str]:
    """Validate machine-observable artifacts without asking an LLM to guess."""

    if _required_items(required_output, "tables") and not _has_complete_table(
        observations
    ):
        if any(int(item.get("table_count") or 0) > 0 for item in observations):
            return ["Требуемая таблица была сформирована не полностью."]
        return ["Требуемая таблица не была сформирована."]
    if _required_items(required_output, "layers") and _layer_count(observations) == 0:
        return ["Требуемый географический слой не был сформирован."]
    return []


def deterministic_checks(
    user_query: str, observations: list[dict[str, Any]], answer: str
) -> list[str]:
    """Reasons to reject ``answer``, or an empty list when the rules see nothing wrong."""

    reasons: list[str] = []
    lowered = answer.lower()

    if not answer.strip():
        reasons.append("Ответ пустой.")
        return reasons

    if not any(
        item.get("retrieved")
        or item.get("aggregate") is not None
        or (item.get("tool") and "retrieved" not in item)
        or item.get("layer_count")
        or item.get("table_count")
        for item in observations
    ):
        return [
            "Данные не получены: отсутствие наблюдений не подтверждает отсутствие объектов."
        ]

    retrieved = [item for item in observations if item.get("retrieved")]
    if (
        retrieved
        and all(
            str(item.get("source_role", "")).startswith("справочник")
            for item in retrieved
        )
        and re.search(r"сколько|количеств", user_query, re.I)
        and not re.search(r"(?:сколько|количеств\w*)\s+тип", user_query, re.I)
    ):
        reasons.append(
            "Получен только справочник: для количества сущностей нужна их выборка."
        )

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
        *,
        required_output: Any = None,
    ) -> Verdict:
        reasons = [
            *required_output_checks(required_output, observations),
            *deterministic_checks(user_query, observations, answer),
        ]
        if reasons:
            # A rule already found something concrete; spending a judge call to confirm it
            # would only add latency.
            return Verdict(sufficient=False, hint=" ".join(reasons), reasons=reasons)

        judge = await self._judge(
            model,
            user_query,
            observations,
            answer,
            required_output=required_output,
        )
        if judge is None:
            reason = "Не удалось проверить достоверность ответа."
            return Verdict(sufficient=False, hint=reason, reasons=[reason])
        sufficient, missing_code = judge
        if sufficient:
            return Verdict(sufficient=True)
        if missing_code == "required_table_not_emitted" and (
            not _required_items(required_output, "tables")
            or _has_complete_table(observations)
        ):
            logger.warning(
                "Scenario data: judge claimed a missing table contrary to execution "
                "evidence; requiring a new content check"
            )
            return Verdict(
                sufficient=False,
                hint="Повторите проверку содержимого ответа: наличие таблицы не подтверждает его факты.",
                reasons=[
                    "Проверяющий отклонил ответ; наличие таблицы не отменяет отказ."
                ],
            )
        if missing_code == "required_layer_not_emitted" and (
            not _required_items(required_output, "layers")
            or _layer_count(observations) > 0
        ):
            logger.warning(
                "Scenario data: judge claimed a missing layer contrary to execution "
                "evidence; requiring a new content check"
            )
            return Verdict(
                sufficient=False,
                hint="Повторите проверку содержимого ответа: наличие слоя не подтверждает его факты.",
                reasons=["Проверяющий отклонил ответ; наличие слоя не отменяет отказ."],
            )
        reason = _JUDGE_REASONS.get(missing_code)
        if reason is None or missing_code == "none":
            reason = "Проверка ответа вернула некорректную причину отказа."
            return Verdict(sufficient=False, hint=reason, reasons=[reason])
        return Verdict(
            sufficient=False,
            hint=reason,
            reasons=[reason],
        )

    async def _judge(
        self,
        model: str,
        user_query: str,
        observations: list[dict[str, Any]],
        answer: str,
        *,
        required_output: Any = None,
    ) -> tuple[bool, str] | None:
        context = bounded_public_observation_context(observations, max_chars=12000)
        required = {
            "tables": _required_items(required_output, "tables"),
            "layers": _required_items(required_output, "layers"),
        }
        prompt = (
            "Ты проверяешь ответ агента по городским данным. Верни строгий JSON "
            '{"sufficient": bool, "missing_code": str, "details": str}.\n'
            "sufficient=false, если ответ не отвечает на заданный вопрос, игнорирует "
            "посчитанные количества из наблюдений, подменяет конкретику общими словами "
            "или обещает данные, которых не привёл.\n"
            "Отсутствие вызовов не означает отсутствие объектов. Ноль допустим только "
            "при успешной пустой выборке с нужным типом и областью. Не путай число "
            "записей справочника с числом сущностей. Сверяй каждое количество с "
            "aggregate соответствующего источника; source_role и source_scope "
            "определяют смысл выборки.\n"
            "Наличие обязательной таблицы или слоя проверяет backend. Не отклоняй ответ "
            "из-за формата представления и не требуй таблицу, если это не указано в "
            "required_output. Если table_count > 0 и table_complete=true, таблица уже "
            "показана пользователю. Не требуй перепечатывать её строки.\n"
            "missing_code выбери только из JSON Schema. details используется лишь в "
            "диагностических логах и не управляет выполнением.\n"
            f"required_output: {json.dumps(required, ensure_ascii=False)}\n\n"
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
                    "Scenario data: answer judge returned no boolean verdict"
                )
                return None
            missing_code = payload.get("missing_code")
            if not isinstance(missing_code, str):
                return None
            if missing_code not in {"none", *_JUDGE_REASONS}:
                return None
            return verdict, missing_code
        except Exception as exc:  # noqa: BLE001 - never fail the run over the judge
            logger.warning(f"Scenario data: answer judge failed: {exc}")
            return None
