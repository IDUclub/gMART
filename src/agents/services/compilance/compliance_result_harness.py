"""Conversation context for questions about a completed compliance run.

The compliance pipeline produces a rich machine-readable summary.  This module is
the single seam that turns that persisted trace into a bounded, trusted LLM context
for later turns in the same chat.  It deliberately does not execute checks itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.agents.services.compilance.compliance_sources import (
    source_reference,
    source_references,
)


@dataclass(frozen=True)
class PreparedComplianceFollowUp:
    """LLM input grounded in the latest stored compliance result."""

    messages: list[dict[str, str]]
    summary: dict[str, Any]


class ComplianceResultHarness:
    """Recover and explain the latest compliance result in a chat."""

    _EXPLICIT_RERUN = re.compile(
        r"\b(?:проверь|проверьте|перепроверь|перепроверьте|повтори|повторите|"
        r"запусти|запустите|выполни|выполните|пересчитай|пересчитайте)\b|"
        r"\b(?:заново|повторно|снова)\b",
        re.IGNORECASE,
    )
    _PERSISTED_EVENT_TYPE = "compliance_summary"
    _MAX_EVIDENCE_PER_RESULT = 20

    def prepare_follow_up(
        self,
        user_query: str,
        messages: list[dict[str, Any]],
        history: list[dict[str, str]],
    ) -> PreparedComplianceFollowUp | None:
        """Return grounded messages, or ``None`` when a new check was requested."""

        summary = self.latest_summary(messages)
        if summary is None or not user_query.strip():
            return None
        if self._EXPLICIT_RERUN.search(user_query):
            return None

        compact_summary = self._compact_summary(summary)
        summary_json = json.dumps(compact_summary, ensure_ascii=False, indent=2)
        system_prompt = f"""
Ты — агент, который отвечает на вопросы о последней уже выполненной проверке
строительных ограничений. Используй только данные из <compliance_result>.
Не запускай новую проверку, не придумывай причины и не меняй статусы.

Правила интерпретации:
- violated — нарушение действительно найдено на проверенных объектах;
- passed — нарушение не найдено; если warnings содержит no_applicable_objects,
  применимых объектов в полном слое сценария не было и норма пройдена по правилу
  пустого множества;
- partial — проверена только часть применимых объектов, поэтому обязательно укажи
  число unchecked_objects и не распространяй вывод на непроверенную часть;
- unverifiable — норму пытались проверить, но не хватило данных/требований либо
  произошла ошибка исполнения; это не нарушение и не успешное прохождение;
- unsupported — норма была пропущена, потому что для неё нет поддерживаемого
  исполняемого шаблона или план проверки не прошёл валидацию;
- not_applicable — явное условие применимости нормы доказуемо не выполняется;
- compliance_status=unknown никогда не называй успешным прохождением.

Если пользователь спрашивает, что «не прошло», сначала раздели фактические
нарушения и нормы, которые не удалось проверить. При перечислении указывай
текстовую отсылку из source_references (краткое обозначение документа и пункт,
если он указан) и текст нормы, когда он есть. Для каждой нарушенной нормы
обязательно приведи такую отсылку. Если название отсутствует, пиши
«Источник не указан»; пункт добавляй только при его наличии. UUID и restriction_id
в ответ не включай, даже если они встречаются в истории. Не придумывай ссылки,
названия документов или номера пунктов. Причины бери из
missing_requirements, warnings и resolved_requirements.reason. Техническую ошибку
объясняй простыми словами, но сохраняй её точный текст. Если ни одна норма не была
полностью проверена (passed_norms=0 и violated_norms=0), начни ответ с фразы
«Ни одна норма не была полностью проверена». В этом случае не пиши отдельно
«нарушения не обнаружены»: итог не доказывает соответствие сценария.
Отвечай по-русски, кратко и конкретно.

<compliance_result>
{summary_json}
</compliance_result>
""".strip()
        return PreparedComplianceFollowUp(
            messages=[
                {"role": "system", "content": system_prompt},
                *history[-6:],
                {"role": "user", "content": user_query},
            ],
            summary=summary,
        )

    @classmethod
    def latest_summary(cls, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Read the newest valid summary from current or legacy stored parts."""

        for message in reversed(messages):
            for part in reversed(message.get("parts") or []):
                kind = part.get("kind")
                payload = part.get("payload") or part.get("data") or {}
                if kind == "data":
                    if payload.get("event_type") != cls._PERSISTED_EVENT_TYPE:
                        continue
                    candidate = payload.get("content")
                elif kind == cls._PERSISTED_EVENT_TYPE:
                    candidate = payload
                else:
                    continue
                if cls._is_summary(candidate):
                    return candidate
        return None

    @staticmethod
    def _is_summary(candidate: Any) -> bool:
        return (
            isinstance(candidate, dict)
            and isinstance(candidate.get("results"), list)
            and isinstance(candidate.get("total_norms"), int)
        )

    @classmethod
    def _compact_summary(cls, summary: dict[str, Any]) -> dict[str, Any]:
        count_fields = {
            key: summary.get(key, 0)
            for key in (
                "request_id",
                "total_norms",
                "violated_norms",
                "passed_norms",
                "unverifiable_norms",
                "unsupported_norms",
                "not_applicable_norms",
                "partial_norms",
            )
        }
        count_fields["results"] = [
            cls._compact_result(result)
            for result in summary.get("results", [])
            if isinstance(result, dict)
        ]
        return count_fields

    @classmethod
    def _compact_result(cls, result: dict[str, Any]) -> dict[str, Any]:
        source = result.get("source") or {}
        compact = {
            key: result.get(key)
            for key in (
                "template",
                "template_version",
                "verification_status",
                "compliance_status",
                "coverage",
                "summary",
                "missing_requirements",
                "warnings",
            )
        }
        compact["source_references"] = source_references(source)
        compact["source"] = {
            key: source.get(key)
            for key in (
                "document_name",
                "clause_number",
                "extraction_text",
                "planner_status",
            )
            if source.get(key) is not None
        }
        requirements = result.get("effective_requirements") or {}
        compact["effective_requirements"] = {
            "layers": [
                {
                    key: item.get(key)
                    for key in ("role", "entity", "entity_type", "required")
                    if item.get(key) is not None
                }
                for item in requirements.get("layers") or []
                if isinstance(item, dict)
            ],
            "attributes": [
                {
                    key: item.get(key)
                    for key in ("role", "on", "required", "min_fill_rate")
                    if item.get(key) is not None
                }
                for item in requirements.get("attributes") or []
                if isinstance(item, dict)
            ],
        }
        compact["unresolved_requirements"] = [
            {
                key: item.get(key)
                for key in ("role", "requirement_type", "layer", "field", "reason")
                if item.get(key) is not None
            }
            for item in result.get("resolved_requirements") or []
            if isinstance(item, dict) and not item.get("resolved", False)
        ]
        compact["evidence"] = [
            {
                key: item.get(key)
                for key in (
                    "object_ref",
                    "generator_ref",
                    "zone_ref",
                    "operation",
                    "measured_value",
                    "unit",
                    "threshold",
                    "operator",
                    "violated",
                    "warnings",
                )
                if item.get(key) is not None
            }
            for item in (result.get("evidence") or [])[: cls._MAX_EVIDENCE_PER_RESULT]
            if isinstance(item, dict)
        ]
        return compact

    @staticmethod
    def fallback_answer(summary: dict[str, Any]) -> str:
        """Produce a truthful broad answer if the model returns no text."""

        lines = [
            f"Последняя проверка охватила {summary.get('total_norms', 0)} норм: "
            f"нарушено — {summary.get('violated_norms', 0)}, "
            f"пройдено на проверенной части — {summary.get('passed_norms', 0)}, "
            f"не удалось проверить — {summary.get('unverifiable_norms', 0)}, "
            f"не поддерживается — {summary.get('unsupported_norms', 0)}."
        ]
        for result in summary.get("results") or []:
            if not isinstance(result, dict):
                continue
            source = result.get("source") or {}
            name = "; ".join(source_references(source))
            if source.get("extraction_text"):
                name += f": {source['extraction_text']}"
            reasons = [
                *[str(item) for item in result.get("missing_requirements") or []],
                *[str(item) for item in result.get("warnings") or []],
            ]
            reason_text = "; ".join(dict.fromkeys(reasons)) or "причина не указана"
            lines.append(
                f"- {name}: {result.get('verification_status', 'unknown')} — "
                f"{reason_text}."
            )
        if not summary.get("passed_norms") and not summary.get("violated_norms"):
            lines.append(
                "Ни одна норма не была полностью проверена, поэтому этот итог не "
                "подтверждает соответствие сценария требованиям."
            )
        return "\n".join(lines)

    @staticmethod
    def normalize_answer(summary: dict[str, Any], answer: str) -> str:
        """Enforce status semantics that must not depend on model obedience."""

        normalized = answer.strip()
        # Old chat history may still contain IDs; never expose them as citations.
        references = {}
        for result in summary.get("results") or []:
            source = result.get("source") or {}
            if result.get("restriction_id"):
                references[result["restriction_id"]] = source_reference(source)
            for equivalent in source.get("equivalent_sources") or []:
                if equivalent.get("restriction_id"):
                    references[equivalent["restriction_id"]] = source_reference(
                        equivalent
                    )
        if references:
            pattern = (
                r"(?<![\w-])(?:"
                + "|".join(
                    re.escape(key) for key in sorted(references, key=len, reverse=True)
                )
                + r")(?![\w-])"
            )
            normalized = re.sub(pattern, lambda match: references[match[0]], normalized)
        if summary.get("passed_norms") or summary.get("violated_norms"):
            return normalized
        normalized = re.sub(
            r"^\s*(?:\*\*)?нарушения не обнаружены[.!]?(?:\*\*)?\s*",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        caveat = (
            "Ни одна норма не была полностью проверена, поэтому результат не "
            "подтверждает соответствие сценария требованиям."
        )
        return f"{caveat}\n\n{normalized}" if normalized else caveat
