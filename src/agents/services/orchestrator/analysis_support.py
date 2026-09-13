"""User-facing recovery requests, scoped context and analysis configuration."""

import hashlib
import os
import re
from dataclasses import replace

from src.agents.runtime.budget import BudgetLimits
from src.agents.services.service_entities.orchestrator_plan import NeededInput
from src.common.service_auth import user_id_from_jwt


def context_scope(token, chat_id):
    if not chat_id:
        return None
    token = token or ""
    # JWT extraction supplies a cache namespace, never permission to access data;
    # actual data access is still enforced by the service transports.
    try:
        identity = user_id_from_jwt(token)
    except Exception:
        identity = hashlib.sha256(token.encode()).hexdigest()
    return hashlib.sha256(f"{identity}:{chat_id}".encode()).hexdigest()


def configured_limits(tokens=None, seconds=None):
    defaults = BudgetLimits()
    changes = {}
    for key in (
        "total_tokens",
        "model_calls",
        "tool_calls",
        "seconds",
        "steps",
        "context_tokens",
        "output_tokens",
    ):
        value = int(
            os.getenv("ORCHESTRATOR_" + key.upper(), str(getattr(defaults, key)))
        )
        if value <= 0:
            raise ValueError(f"ORCHESTRATOR_{key.upper()} must be positive")
        changes[key] = value
    limits = replace(defaults, **changes)
    return replace(
        limits,
        total_tokens=min(tokens or limits.total_tokens, limits.total_tokens),
        seconds=min(seconds or limits.seconds, limits.seconds),
    )


def missing_input(reason, detail=""):
    if "missing_service_normative" in detail:
        action = re.search(r"['\"]required_action['\"]:\s*['\"]([^'\"]+)", detail)
        return NeededInput(
            missing="Применимый норматив обеспеченности",
            reason="Расчётный сервис подтвердил, что в Urban API не задан необходимый норматив для выбранного вида услуг и территории.",
            question=(
                action.group(1)
                if action
                else "Для продолжения нужен применимый норматив: радиус или время доступности и норма обеспеченности. Укажите источник норматива либо территорию, для которой он уже задан."
            ),
            example="Название документа, редакция, пункт и значения норматива для выбранной территории и вида услуг.",
            owner="service",
        )
    if reason == "planning":
        return NeededInput(
            missing="Корректное управляющее решение для анализа",
            reason="Модель не вернула корректное управляющее решение после ограниченных попыток.",
            question="Продолжите сохранённый анализ, чтобы повторить управляющий вызов. Если ошибка повторится, оператору сервиса нужен идентификатор запроса; исходные данные пользователя не считаются отсутствующими.",
            example="Продолжи сохранённый анализ.",
            owner="service",
        )
    if reason in {"time", "tokens", "tool_calls", "model_calls", "steps", "context"}:
        return NeededInput(
            missing=(
                "Доступный бюджет исследования"
                if reason != "context"
                else "Более узкая область анализа"
            ),
            reason="Текущий предел анализа достигнут; подтверждённые результаты сохранены.",
            question="Укажите, какие сценарии или показатели проверить в первую очередь, либо попросите продолжить сохранённый анализ.",
            example="Продолжи сравнение только по обеспеченности школами.",
            owner="budget",
        )
    if reason == "empty_norms":
        return NeededInput(
            missing="Применимые нормативы",
            reason="Источник не вернул норм, поэтому соответствие проверить нельзя.",
            question="Сообщите название документа, редакцию и нужный пункт, если они известны. Для автоматической проверки потребуется доступный нормативный корпус сервиса.",
            example="Проверить пункт указанного СП, редакция и область применения…",
            owner="service",
        )
    if reason == "clarification":
        return NeededInput(
            missing="Уточнение условий задачи",
            reason="Без уточнения следующий расчёт может относиться к неверным данным.",
            question=detail
            or "Укажите сценарии, показатели и условия, которые нужно сравнить.",
            example="Сравни сценарии A и B по обеспеченности школами при населении N.",
        )
    if reason == "stalled":
        return NeededInput(
            missing="Новые данные или уточнённый критерий",
            reason="Повторение прежних шагов не добавляет подтверждений.",
            question="Уточните приоритетный показатель, сравниваемые сценарии или предоставьте недостающие исходные значения.",
            example="Проверь изменение населения и мощности школ в сценариях A и B.",
        )
    return NeededInput(
        missing="Доступный подтверждённый результат сервиса",
        reason=detail
        or "Один из сервисов не завершил необходимую проверку; это не подтверждает отсутствие объектов или нарушений.",
        question="Можно продолжить после восстановления сервиса. Если результат уже есть, сообщите значения вместе с единицами, сценарием и источником либо выберите независимую часть анализа.",
        example="Продолжи по доступным показателям, отметив непроверенную обеспеченность.",
        owner="service",
    )


def blocker_text(missing, completed=0):
    parts = [f"Подтверждённых шагов: {completed}. Анализ пока не завершён."]
    for item in missing:
        parts.extend([f"Не хватает: {item.missing}.", item.reason, item.question])
        if item.example:
            parts.append(f"Пример полезного уточнения: {item.example}")
    return "\n\n".join(parts)
