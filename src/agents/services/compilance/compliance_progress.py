"""Weighted, monotone compliance progress, alongside the legacy counters."""

STAGES = {
    "compliance_territory": (0, 3, "Определение территории"),
    "compliance_scope": (3, 6, "Область проверки"),
    "norms_loading": (6, 10, "Загрузка норм"),
    "check_plan_validation": (10, 14, "Проверка планов"),
    "dedup": (14, 16, "Объединение эквивалентных норм"),
    "source_layers": (16, 27, "Получение источников"),
    "zones_building": (27, 40, "Построение зон"),
    "objects_by_geometry": (40, 52, "Получение объектов в зонах"),
    "requirements_resolution": (52, 55, "Проверка данных"),
    "template_execution": (55, 85, "Проверка норм"),
    "violation_geometry": (55, 85, "Геометрия нарушений"),
    "verdict_aggregation": (85, 91, "Сводка результатов"),
    "cache": (55, 85, "Сохранение результатов"),
    "report_build": (91, 94, "Подготовка отчёта"),
    "archive": (94, 97, "Сборка архива"),
    "llm_summary": (97, 100, "Подготовка рекомендаций"),
}


def progress_event(stage, current=0, total=1, label=None, *, minimum=0):
    start, end, ru = STAGES[stage]
    percent = max(
        minimum, round(start + (end - start) * min(current / total if total else 1, 1))
    )
    label = label or f"{ru} {current}/{total}"
    filled = percent // 10
    return {
        "type": "progress",
        "content": {
            "stage": stage,
            "stage_ru": ru,
            "percent": percent,
            "current": current,
            "total": total,
            "label": label,
            "bar": f"{'▰' * filled}{'▱' * (10 - filled)} {percent}% · {label}",
        },
    }
