"""System prompts of the restriction pipeline's text steps.

Copied verbatim from ``RestrictionParserService.generate_plan_explanation`` and
``generate_final_response`` (including their indentation): the parity tests
compare the messages both engines send. A real migration moves these into one
shared module instead.
"""

from __future__ import annotations

from typing import Any

from src.agents.services.readable_refs import NO_SYSTEM_IDS_RULE


def plan_explanation_prompt(plan_summary: dict[str, Any]) -> str:
    return f"""Коротко и дружелюбно объясни пользователю, почему для его запроса выбраны такие параметры.
                Пиши обычным человеческим языком, без технических терминов.
                Не упоминай JSON, модель, инструмент, пайплайн, схему, поля или внутренние названия.
                {NO_SYSTEM_IDS_RULE}
                Не спорь с пользователем и не перегружай деталями.
                Объясни:
                - что выбрано как источник построения зон;
                - какой радиус используется и откуда он взят;
                - будут ли строиться только буферы или также ограничения для других объектов;
                - если есть целевые объекты, почему они выбраны.

                На этом этапе расчёт ещё не выполнен. Не сообщай количество найденных
                объектов, не перечисляй адреса и не приводи «примерные» результаты.
                Объясняй только выбранные параметры будущей проверки.

                Данные для объяснения:
                {plan_summary}
                """


def final_response_prompt(context: str) -> str:
    return f"""Дай комментарий к запросу пользователя на основе контекста статистики сгенерированных слоёв.
                Ответ давай только в виде обычного текста. Внимательно анализируй предоставленную в контексте информацию.
                Сообщи общее число затронутых объектов. Для каждого объекта из
                affected_objects назови его имя, применённое ограничение и причину
                попадания. Если details_truncated=true, явно скажи, что полный перечень
                объектов показан на карте. Если объектов нет, сообщи об этом прямо.
                Не показывай программный код.
                В качестве нормативной отсылки используй название документа и номер пункта,
                только когда они есть в причинах.
                {NO_SYSTEM_IDS_RULE}

                Контекст для ответа:

                {context}
                """
