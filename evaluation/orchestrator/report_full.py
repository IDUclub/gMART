"""Render measured routing, execution and evidence-review results separately."""

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
RUNS = ROOT / "benchmarks/data/orchestrator_20260910"


def load(stage):
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((RUNS / stage / "runs").glob("*.json"))
    ]


def metric(rows, live=False):
    passed = sum(
        (
            bool(r.get("routing_checks")) and all(r["routing_checks"].values())
            if live
            else r["passed"]
        )
        for r in rows
    )
    return f"{passed}/{len(rows)} ({passed/len(rows):.1%})" if rows else "не завершён"


def execution(rows):
    count = Counter()
    for row in rows:
        steps = (row.get("final") or {}).get("steps", [])
        if row.get("error"):
            count[row["error"]] += 1
        elif steps and all(s["status"] == "completed" for s in steps):
            count["all_steps_completed"] += 1
        elif (
            any(s["status"] == "needs_clarification" for s in steps)
            or (row.get("plan") or {}).get("mode") == "needs_clarification"
        ):
            count["clarification"] += 1
        else:
            count["failed_or_partial"] += 1
    return dict(count)


before, routing_v1 = load("baseline"), load("final")
live_before, live_after = load("live_baseline"), load("live_final")
mixed_before, mixed_after = load("mixed_routing_baseline"), load("mixed_live_64k")
lines = [
    "# Проверка оркестратора — 10 сентября 2026",
    "",
    "**200 базовых + 100 дополнительных составных запросов.** Охвачены restriction, compliance, provision, scenario_data, documents, norms.",
    "",
    "Модель: gpt-oss-20b, удалённый сервер из graphify local-gpu (http://10.32.11.27:8001/v1). Оркестратор работает локально. Все трассы получены от настоящих модели и сервисов.",
    "",
    "## Маршрутизация",
    "",
    "| Этап | Совпадение с эталоном |",
    "|---|---:|",
    f"| Базовые 200, исходный планировщик | {metric(before)} |",
    f"| Базовые 200, первый этап исправлений | {metric(routing_v1)} |",
    f"| Базовые 200, итоговый полный прогон | {metric(live_after,True)} |",
    f"| Mixed 100, первый прогон планировщика | {metric(mixed_before)} |",
    f"| Mixed 100, итоговый полный прогон | {metric(mixed_after,True)} |",
    "",
    "**Это не процент достоверных конечных ответов.** Проверяются режим, точный набор агентов, отсутствие дубликатов, вопрос при уточнении и размеченные детали истории. Не все параметры и порядок шагов покрыты метрикой.",
    "",
    "На первом этапе отложенные 40 примеров дали 25/40 → 37/40. После последующих исправлений они уже не являются новой независимой выборкой. Mixed разделён на 80 диагностических и 20 отложенных примеров. Наборы синтетические; стабильность повторных ответов не гарантируется.",
    "",
    "Неоднозначные эталоны сохранены для сопоставимости: orch-123 ссылается на отсутствующую историю; mix-044 «список оставь» и mix-049 отмена буфера не обязательно требуют повторения завершённой задачи. Отличие от эталона здесь не доказывает ошибку.",
    "",
    "## Полное исполнение",
    "",
    "| Прогон | Трассы | Исходы запросов |",
    "|---|---:|---|",
]
for name, rows, total in [
    ("Исходные 200", live_before, 200),
    ("Повторные 200", live_after, 200),
    ("Mixed 100", mixed_after, 100),
]:
    lines.append(
        f"| {name} | {len(rows)}/{total} | `{json.dumps(execution(rows),ensure_ascii=False)}` |"
    )
lines += [
    "",
    "all_steps_completed означает завершение всех шагов, включая ответы «данных не найдено»; это не доказательство решения задачи. clarification — уточнение; failed_or_partial — ошибка или частичное исполнение.",
    "",
    "Реальные локальные MCP: IDU 8002, Effects 8080, DVD 8100, NormGraph 8020; настроенные Urban MCP/REST; OIDC bridge 8085. Состояние изолировано в fakeredis, запись в ChatStorage отключена, история задаётся набором. Развёрнутый HTTP/SSE-контейнер agents этим прогоном не проверяется.",
    "",
    "Перед повторным прогоном восстановлен локальный Ollama для эмбеддингов bge-m3. Поэтому улучшение исполнения отражает и исправление зависимости стенда. Генеративная модель gMART оставалась удалённой local-gpu. Полный mixed использует подтверждённое сервером окно 65536 токенов для документов; первый медленный запуск mixed с окном 8192 остановлен и сохранён отдельно. Его частичные результаты не смешиваются с полным набором.",
    "",
    "## Проверка обоснованности",
    "",
    "Сохранены независимые REST-снимки сценариев 772 и 848 под той же подтверждённой личностью. Проверка ID подтверждает существование возвращённых объектов, но не полноту фильтра. Буферы и наборы пересечений пересчитываются через GeoPandas/Shapely по входным слоям без вызова реализации MCP; выбор правильного условия моделью этим не проверяется.",
    "",
]
for title, filename in [
    ("Базовые", "geometry_final.json"),
    ("Mixed", "geometry_mixed.json"),
]:
    path = RUNS / filename
    if path.exists():
        g = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"- {title}: геометрические проверки {g['passed']}/{g['checks']}; {g['errors']} трасс не удалось проверить."
        )
for title, filename in [
    ("Базовые", "reference_ids_final.json"),
    ("Mixed", "reference_ids_mixed.json"),
]:
    path = RUNS / filename
    if path.exists():
        g = json.loads(path.read_text(encoding="utf-8"))
        lines.append(
            f"- {title}: существование ID подтверждено независимым REST для {g['passed']}/{g['calls']} проверенных вызовов."
        )
lines += [
    "",
    "Дословное совпадение длинных цитат с полученными источниками: 1/1 в базовом наборе, 10/13 в mixed. Это узкая проверка цитат, не всех утверждений. В mix-003 и mix-042 различается пунктуация; в mix-054 таблица пересказана внутри кавычек. Числа таблицы найдены в источнике, но пересказ нельзя считать дословной цитатой, а вывод «нормы не применяются» из прочерков требует отдельной проверки.",
    "",
    "Пробный автоматический разбор 200 исходных ответов **той же LLM** оказался ненадёжным: судья путал выборку из трёх свойств с полным числом геометрий и предполагал существование норм при пустом результате графа. Поэтому его оценки НЕ используются как процент точности и НЕ публикуются как подтверждённые ошибки. Сырые результаты сохранены для аудита. Итоговые выводы опираются на воспроизводимые проверки и вручную подтверждённые случаи в findings.md; ручная экспертная аттестация всех утверждений 300 ответов не выполнена.",
    "",
    "## Исправления и ограничения",
    "",
    "Подробный разбор подтверждённых ошибок, внесённых изменений и оставшихся ограничений — в [findings.md](findings.md). Результаты каждого запроса — в [results.md](results.md).",
    "",
    "## Проверки кода",
    "",
    "723 unit-теста прошли на Windows; ещё 4 Linux-зависимых теста workspace_store отдельно прошли в одноразовом контейнере до последнего исправления обеспеченности. Frontend: 20 тестов и production build прошли ранее в этой работе. black/isort применены к изменённому Python-коду; graphify update . выполнен.",
    "",
    "Наборы: queries.md/cases.json и mixed_queries.md/mixed_cases.json. Необработанные локальные трассы, манифесты и REST-снимки: benchmarks/data/orchestrator_20260910 (исключены из Git).",
]
lines += [
    "",
    "## Повторная проверка числовых ответов",
    "",
    "После завершения полных 200+100 прогонов дополнительно исправлены ответы обеспеченности: числовой итог формируется непосредственно из полей Effects, без свободной интерпретации LLM; отсутствие расчёта возвращает ошибку, а не успешный ответ. Полные метрики выше относятся к предыдущему снимку source_live_final.zip. Последнее исправление проверено отдельными прогонами, оно не выдаётся за новый полный прогон 300 запросов.",
    "",
]
for stage in [
    "provision_factual_final",
    "mixed_factual_regression",
    "mixed_factual_extended",
]:
    rows = load(stage)
    lines.append(
        f"- {stage}: {len(rows)} запросов, исходы `{json.dumps(execution(rows),ensure_ascii=False)}`."
    )
lines += [
    "",
    "Ручная сверка orch-051 и mix-001: 3060/3637 = 84,1% покрытия суммарного спроса; средняя и медианная обеспеченность подписаны как показатели по зданиям. В mix-001 сохранено различие между 3 школами в выбранном сценарии и 4 объектами в расчёте Effects. Число из предыдущего шага больше не подменяет поле расчёта. Уточнения и ошибки в повторной выборке не считаются подтверждёнными расчётами.",
    "",
    "mix-013 сохраняет заданные 18000 жителей в расчёте, но планировщик всё ещё теряет отдельную просьбу вывести сохранённое население. mix-021 завершает три шага: список школ, буфер 250 м, обеспеченность. Это подтверждает конкретные улучшения и одновременно показывает оставшуюся неполноту составных планов.",
]
path = RUNS / "provision_numerical.json"
if path.exists():
    numerical = json.loads(path.read_text(encoding="utf-8"))
    lines.append(
        f"\nСверка числовых полей ответа с соответствующими сервисами в ответе Effects: {numerical['passed']}/{numerical['checks']} в {len(numerical['cases'])} трассах. Проверены количество, среднее/медиана и рассчитанная доля спроса. Это проверка передачи значений, а не независимая валидация алгоритма Effects."
    )
if len(live_after) != 200 or len(mixed_after) != 100:
    lines.insert(
        2, "**Промежуточный отчёт: полные повторные прогоны ещё выполняются.**\n"
    )
(HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
details = [
    "# Поштучные результаты",
    "",
    "Автоматический разбор не заменяет экспертную проверку. Полные трассы сохранены локально.",
    "",
]
for title, dataset, rows, stage in [
    ("Базовые 200", "cases.json", live_after, "reviews_final"),
    ("Составные 100", "mixed_cases.json", mixed_after, "reviews_mixed"),
]:
    cases = {
        c["id"]: c for c in json.loads((HERE / dataset).read_text(encoding="utf-8"))
    }
    details += [f"## {title}", ""]
    for row in rows:
        details += [
            f"### {row['id']}",
            "",
            cases[row["id"]]["query"],
            "",
            "Маршрут: "
            + " → ".join(s["agent"] for s in (row.get("plan") or {}).get("steps", []))
            + ".",
            "Исход: " + str(execution([row])) + ".",
            "",
        ]
        details += [
            "Содержательная достоверность не выводится из статуса выполнения; проверенные типы операций и конкретные ошибки описаны в отчёте.",
            "",
        ]
(HERE / "results.md").write_text("\n".join(details) + "\n", encoding="utf-8")
print(f"Report: baseline {len(live_after)}/200, mixed {len(mixed_after)}/100")
