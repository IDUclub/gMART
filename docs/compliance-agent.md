# Compliance agent: исполняемые нормативные ограничения

Compliance agent получает норму и опциональный `CheckPlan` из NormGraph,
сопоставляет требования плана с данными сценария и запускает только
зарегистрированный детерминированный шаблон. LLM не вычисляет геометрию, числа,
статусы или provenance и не передаёт произвольную последовательность MCP-вызовов.

Основной endpoint остаётся `GET /compliance/check/stream`. Старый
`GET /restrictions/generate_restrictions/stream` и `RestrictionPlan` сохранены как
legacy-контур и не меняют поведение.

## Поток исполнения

```text
NormGraph restriction + CheckPlan
        │
        ▼
точная валидация schema/template version
        │
        ▼
реестр gMART → effective requirements
        │
        ▼
Urban API layers → data gate → resolved requirements
        │
        ▼
стабильный IDU MCP tool → evidence + coverage
        │
        ├─ Redis checkpoints / SSE reconnect
        └─ ChatStorage structured parts
```

Нормы выполняются независимо. Ошибка одной нормы формирует для неё
`unverifiable/unknown` и не удаляет результаты остальных.

## CheckPlan v1

Минимальный пример:

```json
{
  "schema_version": "1.0",
  "template": "distance_from_source",
  "template_version": 1,
  "params": {
    "source_layer": "schools",
    "targets": ["residential_buildings"],
    "geometry_mode": "buffered",
    "predicate": "intersects",
    "violation_when": "matched",
    "result_mode": "both",
    "distance_m": 100
  },
  "declared_requirements": {
    "layers": [
      {
        "role": "schools",
        "entity": "школа",
        "entity_type": "service",
        "geometry_types": ["Point", "MultiPoint"],
        "required": true
      },
      {
        "role": "residential_buildings",
        "entity": "жилой дом",
        "entity_type": "physical_object",
        "geometry_types": ["Polygon", "MultiPolygon"],
        "required": true
      }
    ],
    "attributes": []
  },
  "source": {
    "restriction_id": "restriction-uuid",
    "document_name": "СП 42.13330.2016",
    "clause_number": "5.5",
    "extraction_text": "..."
  },
  "planner_status": "auto"
}
```

Контракт использует `extra=forbid`, ограниченные enum, длины списков и числовые
диапазоны. Неизвестная версия схемы или точной пары `template@version` не
подбирается автоматически и даёт `unsupported/unknown`. Обязательные требования
реестра добавляются к заявленным NormGraph и не могут быть ослаблены входным
планом.

Поддерживаемые пары:

| Шаблон | IDU MCP tool | Назначение |
| --- | --- | --- |
| `distance_from_source@1` | `CheckDistanceFromSource` | отношение цели к источнику или его буферу |
| `distance_table@1` | `CheckDistanceTable` | индивидуальный буфер по диапазону атрибута источника |
| `presence_within@1` | `CheckPresenceWithin` | полный left/anti-join соседей |
| `zonal_attribute_threshold@1` | `CheckZonalAttributeThreshold` | сравнение атрибута с константой или порогом зоны |
| `zonal_ratio@1` | `CheckZonalRatio` | доля объединённой площади числителя внутри зоны |

Публичный manifest создаётся `TemplateRegistry.public_manifest()`. Общие fixtures
`tests/contract/check_plan_cases.json` выполняются в gMART и NormGraph.

## Гейт данных

Гейт получает полный слой, строит профиль полей, геометрий, null-count и fill-rate,
а затем перебирает кандидаты атрибута в объявленном порядке. В результате для
каждой роли фиксируются конкретный слой/поле, единица и качество `direct` либо
`derived`.

Единственное зарегистрированное преобразование первой версии —
`height_to_floors_v1 = max(1, floor(height_m / 3))`. Оно применяется только при
явном derived-кандидате. Ошибка загрузки, truncation, неправильная геометрия,
нулевой fill-rate либо fill-rate ниже `min_fill_rate` не заменяются догадкой.

Результаты гейта:

- `complete` — проверены все применимые объекты;
- `partial` — часть объектов осталась unchecked;
- `unverifiable` — обязательное требование не разрешено;
- `not_applicable` — полный подтверждённый слой применимых объектов пуст;
- `unsupported` — план или шаблон не поддерживается.

`compliance_status` хранится отдельно: `passed`, `violated` либо `unknown`.
Комбинация `partial + passed` означает только отсутствие нарушений на проверенной
части.

Результаты аудита реальных сценариев и решение по включению T1–T5 находятся в
[compliance-data-audit.md](compliance-data-audit.md).

## SSE и checkpoints

К существующему потоку добавлены статусы:

- `check_plan_validation`;
- `requirements_resolution`;
- `template_execution`;
- `verdict_aggregation`.

И структурированные события:

| `type` | `content` |
| --- | --- |
| `check_plan` | `restriction_id` и принятый план |
| `requirement_resolution` | effective/resolved/missing requirements |
| `compliance_result` | полный результат одной нормы с coverage и evidence |
| `compliance_summary` | итоговые счётчики всего запроса |

После каждого этапа данные сохраняются в Redis. При повторном подключении с тем же
`request_id` сервис сначала отдаёт сохранённый буфер событий; завершённый pipeline
не запускает вычисления повторно.

События `feature_collection` сохранены. Для каждой нормы отдельно отдаются слои
«Нарушения» и «Проверено без нарушений», если соответствующий `result_mode`
разрешает их вернуть.

## ChatStorage и повтор расчёта

В историю записываются структурированные parts:

- `check_plan`;
- `requirement_resolution`;
- `compliance_result`;
- `compliance_summary`;
- существующие `tool_call`.

При восстановлении слоёв ChatStorage повторяет сохранённые стабильные MCP-вызовы и
подставляет заново полученные слои в новые геометрические инструменты. Это
**повторный расчёт на текущем сценарии**, а не гарантия идентичности исторического
результата. Для точного воспроизведения evidence содержит `input_revision`, когда
источник его предоставляет; без immutable revision или снимка интерфейс не должен
утверждать, что результат идентичен прошлому.

## Экспертное ревью

gMART проксирует защищённые операции NormGraph:

- `GET /compliance/check-plans/review?limit=50` — очередь pending/auto планов;
- `POST /compliance/check-plans/{restriction_id}/review` — действие `approve`,
  `reject` или `replace`, опциональная причина `reason` и полный новый план в поле
  `plan` для `replace`. Автор берётся из проверенной пользовательской идентичности,
  а не из тела запроса.

NormGraph сохраняет неизменяемую ревизию с автором и временем. Автоматическое
повторное извлечение не перезаписывает `reviewed` план.

## Наблюдаемость

Структурная строка завершения каждой нормы содержит `request_id`,
`restriction_id`, `template`, `template_version`, `planner_status`, длительности
разрешения данных и исполнения, coverage, fill-rate, число нарушений и код исхода.
Токены и содержимое пользовательского запроса в эту строку не включаются.

Агрегированные process-local метрики доступны через
`GET /system/compliance-metrics`: количество норм по шаблону, статусы
проверяемости, auto/reviewed планы, статистика длительности и fill-rate, ошибки
нижестоящих Urban/IDU MCP операций. После перезапуска процесса счётчики обнуляются;
для долговременных графиков endpoint должен опрашиваться системой мониторинга.

## Проверка и эксплуатация

```bash
env SERVICE_AUTH_SERVER_URL=http://localhost \
  SERVICE_AUTH_REALM=test SERVICE_AUTH_CLIENT_ID=test \
  SERVICE_AUTH_CLIENT_SECRET=test \
  uv run pytest tests/unit

cd frontend && npm run build
```

Лимиты конкретного шаблона находятся в manifest: максимум features и payload,
timeout, допустимые геометрии и версия evidence. Тяжёлые GeoPandas-операции IDU MCP
запускает через `asyncio.to_thread`; ошибки входных данных преобразуются в
`ToolError`.
