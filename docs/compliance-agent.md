# Compliance agent: исполняемые нормативные ограничения

Compliance agent отбирает нормы с валидным исполнимым `CheckPlan` из NormGraph,
сопоставляет требования плана с данными сценария и запускает только
зарегистрированный детерминированный шаблон. LLM не вычисляет геометрию, числа,
статусы или provenance и не передаёт произвольную последовательность MCP-вызовов.

Нормы читаются из NormGraph целиком, страницами по 200 через `list_restrictions`
(курсор `next_after_id` по id ограничения). Одним окном корпус не запрашивается:
ответ на тысячи норм с планами и провенансом исчерпывал память NormGraph и
перезапускал его, а запрос без таймаута зависал. Вызовы NormGraph ограничены
120 с. Нормы без исполнимого плана отбрасываются в gMART, а не фильтром
`executable_only`, чтобы отчёт показывал, сколько норм пропущено без плана.
Требуется NormGraph с `list_restrictions`.

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
        └─ ChatStorage: tool calls + текстовый ответ
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

В историю записываются только вызовы инструментов (`tool_call`) и текстовый ответ
(`text`), включая вопросы для уточнения. Статусы, `check_plan`,
`requirement_resolution`, `compliance_result` и `compliance_summary` не включаются
в сообщение MongoDB, чтобы объёмные результаты проверки не превышали лимит BSON.
Полные результаты остаются доступны в SSE-потоке и Redis-журнале текущего запроса.

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


### Inflected entity names

Before fetching scenario layers, compliance calls `ResolveUrbanEntityTypes`.
The tool reads the global Urban API type dictionaries and matches full phrases
by their Russian word normal forms. For example, `спортивных площадок`, `школ`
and `жилых домов` resolve to `Спортивная площадка`, `Школа` and `Жилой дом`.
The executor uses the returned canonical names; stored restriction text and plans
are not rewritten. Types with no instances in the scenario can still resolve.

Exact catalog names take precedence. All qualifiers and tokens must match, and
ambiguous or unknown phrases remain unresolved (`unverifiable`); morphology does
not substitute synonyms or turn `трёхэтажные жилые дома` into all residential
buildings. `pymorphy3` provides dictionary forms without an LLM round trip.


### Independent plan execution

Compliance accepts only persisted CheckPlans with supported schema/template versions,
valid parameters and `auto`/`reviewed` planner status. Missing, malformed and
`unsupported` plans are discarded immediately after retrieval, before checkpointing
or calculation; the SSE status reports the skipped count. Old checkpoints pass the
same gate before execution. Each accepted norm is executed separately and produces
its own result, so one failure does not discard the others.

A distance mentioned in the request no longer switches compliance to LLM plan
generation. The corpus is never passed to `RestrictionPlanBuilder` in compliance
mode. Ad hoc geometry conditions remain available through `/restrictions`.
Without NormGraph, or without accepted plans, compliance reports that the check
was not performed instead of inventing plans or claiming compliance.


### UI layers and final explanation

Only non-empty violation layers are emitted, as `feature_collection` events named
`Нарушение нормы — СП 42.13330.2016, п. 7.1` (without the clause when absent).
If the document name is missing, the label is `Источник не указан`. A short SP
designation is extracted from a full document title when available; other document
names are retained. Repeated layer names receive a numeric suffix, never a UUID.
Passed checks and non-executed checks emit no map layers. `compliance_result` and
`compliance_summary.results` retain verdicts, coverage, source and evidence but
omit `violated_features` and `passed_features`; geometry is not duplicated there.
The final text lists each violated norm by document/clause, its requirement
and the number of violating objects. Partial checks also state the unchecked count.
Counts are per norm and must not be added as a count of unique objects.
The same textual references, including equivalent sources, are supplied to the
model for follow-up answers. Missing source names use `Источник не указан`;
internal restriction IDs remain in structured results and evidence only. If an
older plan lacks source labels, compliance fills them from the retrieved norm’s
`provenance.name` and `provenance.numbering` without changing its internal ID.

### Equivalent checks

Before execution, plans with identical template versions, parameters, layer roles
and data requirements are grouped. Entity names are resolved through the Urban API
catalogue first. Different thresholds, directions, geometry constraints or attribute
candidate order are not merged. If catalogue resolution fails or is ambiguous, checks
remain separate and the executor reports the missing requirement.

One `check_plan`, result and (only when violated) `feature_collection` is emitted per
group. `check_plan.content.equivalent_sources`, the result's `source.equivalent_sources`
and the summary's `equivalent_sources` preserve all source references. The summary
includes `duplicate_checks`; its existing norm counts refer to executed unique checks.
The final text names equivalent sources. Stored NormGraph norms are not deleted.

### Markdown report

When at least one norm was actually checked (`passed` or `violated`), the stream
closes with an `event: file` frame after the final `chunk`: a link to a Markdown
report (`compliance_report_<scenario_id>_<YYYYMMDD-HHMM>.md`). It is built
deterministically from `compliance_summary` by
`services/compilance/compliance_report.py`, without the LLM, and contains:

- summary counters: checked norms, failed, passed, checked as equivalent, not
  checked, skipped without an executable plan;
- «Не прошли проверку» and «Прошли проверку»: per norm — requirement text,
  coverage and fill-rate, violated/passed object counts, template parameters,
  resolved layers/fields and up to 20 violating objects with measured value,
  condition and related sources/zones; `partial + passed` is marked as partial;
- «Прошли без применимых объектов»: norms that passed only because the scenario
  has no object they apply to (warning `no_applicable_objects`). They are counted
  in «Прошли проверку» with a separate «из них без применимых объектов» row and
  are shown as a formal pass, without coverage or fill-rate. The final answer text
  reports the same count;
- «Проверены как эквивалентные»: each dedup group with the executed norm, its
  inherited verdict and the equivalent sources. Merged norms are distinct by
  `restriction_id`: unnumbered clauses of one document share a label yet are
  counted separately (the label then shows «(2 нормы)»).

`unverifiable`, `unsupported` and `not_applicable` norms appear only in the
counters. Without a checked norm no report and no `file` event are produced.

The file is written to the container's temp directory (no volume) and served by
`StaticFiles` mounted at `/files`, behind `OwnedFilesApp`: a Bearer token is
required and only the author (`sub`) receives it; after one hour it is refused
with `404` and deleted (lazily and by a 5-minute purge task). `url` opens it
inline, `download_url` (`?download=1`) as an attachment; both are built on
`PUBLIC_BASE_URL`. The link, without `role` and `download_url`, is stored in
ChatStorage as a `kind: "file"` part. With several agents replicas the file lives
only on the replica that produced it.

### Geometry retrieval

Compliance explicitly sends `centers_only=false` to `GetServices` and
`GetPhysicalObjects`. Both tools forward the flag to the corresponding Urban API
`*_with_geometry` query, and layer revisions distinguish centers from full geometry.
Deploy the Agents API and IDU MCP together when introducing this tool parameter.
The flag requests stored geometry; it cannot turn a stored Point into a polygon.
Polygon requirements remain enforced when the returned source geometry is a Point.
