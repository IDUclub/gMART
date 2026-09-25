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
Требуется NormGraph с `list_restrictions`, а для сужения проверки темой или
документом — с `resolve_entities`, `list_restriction_documents` и фильтром
`entities` (см. «Область проверки»). Нормы применяются только из документов,
действующих на территории сценария, поэтому нужен и IDU_DVD (`DVD_MCP_SERVER`,
см. «Документы, действующие на территории»).

Основной endpoint остаётся `GET /compliance/check/stream`. Старый
`GET /restrictions/generate_restrictions/stream` и `RestrictionPlan` сохранены как
legacy-контур и не меняют поведение.

## Поток исполнения

```text
запрос → документы, действующие на территории сценария (IDU_DVD)
        ▼
область: режим check | inventory, темы → сущности NormGraph, документы
        │           └─ неоднозначный документ → выбор пользователя в следующем сообщении
        ▼
NormGraph restriction + CheckPlan (list_restrictions с фильтрами области)
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
        ├─ check:     стабильный IDU MCP Check* tool → evidence + coverage
        └─ inventory: CreateRestrictionZones → зона действия нормы
        │
        ├─ Redis checkpoints / SSE reconnect
        └─ ChatStorage: tool calls + текстовый ответ
```

Нормы выполняются независимо. Ошибка одной нормы формирует для неё
`unverifiable/unknown` и не удаляет результаты остальных.

## Область проверки: темы и документы

Без условий проверяются все исполнимые нормы корпуса, как раньше. Запрос может
сузить проверку темой и документом:

- «проверь нормы касательно школ» — только нормы о школах;
- «проверь нормы по СП 42.13330 касательно жилой застройки» — нормы о жилой
  застройке из этого документа.

`ComplianceScopeResolver` (`services/compilance/compliance_scope.py`) работает так:

1. LLM только разбирает запрос: `topics` (виды объектов в именительном падеже) и
   `documents` (как их назвал пользователь; вид документа без номера — «из
   санитарных правил», «по СанПиНу» — тоже документ). Без тем и документов остаётся
   полная проверка.
2. Темы → сущности. NormGraph `resolve_entities` возвращает кандидатов (точное имя,
   алиас, основа слова, названия слоёв текущих CheckPlan, ближайшие по эмбеддингу)
   с числом норм и исполнимых норм. Названия слоёв нужны для объектов, которые
   встречаются только в плане проверки («детский сад» как слой нормы о жилых домах).
   LLM выбирает из них сущности, означающие тему; выбрать можно только
   предложенные NormGraph имена. Если модель не выбрала ничего, берутся точные
   совпадения по имени, алиасу или названию слоя. Тема без сущностей — сообщение «нет объектов,
   соответствующих теме», проверка не выполняется. Несколько тем объединяются по
   ИЛИ.
3. Документы. Обозначение, написанное в запросе (`СП 42.13330`, `СанПиН 2.2.1/…`,
   распознаётся тем же `parse_reference`, что и в document-QA), выбирает документ
   без вопроса, только если ему соответствует ровно один документ NormGraph
   (пробелы не важны, число не может продолжаться: «СП 42» ≠ «СП 421»). Иначе —
   и для описаний вроде «санпин о санитарных разрывах» — пользователю предлагается
   нумерованный список до 10 документов с исполнимыми нормами по теме
   (`list_restriction_documents(executable_only, entities)`), ранжированный по
   совпадению слов, а при отсутствии совпадений — по числу исполнимых норм.
4. Нормы читаются `list_restrictions` с фильтрами `entities` и `document_names`.
   Норма относится к теме, если сущность — её subject или object либо объявленный
   слой текущего CheckPlan. Если исполнимых норм не осталось, проверка не
   расширяется на весь корпус: ответ называет число найденных норм и документы, где
   исполнимые нормы по теме есть.

### Выбор документа

Вопрос отдаётся событием `clarification`:

```json
{
  "type": "clarification",
  "content": {
    "question": "Не удалось однозначно определить документ «СП 42». …\n1. СП 42.13330.2011 — исполнимых норм: 1\n2. СП 42.13330.2016 — исполнимых норм: 4\n…",
    "options": [
      {"number": 1, "label": "СП 42.13330.2011 — исполнимых норм: 1", "value": "СП 42.13330.2011"},
      {"number": 2, "label": "СП 42.13330.2016 — исполнимых норм: 4", "value": "СП 42.13330.2016"}
    ]
  }
}
```

Ожидающий выбор хранится в Redis (`pipeline:{ключ}:compliance_choice`, 24 ч) под
ключом диалога: `chat_id` для `/compliance`, чат оркестратора для `/orchestrator`,
`a2a-compliance:{contextId}` для A2A. Следующее сообщение того же диалога
разбирается против сохранённого списка: номера («2», «1, 3», «1-3», «второй»),
«все» или короткое название документа — без LLM; остальное классифицирует LLM.
Выбор продолжает **исходный** запрос (темы и сущности берутся из сохранённого
выбора). Несуществующий номер — вопрос повторяется. Новый вопрос вместо выбора
сбрасывает ожидание и обрабатывается как обычный запрос. Номер или «все» без
ожидающего списка (например, после истечения срока) не запускает полную проверку:
агент просит написать, какие нормы проверить.

Оркестратор, увидев ожидающий выбор в своём чате и ответ на него, не вызывает
планировщик, а сразу запускает шаг compliance с ответом пользователя.

Область фиксируется checkpoint `compliance_scope` (переподключение не повторяет
LLM-разбор), выводится статусом `compliance_scope`, строкой «Область проверки — …»
в итоговом тексте и отчёте и полем `scope` в `compliance_summary`.

### A2A

`POST /compliance/a2a` (карточка `GET /compliance/.well-known/agent-card.json`,
имя `compliance-agent`) запускает тот же пайплайн без записи в ChatStorage.
`scenario_id` обязателен (DataPart, metadata или `scenario_id=…` в тексте). Выбор
документа завершает задачу в `input-required`: текст вопроса и DataPart с
`options`. Ответ — следующее сообщение с тем же `contextId`. Слои нарушений
отдаются артефактами `compliance-layer-N` (GeoJSON), сводка — `compliance-summary`,
ссылка на отчёт — `compliance-file`.

## Документы, действующие на территории

И проверка, и перечень ограничений применяют только нормы документов, действующих
там, где находится сценарий. `ComplianceTerritoryFilter`
(`services/compilance/compliance_territory.py`) спрашивает IDU_DVD
`list_documents(scenario_id)` — общие документы под границей проекта сценария,
внутри неё и выше (муниципальные, региональные, федеральные) — и сопоставляет их
с документами NormGraph (`list_restriction_documents`, до 500) по `doc_id`
IDU_DVD, для старых синхронизаций — по имени без учёта регистра и пробелов.
Оставшиеся имена добавляются к фильтру `document_names` запроса норм; документ,
выбранный пользователем, пересекается с ними.

Фильтр строгий. Без `DVD_MCP_SERVER`, при ошибке IDU_DVD или если на территории
нет ни одного документа графа норм, нормы не применяются: ответ объясняет причину,
проверка и перечень не выполняются. Документ, названный в запросе и не действующий
на территории, отклоняется («… не действует на территории сценария»); списки для
выбора документа содержат только действующие документы. Итоговый текст называет
число документов графа норм, которые действуют и не действуют на территории;
`compliance_summary` и `restriction_inventory` содержат поле `territory`.

Результат фиксируется checkpoint `compliance_territory` и выводится статусом
`compliance_territory`.

## Ограничения на территории (перечень)

На вопрос «какие ограничения есть на территории проекта?» агент не проверяет
объекты, а показывает, где и какие нормы действуют. Тот же LLM-разбор области
возвращает `mode`: `inventory` для вопросов о действующих ограничениях и зонах
(«что ограничивает застройку», «покажи зоны ограничений», «какие ограничения
дают школы»), `check` — для проверки и поиска нарушений и при сомнении. Темы и
документы сужают перечень так же, как проверку («какие ограничения по школам из
СП 42»); режим сохраняется в ожидающем выборе документа.

Для каждой исполнимой нормы (эквивалентные объединяются, как в проверке)
`RestrictionZoneBuilder` (`services/compilance/compliance_inventory.py`) строит
зону по данным сценария. Нужны только слои, из которых строится зона: цели и их
атрибуты не загружаются и не требуются.

| Шаблон | Зона | Вид |
| --- | --- | --- |
| `distance_from_source` | буфер `distance_m` вокруг источников; `source_geometry` — сами источники | ограничения; при `violation_when=not_matched` — требуемого размещения |
| `distance_table` | буфер индивидуального радиуса по диапазону атрибута источника; источники без значения пропускаются | так же |
| `presence_within` | буфер `distance_m` вокруг объектов, рядом с которыми должны быть соседи | требуемого размещения |
| `zonal_attribute_threshold` | функциональные зоны нормы, обрезанные по территории проекта, с порогом (постоянным или атрибутом зоны) | ограничения |
| `zonal_ratio` | функциональные зоны нормы в границах проекта с требуемой долей площади | ограничения |

Зональная норма для всех зон (`functional_zones`) с одним порогом действует на
всю территорию проекта: зона — граница проекта (`GetProjectTerritory`). Без
границы проекта такая норма считается непостроенной, а зоны конкретного типа
выдаются необрезанными. Буферы строятся полностью, только вокруг объектов
сценария; объекты контекста не загружаются.

Статусы норм: `shown` — зона есть на карте; `no_objects` — в полном слое
сценария нет объектов, от которых действует норма; `unverifiable` — не хватает
данных; `unsupported` — план не прошёл валидацию. Нормы без исполнимого плана
учитываются только счётчиком.

IDU MCP:

- `CreateRestrictionZones(layer_name, geometry_mode, source_layer, layers, …)` —
  режимы `buffer`, `attribute_buffer`, `geometry`; `threshold_field` копирует порог
  из атрибута зоны, `clip_layer` обрезает по объединению слоя, `properties`
  добавляет атрибуты нормы. Ответ `{layer_name: FeatureCollection}`, в `meta` —
  число исходных объектов, зон и пропущенных объектов.
- `GetProjectTerritory(scenario_id)` — `{"project_territory": FeatureCollection}`
  с границей проекта.

Атрибуты зоны: `restriction_title`, `restriction_description` (текст нормы),
`restriction_id`, `zone_kind` (`restriction` | `required`), `applies_to`
(объекты, к которым относится норма), `provenance` (документ и пункт), для
буферов — `buffer_size`, для зональных норм — `operator`, `threshold`, `unit`.

События SSE: статус `restriction_zones`; `restriction_zone` на каждую норму (без
геометрии); `feature_collection` «Зона ограничения — СП 42.13330.2016, п. 7.1» или
«Зона требуемого размещения — …» только для непустых зон; итоговое
`restriction_inventory` со счётчиками и списком зон; текст с перечнем зон на
карте; ссылка на отчёт `restriction_inventory_report_<scenario_id>_<время>.md`.
Checkpoint `restriction_inventory` защищает от повторного построения при
переподключении. В историю записываются вызовы получения данных и
`CreateRestrictionZones` только для показанных зон, поэтому ChatStorage
восстанавливает их при открытии чата (ChatStorage подставляет накопленные слои в
`layers` этого инструмента). A2A отдаёт сводку артефактом
`compliance-restriction_inventory`.

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

- `compliance_scope` — разбор тем и документов и применённая область;
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

События `feature_collection`: для каждой нарушенной нормы отдельно отдаётся слой
«Нарушение нормы — …», а перед ссылкой на отчёт — один общий слой «Объекты без
нарушений» со всеми проверенными объектами, которые не нарушили ни одной нормы.

## ChatStorage и повтор расчёта

В историю записываются только вызовы инструментов (`tool_call`) и текстовый ответ
(`text`), включая вопросы для уточнения. Статусы, `check_plan`,
`requirement_resolution`, `compliance_result` и `compliance_summary` не включаются
в сообщение MongoDB, чтобы объёмные результаты проверки не превышали лимит BSON.
Полные результаты остаются доступны в SSE-потоке и Redis-журнале текущего запроса.

Из вызовов получения данных (`GetServices`, `GetPhysicalObjects`,
`GetFunctionalZones`) в историю попадают только те, что вернули объекты. Клиент,
открывающий чат заново, повторяет сохранённые вызовы через ChatStorage и показывает
их результаты слоями; вызов без объектов (тип зоны, которого нет в сценарии)
превращался бы в пустой слой с одним `meta`.

При восстановлении слоёв ChatStorage повторяет сохранённые стабильные MCP-вызовы и
подставляет заново полученные слои в новые геометрические инструменты. Это
**повторный расчёт на текущем сценарии**, а не гарантия идентичности исторического
результата. Для точного воспроизведения evidence содержит `input_revision`, когда
источник его предоставляет; без immutable revision или снимка интерфейс не должен
утверждать, что результат идентичен прошлому.

Проверка не ограничивает число объектов: шаблон получает все объекты, пришедшие из Urban
API, и evidence хранит все ссылки (`generator_refs`, `zone_refs`, `used_fields`). Для
тяжёлых слоёв остаётся только тайм-аут операции.

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
Passed and non-executed checks emit no per-norm layers. Instead, right before the
report `file` frame (after the final `chunk`), one `feature_collection` named
`Объекты без нарушений` merges every checked object that passed at least one norm
and violated none (`services/compilance/compliance_layers.py`). Each object appears
once, keyed by `object_ref.id`; per-norm verdict fields are dropped and
`passed_norms` lists the document/clause references it passed, equivalent norms
included. The layer is omitted when no checked object is compliant. `compliance_result` and
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

`PUBLIC_BASE_URL` must be the public origin through which clients reach the agents
app (for the Urban Assistant proxy, e.g. `https://<host>/gmart`). Left empty, the
links are relative (`/files/…`) and a client served from another origin resolves
them against itself, receiving its own page instead of the report.

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
