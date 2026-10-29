# Контракт статусов проверки норм (ICII-226)

Документ описывает текущую реализацию gMART, включая отличия от спецификации.
Обзор для пользователя: [compliance-status-guide.md](compliance-status-guide.md).
Запуск, загрузки, CheckPlan и ревью: [compliance-agent.md](compliance-agent.md).
Имена Python-пакетов сохраняют написание `compilance`.

## Запуск и транспорт SSE

`GET|POST /compliance/check/stream` принимает запрос, `scenario_id` либо `upload_ids`,
модель и контекст диалога; при переподключении передавайте прежний `request_id`.
Подробности параметров и авторизации — [frontend-service.md](frontend-service.md).
Ниже приведены JSON payload событий пайплайна, а не единый JSON-ответ.
Файл на HTTP-проводе передаётся отдельно: `event: file`, плоский дескриптор в `data`.

### Последовательность check

1. `pipeline_started` с `request_id`; возможны служебные события создания чата.
2. Территория (`compliance_territory`), область (`compliance_scope`), загрузка норм
   (`norms_loading`, также прежний статус `norm_retrieval`), валидация
   (`check_plan_validation`), `check_plan` для принятых планов и `dedup`.
   При выборе документа или сопоставлении файлов возможен `clarification`: это остановка,
   а не успешное завершение проверки.
3. `source_layers`, затем `zones_building` для норм,
   затем `objects_by_geometry` с уведомлениями по завершённым типам сервисов.
   `tool_call` содержит вызовы источников, `CreateRestrictionZones`, получения целей.
4. `requirements_resolution`; начальный `compliance_progress`.
5. Для каждой нормы: `template_execution`, исполнение, `violation_geometry`, `cache`,
   `tool_call` выполненной проверки, `requirement_resolution`, `compliance_result`,
   `compliance_progress`, обновлённый `progress` этапа `template_execution`.
   При чтении checkpoint некоторые операции и их статусы пропускаются.
6. `verdict_aggregation`, `report_build`, `archive`; `compliance_summary` уже содержит
   итоговые счётчики и относительные пути внутри архива.
7. `llm_summary`; `progress.percent=100`; финальные `chunk` (до 600 символов каждый,
   последний `done=true`). Модель сначала полностью генерирует ответ, затем он делится на чанки.
8. `feature_collection` для каждого доступного слоя нормы; затем `file` для `summary.md`;
   последним содержательным событием — `file` для ZIP. `chunk.done` завершает только текст:
   **не закрывайте соединение до слоёв и файлов**.

Файлы могут отсутствовать, если нет владельца / file store или сохранение файла завершилось
`OSError`. Слои и результат при этом сохраняются, pipeline может завершиться успешно.
Нельзя требовать строго по одному сообщению на каждый этап: статусы повторяются и ветвятся.

Пример последнего текстового чанка (полученные до него чанки нужно конкатенировать):

```json
{"type":"chunk","content":{"text":"Сначала рассмотрите нарушенные нормы, затем уточните данные частичных проверок.","done":true}}
```

```json
{"type":"pipeline_started","content":{"request_id":"run-1"}}
```

```json
{"type":"status","content":{"status":"template_execution","text":"Проверяю СП 42.13330.2016, п. 7.1 · 0/1"}}
```

```json
{"type":"progress","content":{"stage":"template_execution","stage_ru":"Проверка норм","percent":70,"current":1,"total":2,"label":"Проверка норм 1/2","bar":"▰▰▰▰▰▰▰▱▱▱ 70% · Проверка норм 1/2"}}
```

### Этапы progress

`current`, `total` — неотрицательные целые; `label` — пояснение текущей операции,
`stage_ru` — постоянное название. `bar` содержит 10 сегментов, заполнено `percent // 10`.
Внутри диапазона процент интерполируется по `current/total` с округлением;
при `total=0` этап считается законченным. `_buf` ограничивает процент снизу предыдущим
значением и сохраняет его в checkpoint. Это взвешенный ход работы, не оценка времени.

| stage | stage_ru | Диапазон, % |
|---|---|---|
| compliance_territory | Определение территории | 0–3 |
| compliance_scope | Область проверки | 3–6 |
| norms_loading | Загрузка норм | 6–10 |
| check_plan_validation | Проверка планов | 10–14 |
| dedup | Объединение эквивалентных норм | 14–16 |
| source_layers | Получение источников | 16–27 |
| zones_building | Построение зон | 27–40 |
| objects_by_geometry | Получение объектов в зонах | 40–52 |
| requirements_resolution | Проверка данных | 52–55 |
| template_execution | Проверка норм | 55–85 |
| violation_geometry | Геометрия нарушений | 55–85 |
| cache | Сохранение результатов | 55–85 |
| verdict_aggregation | Сводка результатов | 85–91 |
| report_build | Подготовка отчёта | 91–94 |
| archive | Сборка архива | 94–97 |
| llm_summary | Подготовка рекомендаций | 97–100 |

### Счётчики и результат нормы

Старое событие `compliance_progress` сохранено без изменения формы. Это счётчик планов,
не новый процентный progress; отдельного `partial_norms` в нём нет.

```json
{"type":"compliance_progress","content":{"total_norms":1,"completed_norms":1,"pending_norms":0,"passed_norms":0,"violated_norms":1,"unverifiable_norms":0,"unsupported_norms":0}}
```

Пример сериализуемого результата для порога этажности (пустые массивы evidence и
requirements здесь допустимы; реальные проверки обычно заполняют их):

```json
{"type":"compliance_result","content":{"restriction_id":"00000000-0000-4000-8000-000000000001","template":"object_attribute_threshold","template_version":1,"verification_status":"complete","compliance_status":"violated","status":"violated","status_reason":"Нарушено зон: 1 из 1.","check_type_en":"floors","check_type_ru":"этажность","archive_path":"violated/00000000-0000-4000-8000-000000000001","coverage":{"applicable_objects":1,"checked_objects":1,"unchecked_objects":0,"fill_rate":1.0},"summary":{"violated_objects":1,"passed_objects":0},"effective_requirements":{"layers":[],"attributes":[]},"resolved_requirements":[],"missing_requirements":[],"warnings":[],"evidence_schema_version":"1.0","source":{"document_name":"СП 42.13330.2016","clause_number":"7.1","extraction_text":"Этажность не более 5 этажей."},"evidence":[]}}
```

`status` — новый итог. Старые `verification_status` (`complete|partial|unverifiable|
not_applicable|unsupported`) и `compliance_status` (`passed|violated|unknown`) остаются.
Например, частичная проверка может иметь старое `compliance_status=passed`, а новый
`status=partial`. UI использует новый статус и не восстанавливает вердикт из текста.
В `compliance_result` нет `status_ru`: русские подписи находятся в геослое.
`violated_features` / `passed_features` исключены из SSE результата и сводки.

`requirement_resolution.content`: `restriction_id`, `effective_requirements`,
`resolved_requirements`, `missing_requirements`. `check_plan.content`: `restriction_id`,
`plan`, а фактически отправляемый payload также содержит `equivalent_sources`.
`source` результата также содержит исходный CheckPlan и `equivalent_sources`.
Evidence сохраняет измерение, порог, оператор, единицу, ссылки `object_ref`,
`generator_ref(s)`, `zone_ref(s)`, `operation`, `used_fields`, provenance, warnings,
`input_revision`; в зависимости от шаблона — счётчики соседей, площади либо demand.

### Сводка check

```json
{"type":"compliance_summary","content":{"request_id":"run-1","total_norms":1,"violated_norms":1,"passed_norms":0,"unverifiable_norms":0,"unsupported_norms":0,"not_applicable_norms":0,"partial_norms":0,"results":[],"counts":{"violated":1},"total_restrictions":1,"executed_norms":1,"formal_passed_norms":0,"duplicate_checks":0,"skipped_without_plan":0,"equivalent_sources":{},"scenario_id":772,"created_at":"2026-10-29T12:00:00+00:00","files":{"summary":"summary.md","archive":"compliance_772_20261029-1200.zip"}}}
```

В примере `results` сокращён до пустого массива для чтения: в реальном ответе здесь
описанный выше результат и дополнительные записи всех полученных норм, включая
незапущенные и эквивалентные. Поля `scope`, `territory`, `user_files` добавляются по контексту;
`query` также включён в фактически выдаваемую сводку и `summary.json`.
`total_norms` — исходные агрегаты уникальных результатов; `total_restrictions` — длина
полного списка. `counts` считает новые статусы без `equivalent_to`, нулевые ключи могут
отсутствовать. `executed_norms` считает записи с `archive_path`, а не все найденные нормы.
`compliance_progress.passed_norms` использует прежний compliance_status, тогда как
`compliance_summary.passed_norms` использует новый status. Поэтому частичный passed
в progress может не попасть в passed итоговой сводки. Поля `unchecked_norms` нет.

### Последовательность inventory

Общие территория, область, загрузка, валидация, дедупликация → `source_layers` →
`zones_building` с `tool_call` и `restriction_zone` → `cache` → `report_build` → `archive`
→ `restriction_inventory` → `llm_summary` → `progress=100` → один `chunk(done=true)`
→ слои норм → `summary.md` → ZIP. Нет Check*, выборки целей, `requirement_resolution`,
`compliance_result`, `compliance_progress` и `compliance_summary` этого прогона.
При восстановлении подготовленного inventory `restriction_zone` повторно не генерируется.

```json
{"type":"restriction_zone","content":{"restriction_id":"00000000-0000-4000-8000-000000000001","template":"distance_from_source","template_version":1,"status":"not_checked","status_ru":"не проверялось","build_status":"shown","zone_kind":"restriction","zone_count":1,"skipped_objects":0,"description":{},"missing_requirements":[],"source":{"document_name":"СП 42.13330.2016","clause_number":"7.1"}}}
```

```json
{"type":"restriction_inventory","content":{"request_id":"run-1","mode":"inventory","total_norms":1,"shown_norms":1,"restriction_zones":1,"required_zones":0,"no_objects_norms":0,"unverifiable_norms":0,"unsupported_norms":0,"zones":[],"duplicate_checks":0,"skipped_without_plan":0,"results":[{"restriction_id":"00000000-0000-4000-8000-000000000001","source":{},"template":"distance_from_source","template_version":1,"status":"not_checked","status_reason":"Перечень ограничений; проверка объектов не выполнялась","archive_path":null}],"counts":{"not_checked":1},"total_restrictions":1,"executed_norms":0,"formal_passed_norms":0,"scenario_id":772,"created_at":"2026-10-29T12:00:00+00:00","files":{"summary":"summary.md","archive":"compliance_772_20261029-1200.zip"}}}
```

`zones` в примере сокращён; в реальном ответе он содержит payload `restriction_zone`.
`shown_norms` и счётчики restriction/required относятся к внутреннему `shown`, а не
к выполнению нормы. `status=not_checked` заменяет только успешный `shown` в payload;
неуспешные `no_objects|unverifiable|unsupported` остаются статусами построения.
В `summary.results` все inventory-нормы имеют `not_checked`, `archive_path=null`.
Слои inventory содержат только зоны, без `object` и `violation`; папок норм в ZIP нет.

## Слой нормы и статусы

Один слой `Норма — СП 42.13330.2016, п. 7.1`, с суффиксом ` (2)` при совпадении
названий; без пункта, если его нет, и `Источник не указан`, если нет документа.
Слой может быть пустым, в том числе при формальном выполнении. Незапущенная норма
может не иметь слоя. Нет отдельных общих слоёв прошедших объектов.
Координаты — EPSG:4326. `GeometryCollection` разворачивается в отдельные Feature;
части одной зоны сохраняют одинаковый `zone_ref`.

### Общие свойства каждой Feature

| Поле | Тип и смысл |
|---|---|
| feature_kind | `zone`, `violation`, `object` |
| norm_id | UUID ограничения, ключ группировки |
| norm_status / norm_status_ru | итог нормы и русская подпись |
| norm_status_reason | строка причины либо `null` |
| check_type_en / check_type_ru | детерминированный тип проверки |
| norm_reference | сокращённая ссылка на документ/пункт |
| document_name / clause_number | исходные названия либо `null` |
| norm_text | `extraction_text` либо `null` |

`meta.group_by="norm_id"`. `meta.group_properties` перечисляет десять общих полей
от `norm_id` до `norm_text` (без `feature_kind`). Meta слоя содержит `norm_id`,
`norm_status`, `norm_status_ru`, `norm_status_reason`; для check агенты также добавляют
`check_type_en`, `check_type_ru`. Не требуйте фиксированного порядка ключей JSON.

В MCP `distance_from_source`, `distance_table`, `presence_within` дают тип
`distance/расстояние`; `accessibility_within` — `accessibility/доступность`,
`zonal_ratio` — `area_ratio/доля площади`, provision — `provision/обеспеченность`.
Атрибутные проверки уточняют тип: floors/storeys/number_of_floors/этажность →
`floors/этажность`; height/height_m/building_height/высота → `height/высота`;
area/area_m2/площадь → `area/площадь`; иначе `attribute_threshold/пороговое значение`.
Проверяется последняя часть dotted field. Fallback агентов без MCP-слоя для
`presence_within` использует `presence/наличие`; приоритет имеет фактическое поле слоя.

### Поля zone

| Поле | Тип и смысл |
|---|---|
| zone_ref | объект `{"id":"norm/<norm_id>/zone/0"}`, локальная ссылка |
| status / status_ru / status_reason | статус этой зоны, подпись, причина |
| zone_kind | check: `source_buffer`, `source_geometry`, `neighbor_buffer`, `accessibility_buffer`, `functional_zone`, `project_territory`; inventory сохраняет свойства построенной зоны |
| applies_to | список имён слоёв/сущностей, к которым применима зона |
| buffer_size | радиус в метрах либо `null` |
| operator / threshold / unit | условие, числовой порог и единица либо `null` |
| source_object_ref | ссылка исходного объекта либо `null` |
| objects_inside | число локально подходящих целей; в inventory `null` |
| objects_required | 0 для запрета, 1 для обязательного размещения, minimum_neighbors для соседства, иначе `null` |

Inventory дополнительно сохраняет исходные свойства `CreateRestrictionZones`:
`restriction_title`, `restriction_description`, `restriction_id`, `provenance`,
`template` и параметры нормы, если они были переданы. Список этих исходных
атрибутов открыт; не удаляйте незнакомые свойства при сохранении GeoJSON.

### Поля violation и object

`violation`: `object_ref` (объект ссылки), `object_name` (строка), `zone_ref`,
`violation_geometry_mode` (`intersection|object|zone`), `measured_value`,
`required_value`, `deficit` (числа либо `null`), `unit`, `explanation_ru`.
Объяснение детерминированное: например, `Этажность: 6 эт., требуется ≤ 5 эт., превышение 1 эт.`.
Для строгого `<` или `>` при равенстве дефицит равен нулю, но объяснение указывает
недопустимость равенства. Геометрия нарушения не обязательно Polygon: бывает Point/LineString.

`object`: сохраняет исходные properties и добавляет `object_ref`, `object_name`,
`type` (service_type, physical_object_type, object_type либо source_layer),
`zone_refs` (список ссылок на связанные зоны). Это связанные объекты, а не отдельный
вердикт «выполнено»: один объект может участвовать в нескольких нормах.
Обеспеченность через ObjectEffects дополнительно может добавлять объектам
измерение, требуемое значение, дефицит, `unit="share"`, объяснение и режим `zone`;
не предполагайте, что эти поля бывают только у `violation`.

### Пример слоя

Полный минимальный слой inventory; в check к тем же общим свойствам добавляются
поля проверки, приведённые выше. Идентификаторы в примере условные.

```json
{"type":"feature_collection","content":{"name":"Норма — СП 42.13330.2016, п. 7.1","feature_collection":{"type":"FeatureCollection","meta":{"group_by":"norm_id","group_properties":["norm_id","norm_status","norm_status_ru","norm_status_reason","check_type_en","check_type_ru","norm_reference","document_name","clause_number","norm_text"],"norm_id":"00000000-0000-4000-8000-000000000001","norm_status":"not_checked","norm_status_ru":"не проверялось","norm_status_reason":null},"features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[30,60],[30.01,60],[30.01,60.01],[30,60.01],[30,60]]]},"properties":{"feature_kind":"zone","norm_id":"00000000-0000-4000-8000-000000000001","norm_status":"not_checked","norm_status_ru":"не проверялось","norm_status_reason":null,"check_type_en":"distance","check_type_ru":"расстояние","norm_reference":"СП 42.13330.2016, п. 7.1","document_name":"СП 42.13330.2016","clause_number":"7.1","norm_text":null,"zone_ref":{"id":"norm/00000000-0000-4000-8000-000000000001/zone/0"},"status":"not_checked","status_ru":"не проверялось","status_reason":null,"zone_kind":"restriction","applies_to":["Театр"],"buffer_size":100,"operator":null,"threshold":null,"unit":null,"source_object_ref":null,"objects_inside":null,"objects_required":null}}]}}}
```

### Правила статусов и формальное выполнение

| status | *_ru | Значение |
|---|---|---|
| passed | выполнено | все применимые проверенные условия соблюдены |
| violated | не выполнено | есть нарушение хотя бы в одной зоне |
| partial | частично | нет установленного нарушения, но часть данных/объектов не проверена |
| not_checked | не проверялось | проверка не запускалась, в том числе inventory |

Для `violated` и `partial` зоны причина обязательна. Итог нормы: нарушение имеет
приоритет над неполнотой; иначе partial при непроверенных объектах, иначе passed.
`norm_status` может отличаться от `status` конкретной зоны. Для незапущенных норм
причина находится в result/summary, слоя может не быть.

`no_applicable_objects` — предупреждение полного пустого набора применимых объектов:
`verification_status=complete`, `compliance_status=passed`, новый `status=passed`,
coverage `0/0/0`, fill_rate `1`. Это **формальное выполнение**, а не доказательство
полноты проекта. В `summary.results` добавляется `formal_pass=true`, в сводку —
`formal_passed_norms`; в одиночном `compliance_result` отдельного поля `formal_pass` нет.
Сбой получения слоя не превращается в пустой полный набор и формальное выполнение.

### Шаблоны: зона и геометрия нарушения

| Шаблон | Зона | Условие нарушения | violation_geometry_mode |
|---|---|---|---|
| distance_from_source, matched | буфер источника или его геометрия | цель соответствует предикату зоны | intersection |
| distance_from_source, not_matched | тот же источник | требуемой цели в зоне нет | object для доступных целей вне зоны |
| distance_table | индивидуальный буфер по bands | как distance_from_source | intersection / object |
| presence_within | буфер проверяемого объекта | в любом обязательном слое меньше minimum_neighbors | object исходного объекта |
| accessibility_within | буфер по приближённому радиусу | недостаточно соседей | object исходного объекта |
| zonal_attribute_threshold | функциональная зона, обрезанная проектом | атрибут цели нарушает порог зоны | intersection |
| zonal_ratio | функциональная зона в проекте | процент площади не соответствует порогу | zone |
| object_attribute_threshold | территория проекта | атрибут объекта нарушает порог | object |
| service_provision | территория проекта | обеспеченность ниже порога ObjectEffects | zone |

Если для `not_matched` нет ни одной цели вообще, зона всё равно `violated`, но
Feature `violation` может отсутствовать: нечего отдавать как геометрию цели.
У presence/accessibility геометрия нарушения появляется, когда доступен исходный объект.
Не определяйте статус зоны по наличию красных Feature. Режимы задаёт реализация
шаблона; произвольного выбора режима геометрии из текста пользователем сейчас нет.
Буферные зоны check также обрезаются по территории проекта при наличии её слоя.

Пример свойств нарушения этажности **до добавления десяти общих полей** из таблицы
выше; фактическая Feature в слое содержит и эти свойства, и общие поля нормы:

```json
{"feature_kind":"violation","object_ref":{"id":"physical_object/1","namespace":"physical_object","entity_id":1,"geometry_id":null,"layer":"Жилой дом","name":"Дом"},"object_name":"Дом","zone_ref":{"id":"norm/00000000-0000-4000-8000-000000000001/zone/0"},"violation_geometry_mode":"object","measured_value":6.0,"required_value":5.0,"deficit":1.0,"unit":"floors","explanation_ru":"Этажность: 6 эт., требуется ≤ 5 эт., превышение 1 эт."}
```

Для той же цели `object` сохраняет исходные атрибуты (например, `floors: 6`),
`type` и список связей; общие поля нормы также повторяются:

```json
{"feature_kind":"object","physical_object_id":1,"name":"Дом","floors":6,"source_layer":"Жилой дом","object_ref":{"id":"physical_object/1","namespace":"physical_object","entity_id":1,"geometry_id":null,"layer":"Жилой дом","name":"Дом"},"object_name":"Дом","type":"Жилой дом","zone_refs":[{"id":"norm/00000000-0000-4000-8000-000000000001/zone/0"}]}
```

## GetServicesByGeometry и двухфазная выборка

Публичная MCP-сигнатура (зависимости авторизации и клиента не передаются пользователем):

```text
GetServicesByGeometry(
  scenario_id: int,
  services_names: list[str],
  zones_layers: list[str],
  layer_name: str,
  layers: dict[str, dict],
  buffer_m: float = 0,
  intersection_type: "intersects" | "within" = "intersects",
  physical_object_type_ids: list[int] | None = None
) -> dict[str, dict]
```

Область — точное объединение геометрий указанных слоёв; дополнительный буфер
строится в локальной UTM и возвращается в WGS84, без упрощения. Нужны непустой
`zones_layers`, существующие валидные геометрии, Polygon/MultiPolygon области,
конечный неотрицательный `buffer_m`, непустые имена сервисов и `layer_name`.
Пустой слой допустим: HTTP выборки объектов не вызывается. Имена нормализуются
`strip().capitalize()`, разрешаются в канонические service_type_id, повторения убираются.
`physical_object_type_ids` принимается, но пока не передаётся в Urban API и не фильтрует
результат: поддержки этого фильтра у by_geometry нет.

```json
{"scenario_id":772,"services_names":["Театр"],"zones_layers":["Норма — СП 42.13330.2016, п. 7.1"],"layer_name":"Цели — Театр","layers":{"Норма — СП 42.13330.2016, п. 7.1":{"type":"FeatureCollection","features":[]}},"buffer_m":0,"intersection_type":"intersects","physical_object_type_ids":null}
```

Пример формы непустого ответа (значение revision иллюстративное):

```json
{"Цели — Театр":{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Point","coordinates":[30.005,60.005]},"properties":{"service_id":17,"service_type_id":9,"service_type_name":"Театр","service_type":{"id":9,"name":"Театр"},"name":"Театр","capacity":100,"is_capacity_real":true,"object_geometry_id":25,"is_scenario_service":true,"is_scenario_geometry":true,"is_locked":false,"object_ref":{"id":"service/17/geometry/25","namespace":"service","entity_id":17,"geometry_id":25,"layer":"Цели — Театр","name":"Театр"}}}],"meta":{"calls_made":1,"mode":"single","component_count":1,"part_count":1,"intersection_type":"intersects","per_type_counts":{"9":1},"complete":true,"truncated":false,"revision":"scenario:772:services_by_geometry:<sha256-wkb>:9:intersects"}}}
```

Исходные properties Urban API сохраняются; `service_type_id`, `service_type_name`,
`service_type` добавляются при отсутствии, `object_ref` нормализуется общим helper.
`object_ref` содержит `id`, `namespace`, `entity_id`, `geometry_id`, `layer`, `name`.
ID для сервиса с геометрией: `service/17/geometry/25`, без object_geometry_id — `service/17`.
Название берётся из name, адреса либо имени слоя; безымянное служебное название API
заменяется читаемой подписью.
Дедупликация по `str(service_id)`; отсутствие service_id/geometry и GeometryCollection — ошибка.
`per_type_counts` — принятые уникальные ID по типам; `calls_made` включает HTTP-попытки
и отклонённые части, `part_count` — успешные ответы, `component_count` — исходные
полигональные компоненты. Revision хеширует область, типы и предикат, не снимок данных API.

### Один тип — одна общая выборка

1. `GetProjectTerritory` → `project_territory`.
2. Источники сервисов: по территории с максимальным радиусом всех норм этого типа,
   включая источники вне проекта. Имена слоёв `Источники — <тип>`.
   Физические объекты получают через `GetPhysicalObjects(centers_only=false)`,
   функциональные зоны — через `GetFunctionalZones`.
3. `CreateRestrictionZones` отдельно для каждой нормы; сохраняются стабильные ссылки слоёв.
4. Для каждого типа целей объединяются зоны только тех норм, которым он нужен;
   один MCP-вызов с `layer_name="Цели — <тип>"`. Если все нормы требуют `within`,
   запрос использует `within`; иначе `intersects`. Это уточнение реализации относительно
   спецификации, где для данной фазы всегда указан `intersects`.
5. Check* проверяет точные геометрии локально относительно каждой зоны.

Предикат выводится из `params.predicate`, а для zonal_attribute_threshold — из
`join_predicate` с fallback на predicate; только `within` преобразуется в `within`,
остальные варианты — в `intersects`. Локальная функция поддерживает также `contains`:
это `object_geometry.contains(zone_geometry)`. `within` означает целую геометрию внутри,
точка на границе не within; `intersects` учитывает касание границы. Нет проверки по центроиду/bbox.
Provision использует ObjectEffects, не делает общую выборку целей сервисов.

### Split, повторы и ограничения

Urban API вызывается через `POST /api/v1/scenarios/{scenario_id}/services/by_geometry/geojson`
с одним service_type_id и intersection_type в query и геометрией в теле.
413 и **только размерные** 422 вызывают lossless split: сначала компоненты, для
единственного полигона — пересечение с четырьмя квадрантами bbox, рекурсивно.
Иные 422 — ошибка валидации, не повод дробить геометрию. После split API запрашивается
с `intersects`; исходный предикат ещё раз применяется ко всей исходной области локально.
Части выполняются последовательно, типы — параллельно под общим семафором экземпляра.
Это не упрощение геометрии. При превышении глубины — ToolError.
Транзиентные сетевые сбои и поддерживаемые reset-by-peer повторяет JsonApiHandler
ограниченно (по умолчанию 3 попытки, backoff 0.5 × 2^(attempt−1) секунды).

## Архив и файлы

`compliance_<scenario_id|files>_<YYYYMMDD-HHMM>.zip`, ZIP_DEFLATED.
Время берётся из `created_at` UTC, а не из локального времени браузера.

```text
README.md
summary.md
summary.json
manifest.json
layers.geojson
violated/<restriction_uuid>/
  report.md
  result.json
  result.geojson
  layers/<role>__<entity>.geojson
partial/<restriction_uuid>/...  # только реально запущенные проверки
passed/<restriction_uuid>/...
```

Неподдержанные/незапущенные нормы представлены в summary.json, без папки.
Формальное выполнение считается complete и получает папку даже без Check*.
Эквивалентные нормы ссылаются на папку представителя; `equivalent_to` указывает его UUID.
В `result.json.equivalent_sources` и report.md перечислены источники группы.

### Содержимое и схемы

| Файл | Содержание |
|---|---|
| README.md | краткая русская инструкция: summary, группировка norm_id, feature_kind, окраска status, каталоги статусов, эквивалентные нормы, SHA-256 и исключение самого manifest |
| summary.md | сценарий/время/область/территориальные документы; счётчики; формальные выполнения; таблицы «Не выполнено», «Частично», «Выполнено» (пункт, тип, нарушений, причина, ссылка); «Не запущено» с агрегатами причин; сведения о файлах пользователя; «Где смотреть подробно» |
| summary.json | полный JSON сводки из SSE, включая results, counts, query, scenario_id, created_at, files; для inventory — сводка inventory с results |
| manifest.json | request_id, scenario_id, created_at, gmart_version, templates, parameters, files |
| layers.geojson | единый FeatureCollection всех слоёв норм, meta.group_by и объединённый group_properties; общие атрибуты остаются на Feature |
| report.md | документ/статус, исходный текст, тип ru/en, шаблон@версия, параметры/предикат, входные слои/источники/число объектов, fill-rate, разрешение атрибутов, результат и причина, детали применимости/населения/доступности, таблица нарушений до 50 строк, эквивалентные нормы, список файлов |
| result.json | результат нормы без violated_features/passed_features плюс check_plan, equivalent_sources, effects_tool_calls, timings_ms, input_revision, tool_calls, input_layers |
| result.geojson | полный слой нормы с zone/violation/object; пустой FeatureCollection допустим |
| layers/*.geojson | входные слои и подготовленные зоны, как использованы проверкой; role/entity в имени очищаются от недопустимых символов |

`result.json.input_layers[]`: `name`, `path` (относительно папки нормы), `count`,
`origin` (точное имя инструмента, `user_files` либо `источник не указан`).
`timings_ms` — словарь длительностей этапов. `tool_calls[]` имеет форму
`{"function":{"name":"GetServicesByGeometry","arguments":{...}}}`; сырые layers
не нужны в записанных аргументах: они находятся в отдельных файлах и восстанавливаются по ссылкам.
`input_revision` берётся из записанного Check*-вызова, может быть `null`.

`manifest.templates` — отсортированный список `template@version` реально архивированных
проверок. `parameters` содержит `scope`, `territory`, `user_files`, `query` (возможно null).
`files` — объект `{ "relative/path": "<sha256 исходных байтов>" }`, не массив.
Сам manifest, ZIP и внутренние checkpoints в этот словарь не включены. В ZIP есть manifest.
У inventory templates пуст, нет каталогов passed/partial/violated.

### /files, TTL и удаление кэша

Внутренний файл события:

```json
{"type":"file","content":{"name":"compliance_summary","title":"Сводный отчёт о проверке норм","role":"result","url":"/files/compliance_summary/<file-id>","download_url":"/files/compliance_summary/<file-id>?download=1","filename":"summary.md","mime_type":"text/markdown","source_service":"gmart"}}
```

На HTTP-проводе:

```text
event: file
data: {"name":"compliance_archive","title":"Архив проверки норм","role":"result","url":"/files/compliance_archive/<file-id>","download_url":"/files/compliance_archive/<file-id>?download=1","filename":"compliance_772_20261029-1200.zip","mime_type":"application/zip","source_service":"gmart"}
```

`url` открывает inline, `download_url` отдаёт attachment. Пути относительны к корню
agents (учитывайте proxy prefix). `OwnedFilesApp` требует Bearer, доступ только владельцу
по проверенному sub; без токена 401, чужой/истёкший/неизвестный файл 404.
Для загрузки — fetch с Authorization → Blob → object URL → скачивание; обычный
переход по ссылке без Bearer недостаточен. Отдельно выданные файлы живут 3600 секунд,
удаляются лениво и периодической очисткой раз в 300 секунд. Хранилище локально контейнеру.

Кэш прогона: `<GMART_CACHE_DIR>/cache/compliance/<request_id>`; без настройки —
`<system-temp>/gmart/cache/compliance/<request_id>`. Он не раздаётся напрямую через /files.
`LayerCache.purge_expired` удаляет каталоги по mtime; по умолчанию compliance TTL
4200 секунд (1 час плюс 10 минут), переопределяется `GMART_COMPLIANCE_CACHE_TTL_SECONDS`.
У загрузок upload_layers собственный TTL 24 часа. Истечение кэша и ссылки — разные события.
Пересоздание контейнера теряет локальные файлы; несколько реплик требуют общей файловой
системы/маршрутизации к нужной реплике, если нужно повторное чтение.

## LLM, checkpoint и переподключение

Финальный LLM-комментарий использует запрос и компактную детерминированную сводку,
без сырых планов, геометрии и evidence. Максимум 50 уникальных норм, сначала violated,
затем partial, затем passed; `omitted_norms` и `note="и ещё N"` сообщают об остатке.
Незапущенные передаются агрегатами причин. Это лимит контекста рекомендации,
**не лимит числа выполняемых проверок**. В inventory список проверенных норм пуст.
Модель вызывается с `think=false`, temperature 0.2 и timeout 60 секунд.
Пустой ответ, ошибка или timeout → `_compliance_summary_text` либо `_inventory_text`.
При `total_norms=0` модель не вызывается. Нет отдельного LLM-комментария каждой геометрии.

Redis хранит буфер событий и checkpoint; диск — `inputs.json` (снимки общих слоёв,
зон и вызовов) либо `inventory.json`, `.executions/<uuid>.json`, `archive.json`.
Результат каждой нормы пишется на диск сразу, `compliance_executions` в Redis —
пакетами по 25 и остатком. Итоговые checkpoints: requirements resolution, template
execution, verdict aggregation / restriction_inventory, `compliance_final_text`,
`compliance_progress_percent`, `compliance_completed`.
Повтор с тем же request_id сначала читает события; подготовленные входы, завершённые
проверки, текст и ZIP переиспользуются, пока доступны кэш и checkpoints. Завершённый
pipeline не запускается заново. Это восстановление прогона, не долговременный архив.
После истечения TTL нельзя обещать восстановление без вычисления.

## A2A, файлы пользователя и ChatStorage

### Артефакты A2A

`POST /compliance/a2a` исполняет тот же pipeline без записи в ChatStorage.

| artifactId | Данные |
|---|---|
| compliance-answer | текст, части append=true |
| compliance-layer-N | GeoJSON слоя нормы, application/vnd.geo+json |
| compliance-summary | compliance_summary, application/json |
| compliance-restriction_inventory | restriction_inventory, application/json |
| compliance-file | дескриптор summary.md, application/json |
| compliance-archive | дескриптор ZIP (не байты ZIP), application/json |

`status` превращается в status-update WAITING; clarification — INPUT_REQUIRED;
error/pipeline_suspended — FAILED. Итог task COMPLETED после потока. `progress`,
`compliance_progress`, tool_call, check_plan и отдельные results артефактами не выдаются.
Для файлов по ссылке действуют те же Bearer/owner/TTL.

### Режим user-files

`upload_ids` без сценария — standalone; со сценарием — overlay. Роли `project`,
`context`, `border`; сопоставление групп сущностям может потребовать clarification
`kind="entity_mapping"`. `user_files` сообщает mode, upload_ids, layers, mapping, pending.
Граница берётся из border, иначе convex hull проектных геометрий. В overlay объекты
сценария той же категории внутри границы заменяются проектными; окружение участвует
как источник/сосед, проверяемыми объектами служит проект. Это `UserLayerSource`,
не обещание двухфазной сценарной выборки через общий ScenarioComplianceSource.
Формат слоёв/статусов/сводки/ZIP общий; имя ZIP без сценария содержит `files`.
Нормы без нужных объектов попадают в `user_files.not_checked`; service_provision
не считает обеспеченность файлов, сообщает `user_files:scenario_only_template`.
Inventory с файлами не строится: сервис поясняет, что перечень работает по сценарию.
Вызовы над пользовательскими геометриями в историю не записываются.

### Повтор ChatStorage: phase 4 pending

История compliance сохраняет tool_call и обычный текст, включая финальную рекомендацию;
объёмные result/summary/progress не сохраняются как структурированные части сообщения.
Дескрипторы файлов сохраняются как `kind="file"`, без `role` и `download_url`:
эта ветка сериализации выполняется до фильтра text_only. Ссылки живут только час;
наличие части file в истории не продлевает доступность файла.
Сценарная цепочка содержит `GetProjectTerritory → GetServicesByGeometry →
CreateRestrictionZones → GetServicesByGeometry → Check*`.
`GetServicesByGeometry` предоставляет слой по `layer_name` и читает ссылки `zones_layers`
из накопленного `layers`. В phase 4 ChatStorage ещё должен добавить инструмент в
`_LAYER_READING_TOOLS`, `_LAYER_TOOLS`, `_LAYER_REF_ARGUMENTS`, `_provided_refs` и
проверить восстановление **norm_layer** Check*, а не старых корзин passed/violated.
Поддержка этой цепочки в ChatStorage **pending**, не гарантия текущего replay.
Replay на изменившемся сценарии — новый расчёт, не исторический снимок; ZIP хранит снимок входов.

## Ошибки, переменные окружения и TODO

Ошибки входных KeyError/TypeError/ValueError публичный MCP-интерфейс преобразует в
ToolError `Некорректные данные поиска сервисов: …`; неизвестные типы, некорректный
ответ API и исчерпанный split также ToolError. Ошибки API не превращаются в полный
пустой слой. В сценарном пайплайне retrieval-сбои помечают затронутые нормы
`not_checked`, причины: `layer:project_territory:retrieval_failed`,
`source_layers:retrieval_failed`, `objects_by_geometry:retrieval_failed`,
`zones_building:retrieval_failed`; исходное чувствительное исключение остаётся в логах.
Остальные нормы продолжают работу. TokenExpiredError проходит наружу: pipeline
уведомляет `token_expired` / `pipeline_suspended`, продолжение через обновление токена
и прежний request_id (см. frontend-service). Ошибки REST используют общий mapper
AgentsInputException 400, AgentsNotFound 404, DownstreamServiceError 502;
после начала SSE ошибка передаётся событием, а не сменой уже отправленного HTTP-кода.

| Переменная | По умолчанию / назначение |
|---|---|
| COMPLIANCE_BY_GEOMETRY_CONCURRENCY | 8; агенты ограничивают MCP-запросы, MCP — HTTP по типам; настраивается в обоих приложениях |
| COMPLIANCE_BY_GEOMETRY_MAX_SPLIT_DEPTH | 6; положительная глубина квадрантного деления, IDU MCP |
| GMART_CACHE_DIR | system-temp/gmart; базовый каталог, helper добавляет cache |
| GMART_COMPLIANCE_CACHE_TTL_SECONDS | 4200; TTL каталога прогона по mtime |
| IDU_MCP_SERVER | адрес IDU MCP для геометрии/Urban API |
| URBAN_API_URL | Urban API в IDU MCP |
| NORM_GRAPH_MCP_SERVER | источник сохранённых исполнимых норм |
| OBJECTS_EFFECTS_MCP_SERVER | ObjectEffects для service_provision |
| OLLAMA_API_URL | LLM итоговой рекомендации и разбора области |
| REDIS_URL | состояние, буфер SSE и checkpoints |
| CHAT_STORAGE | история сценарных вызовов и текста |

TODO из §8 спецификации: изохроны ObjectNat вместо прямолинейного буфера
accessibility_within; physical_object_type_ids фильтр и by_geometry физических
объектов после поддержки Urban API; spatial_predicate в CheckPlan schema 1.1
после согласования NormGraph (`extra=forbid` сейчас); подробный LLM-комментарий
к нарушению. Дополнительно pending phase 4: replay ChatStorage.

### Отличия реализации от спецификации

- Общая выборка целей использует within, если все связанные нормы требуют within;
  спецификация предписывает intersects для этой фазы.
- Split поддерживает квадранты единственной компоненты, а 422 дробит только при
  признаках превышения размера; обычная ошибка 422 не повторяется дроблением.
- `presence_within` в MCP имеет check_type distance, а не отдельный presence;
  слой агентов сохраняет тип MCP, fallback агентов имеет presence.
- Inventory архивирует общий слой и сводки без папок отдельных норм; выдаёт
  restriction_inventory вместо compliance_summary. В нём нет объектов целей.
- Не каждая violated-зона имеет violation Feature (нет геометрии отсутствующей цели).
  Режим геометрии фиксирован шаблоном, не свободно выбирается из формулировки.
- События file условны при недоступном файловом хранилище; result.status_ru отсутствует,
  русские подписи есть в GeoJSON. Пустой слой формально выполненной нормы допустим.
