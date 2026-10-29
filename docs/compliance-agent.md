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
ссылка на `summary.md` — `compliance-file`, на ZIP — `compliance-archive`.

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

## Проверка файлов пользователя

Вместо объектов сценария можно проверить собственный проект пользователя —
слои, выгруженные из генератора застройки или ГИС. Агент сам подбирает нормы по
загруженным объектам и запросу, запускает те же детерминированные шаблоны и
отдаёт результат в общем формате: `compliance_result`, слои норм со статусами,
`summary.md` и ZIP. Подробнее — [контракт](compliance-status-contract.md) и
[руководство](compliance-status-guide.md).

### Загрузка

`POST /compliance/uploads` (multipart, поле `files`, можно несколько) → 201 и
`upload_id`. Поддерживаются GeoJSON (`.geojson`, `.json`; BOM и координаты Z
допустимы), GeoPackage (`.gpkg`, каждый слой отдельно), GML, KML и MIF.
ZIP и 7z могут содержать эти форматы, Shapefile (`.shp` + `.shx` + `.dbf`
[+ `.prj`, `.cpg`]), MapInfo TAB (`.tab` + `.dat` + `.map` + `.id`) и MIF/MID.
Одиночные TAB и Shapefile нужно загрузить ZIP-архивом с сопутствующими файлами;
MIF без MID читается, если драйвер может открыть геометрию без атрибутов.
Поддерживается один уровень вложенных ZIP/7z с общим бюджетом лимитов.
Документы, растры и другие невекторные члены пропускаются до проверки лимитов
распаковки; ответ содержит счётчик по расширениям (`skipped_members:*`).
Имена файлов ZIP без UTF-8-флага читаются как cp866; кодировка DBF выбирается
по `.cpg`, затем по строгому UTF-8 в записях, LDID и, наконец, cp1251.

Все слои переводятся в EPSG:4326. Параметр формы `crs` принимает EPSG-код,
PROJ-строку, WKT или ключ реестра: `msk76_zone1`, `msk47_zone1`, `msk47_zone2`,
`msk47_zone3`, `epsg3857`. Он применяется к engineering/Nonearth и к слоям без
СК с координатами вне диапазона долготы и широты; предупреждение —
`crs_from_request:<ключ>` (для EPSG/PROJ/WKT — `custom`). Заявленная в файле СК
сохраняется. GeoJSON без `crs` и Shapefile без `.prj` с координатами долготы и
широты остаются на месте, даже когда для других слоёв той же загрузки передан
`crs`. Engineering/Nonearth без параметра `crs` отклоняется, даже если числа
похожи на долготу и широту.

В пределах одного файла/архива, включая вложенный архив, слой без СК с
проекционными координатами сначала получает единственную заявленную
проекционную СК пригодных соседних слоёв, если его границы находятся внутри
общих границ этих слоёв или не дальше 50 км от них. Предупреждение —
`crs_from_sibling:<ключ>` (или `custom`). СК соседей учитывается после
восстановления датума; параметр `crs` имеет приоритет. Engineering/Nonearth
и координаты долготы/широты под это правило не попадают.

Без подходящей СК соседей и явного `crs` проекционные координаты распознаются как
МСК-47 (по имени или однозначному диапазону) либо EPSG:3857 (по имени или
попаданию результата в Россию); предупреждение — `crs_missing_assumed:<ключ>`.
В остальных случаях файл отклоняется (400) с просьбой указать `crs`.
Пересечение диапазонов нескольких зон (в частности, МСК-47 зона 1 и МСК-76
зона 1) считается неоднозначным: требуется имя СК, пригодные соседи или `crs`.
МСК-76 не угадывается по координатам: используйте `msk76_zone1`.
Для ESRI `.prj` МСК-76 с точным совпадением параметров проекции и эллипсоида
восстанавливается преобразование датума из реестра
(`crs_datum_completed:msk76_zone1`); уже заданное преобразование сохраняется.
МСК-48 в реестр пока не включена: требуется явная подтверждённая СК.
Файлы Гаусса–Крюгера СК-42 / ГСК-2011 без СК могут быть неотличимы от
EPSG:3857 — загружайте их с `crs` или `.prj`.

Лимиты задаются окружением agents (МиБ = 1024 × 1024 байт):

| Переменная | По умолчанию | Что ограничивает |
| --- | ---: | --- |
| `COMPLIANCE_MAX_UPLOAD_MB` | 50 | размер запроса загрузки |
| `COMPLIANCE_MAX_ARCHIVE_MEMBERS` | 2000 | число векторных членов архивов |
| `COMPLIANCE_MAX_UNPACKED_MB` | 200 | распакованные векторные данные |
| `COMPLIANCE_MAX_LAYERS` | 500 | число слоёв в загрузке |

Вложенные архивы расходуют общий бюджет. Лимит объектов остаётся 200 000.
Переменные проброшены через compose и `vars.*` workflow деплоя; для больших
архивов можно, например, задать `COMPLIANCE_MAX_UPLOAD_MB=300`.
RAR не поддерживается — перепакуйте в ZIP/7z. MapInfo NativeX
(`.tda/.tin/.tma`), растровые и seamless TAB не читаются как векторные слои;
для них возвращается предупреждение, если в архиве есть пригодные слои.
Неполные TAB/Shapefile пропускаются, если остаются пригодные слои;
предупреждение — `incomplete_layer_skipped:<слой>:<недостающие расширения>`.
Если пригодных слоёв нет, сохраняется ошибка о недостающих файлах.
Таблицы без активной геометрии (в том числе TAB, DBF, GeoPackage и GML)
пропускаются с `no_geometry_layer_skipped:<слой>` и не участвуют в выборе СК.

Загрузка хранится 24 часа и доступна только автору (`sub` токена); чужая загрузка
отвечает 404, как несуществующая. В Redis (`pipeline:{upload_id}:compliance_upload`)
лежит только запись о загрузке: роли, ответы о сопоставлении и сводка каждого
слоя с именем его файла. Сами слои (GeoJSON в EPSG:4326) — файлы в кэше
приложения на диске контейнера (`GMART_CACHE_DIR`, по умолчанию системный temp;
в docker compose — том `gmart_cache`, поэтому загрузки переживают деплой).
Загрузка, чьи файлы кэша пропали (контейнер пересоздан без тома), отвечает 404,
как истёкшая.

Каждый файл, член архива и слой GeoPackage — отдельный слой с ролью:

| Роль | Смысл | Как определяется автоматически |
| --- | --- | --- |
| `project` | объекты проекта; только они проверяются | по умолчанию |
| `context` | окружение: источники и соседи дистанционных норм, сами не проверяются | все объекты имеют CLASSID с префиксом 701, 702, 603, 604, 606, 609, 704 или 706; либо имя содержит `surrounding`, `osm`, `context`, `existing`, «контекст», «окружен», «сущ» (например, «сущпол по ППТ») |
| `border` | граница проекта | все объекты имеют CLASSID с префиксом 6010; имя начинается с `gr_dpt`; все объекты `type=border` или имя с `border`, `boundary`, «границ» |

Роль можно задать при загрузке для всех слоёв (`role` в форме) или по слою:
`PATCH /compliance/uploads/{upload_id}` с `{"roles": {"<слой>": "context"}}`.
`GET /compliance/uploads/{upload_id}` возвращает слои, роли и число объектов,
`DELETE` удаляет загрузку.

### Запуск

`GET|POST /compliance/check/stream` с `upload_ids` (в GET — повторяющийся
query-параметр). `scenario_id` необязателен, если есть `upload_ids`:

- **без сценария** документы, действующие на территории проекта, ищутся по
  границе из файлов (при её отсутствии — по выпуклой оболочке объектов проекта):
  IDU MCP `GetTerritoriesByGeometry` спускается от страны по
  `intersecting_territories` Urban API до самых глубоких территорий, IDU_DVD
  `list_documents(territory_ids)` отдаёт документы найденных территорий, их
  предков и потомков. При полном спуске (`complete=True`) документы строго ниже
  найденных листьев исключаются: ни один потомок листа не пересекает границу.
  При неполном спуске (`complete=False`, исчерпан бюджет запросов) ничего не
  исключается. Если при полном спуске найдена только корневая территория,
  ответ сообщает, что граница проекта находится вне регионов, загруженных
  в Urban API, и применимы только документы корневого уровня. Документы без
  территориальной привязки при этом не учитываются;
- **со сценарием** документы определяются по сценарию, как обычно, а слои файлов
  накладываются на него: объекты сценария той же сущности **внутри границы**
  проекта заменяются объектами файлов, снаружи остаются контекстом. Сущности,
  которых в файлах нет, берутся из сценария.

Оркестратор принимает `compliance_upload_ids` (тело `POST /orchestrator/route/stream`
или query `GET`): при загрузках compliance доступен и без `scenario_id`, а
планировщик направляет просьбы проверить проект в шаг compliance. A2A принимает
`upload_ids` или GeoJSON прямо в DataPart: `files: [{name, role, geojson}]` —
такие слои сохраняются как загрузки вызывающего пользователя.

Для файлов доступен только режим проверки: вопрос о перечне ограничений
(`inventory`) с файлами отвечает, что перечень строится по сценарию.

### Сопоставление объектов с сущностями норм

Нормы называют объекты типами Urban API («Жилой дом», «Школа», «Детская
площадка»), файлы — тегами. Объекты с одинаковыми классифицирующими тегами
(`type`, `building`, `amenity`, `leisure`, `landuse`, `natural`, `highway`, `shop`,
`man_made`, `healthcare`, `blockType`) образуют группу, и группа сопоставляется
целиком (`services/compilance/compliance_entity_mapping.py`):

Коды `CLASSID` / `CLASS_ID` читаются без учёта регистра; числовое окончание `.0`
удаляется. К группе добавляются код и название класса ФГИС ТП, а для слоёв
схемы ГИСОГД — `layer_kind`. Справочник на 463 кода хранится в
`services/compilance/data/fgistp_classifier.json`: публичная выгрузка
классификатора Приказа Минэкономразвития № 10 от 09.01.2018; названия и группы
требуют сверки с официальным приказом, сокращённые группы сохранены как даны.

До OSM-правил применяются правила по коду: жилые зоны `7010101xx` →
`residential`, общественно-деловые `701010200`/`7010103xx` → `business`,
производственные `701010401–403` → `industrial`, инженерная и транспортная
инфраструктура `701010404–405` → `transport` с уверенностью 0.7, рекреационные
`7010106xx` → `recreation`, специальные `7010107xx`/`701010800` → `special`.
Историческая зона `701010303` также имеет уверенность 0.7; если такое
сопоставление нужно норме, агент просит подтверждение. Сельскохозяйственные
`7010105xx` пока не сопоставляются: тип `agriculture` в текущих правилах
интеграции не используется. `701010900` и `701011000` сущностями не являются.
Объекты `602010101`, `602010102`, `602010401`, `602010402`, `602010408`
однозначно означают «Детский сад», «Школа», «Больница», «Поликлиника», «Аптека».
Остальные `602…` передаются LLM с названием класса в примере.

Границы `601…`, ЗОУИТ `603…`, ОКН `604…`, зоны риска ЧС `606…`,
`609…`, категории земель `702…`, `704…` и `706…` распознаются, но новыми
сущностями норм пока не становятся. В сводке сохраняется причина
`not_supported_entity:<группа>`, в отчёте — «распознано, нормами пока не
используется». Это относится и к территориальным зонам ПЗЗ, красным линиям и
слоям ГИСОГД `redline`, `formedland`, `constructionzoneborders`,
`elementplanningstructure`, `publicterritoryborders`, `indentline`, `easement`
(включая варианты `_line`/`_poly`): причина `not_supported_entity:gisogd`.
`gr_dpt` используется как граница проекта. Функциональные зоны определяет
только таблица; LLM функциональные зоны не предлагает.

1. **Таблица правил** — однозначные случаи: `building=residential|apartments|…` →
   «Жилой дом», прочие `building` → «Нежилое здание», `amenity|building=school` →
   «Школа», `kindergarten` → «Детский сад», `type=playground` → «Детская площадка»,
   `type=sport`/`leisure=pitch` → «Спортивная площадка», `highway` → «Местная
   дорога», кварталы генератора `type=block` → функциональные зоны по `blockType`
   (`residential`, `special`, `square` → `recreation`, `surface_parking` →
   `transport`; `school`/`police` → `business` с уверенностью 0.6) и т. д.
   `type=border`, `entry_point` и пустые кварталы сущностями не являются.
2. **LLM** классифицирует остальные группы, выбирая только из типов Urban API —
   сущностей отобранных норм (канонические имена каталога) и сущностей таблицы,
   — и возвращает уверенность 0..1. Имя вне списка не принимается. Функциональных
   зон среди кандидатов LLM нет: зоны получаются только из кварталов генератора по
   таблице, иначе площадка или здание превратились бы в зону, действующую на всё
   внутри неё. Ответы LLM сохраняются в загрузке (`classified`, ключ — группа и
   набор кандидатов), поэтому повторный запуск — в том числе после ответа
   пользователя — сопоставляет те же группы так же и не порождает новых вопросов.
3. Предложение с уверенностью ниже **0.8** задаётся пользователю событием
   `clarification` с `kind: "entity_mapping"` и `options` `[{number, label, value,
   entity_type, confidence, count, layers}]` только когда объекты группы нужны
   какой-то норме: объекты проекта — если их сущность есть среди отобранных норм,
   объекты окружения — если какая-то норма читает их сущность как источник, соседа
   или зону (окружение не бывает проверяемой целью). Остальные сомнения не
   задаются: такие группы просто не используются.

Ответ — следующее сообщение того же диалога (`chat_id`, чат оркестратора или
`contextId` A2A); `upload_ids` и `scenario_id` в нём не нужны — без них сообщение
принимается как ответ, а если вопроса в диалоге нет, агент отвечает, что проверять
нечего. «да» принимает всё, «нет» исключает спорные группы из проверки,
«2 — Спортивная площадка; 3 — нет» правит пункты (неупомянутые принимаются);
остальное разбирает LLM, новый вопрос сбрасывает ожидание. Подтверждения
сохраняются в самой загрузке (в пределах её 24 часов), и ответ продолжает
**исходный** запрос с той же областью проверки. Ожидание хранится в
`pipeline:{ключ диалога}:compliance_mapping` и checkpoint `compliance_mapping`.

Атрибуты объектов приводятся к полям Urban API, которые читают планы:
`building.floors`/`floors` из `levels`, `building:levels` или числа значений
`levelHeights`; `building.height`/`height_m` из `height` или суммы `levelHeights`;
`building.living_area` из `livingSpace`, `building.footprint_area` из `groundSpace`,
`population`, `capacity`. Идентификатор объекта — `user/<слой>/<id>`, имя —
«<сущность> — <слой>, №<id>».

### Какие нормы проверяются

Нормы отбираются как обычно — территория, темы и документы запроса, исполнимые
планы. Затем `UserLayerSource` (`compliance_user_layers.py`) решает, на каких из них
хватает данных:

- проверяемые роли шаблона (цели `distance_*`, `objects_layer`, зоны
  `zonal_ratio`) получают **только объекты проекта**; норма, у которой в проекте
  нет ни одного объекта целевой сущности, не запускается;
- остальные роли (источники, соседи, зоны) получают объекты проекта и контекста
  (и сценария при наложении); без сценария норма, для которой таких объектов нет
  ни в проекте, ни в окружении, не запускается;
- `service_provision` считается ObjectEffects по сценарию Urban API и для файлов не
  запускается;
- требование только точечной геометрии получает представительную точку контура.

Незапущенные нормы попадают в `compliance_summary.user_files.not_checked` (с
причинами), в итоговый текст («Не проверено из-за отсутствия нужных объектов в
файлах») и в раздел отчёта «Не проверено: нет данных в файлах». Событие
`user_files` (до проверки) и поле `compliance_summary.user_files` содержат слои,
роли, сопоставление групп (`source`: `table` | `llm` | `user` | `none`) и число
объектов по сущностям.

Вызовы шаблонов над слоями файлов не записываются в ChatStorage: файлы там не
хранятся, и повтор расчёта при открытии чата невозможен. Слои нарушений и отчёт
доступны в потоке и по ссылке на отчёт.

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
- `GetTerritoriesByGeometry(geometry)` — `{"territories": [{territory_id, name,
  level, territory_type, parent_id, parent_name}], "complete": bool}`: самые
  глубокие территории Urban API, в которых лежит геометрия (граница проекта из
  файлов пользователя).

Атрибуты зоны: `restriction_title`, `restriction_description` (текст нормы),
`restriction_id`, `zone_kind` (`restriction` | `required`), `applies_to`
(объекты, к которым относится норма), `provenance` (документ и пункт), для
буферов — `buffer_size`, для зональных норм — `operator`, `threshold`, `unit`.

Inventory выдаёт `status` и `progress`, `restriction_zone` без геометрии, затем
`restriction_inventory`, финальный `chunk`, слои «Норма — …» со статусом
`not_checked`, `summary.md` и последним ZIP. Нет Check*, `compliance_result`
и `compliance_summary`. Диск хранит `inventory.json`, завершённый ZIP переиспользуется.
A2A выдаёт `compliance-restriction_inventory`, `compliance-layer-N`,
`compliance-file` и `compliance-archive`. [Последовательность и поля](compliance-status-contract.md#последовательность-inventory).

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

Контракт статусов и архива: [compliance-status-contract.md](compliance-status-contract.md).

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
| `progress` | stage, stage_ru, percent, current, total, label, bar |
| `compliance_progress` | прежние счётчики выполненных и ожидающих норм |
| `file` | summary.md, затем ZIP последним |

После каждого этапа данные сохраняются в Redis. При повторном подключении с тем же
`request_id` сервис сначала отдаёт сохранённый буфер событий; завершённый pipeline
не запускает вычисления повторно.

После итогового LLM-текста приходят `feature_collection` «Норма — …» для доступных
результатов, включая выполненные нормы. Внутри — `zone`, `violation`, `object`;
зоны окрашиваются по status, общие атрибуты группируются по norm_id.
Затем `file` summary.md и последним ZIP. [Карта и архив](compliance-status-guide.md).

## ChatStorage и повтор расчёта

В историю записываются вызовы инструментов (`tool_call`), текстовый ответ
(`text`), включая вопросы для уточнения, и дескрипторы `file` без role/download_url.
Статусы, `check_plan`,
`requirement_resolution`, `compliance_result` и `compliance_summary` не включаются
в сообщение MongoDB, чтобы объёмные результаты проверки не превышали лимит BSON.
Полные результаты остаются доступны в SSE-потоке и Redis-журнале текущего запроса.

Новая сценарная цепочка включает GetProjectTerritory, GetServicesByGeometry,
CreateRestrictionZones и Check*. GetServicesByGeometry предоставляет layer_name
и читает zones_layers из layers. ChatStorage повторяет эту цепочку и для Check*
показывает norm_layer.

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


### Слои и итоговая рекомендация

Слой «Норма — документ, п. пункт» содержит feature_kind=zone|violation|object.
Группировка по norm_id, цвет зоны по status; norm_status описывает норму в целом.
Слои есть и для выполненных норм, при формальном выполнении слой может быть пустым.
Названия без UUID, при совпадении добавляется числовой суффикс.
Геометрия исключена из compliance_result и compliance_summary.results.

LLM формирует рекомендации по компактной сводке максимум 50 норм, без сырых
геометрий и evidence; timeout 60 секунд. Ошибка или пустой ответ включают
детерминированный fallback. Это лимит контекста рекомендации, не числа проверок.
Финальный текст предшествует слоям. [Поля и примеры](compliance-status-contract.md).

### Equivalent checks

Before execution, plans with identical template versions, parameters, layer roles
and data requirements are grouped. Entity names are resolved through the Urban API
catalogue first. Different thresholds, directions, geometry constraints or attribute
candidate order are not merged. If catalogue resolution fails or is ambiguous, checks
remain separate and the executor reports the missing requirement.

One `check_plan`, result and available per-norm `feature_collection` is emitted per
group. The result's `source.equivalent_sources`
and the summary's `equivalent_sources` preserve all source references. The summary
includes `duplicate_checks`; its existing norm counts refer to executed unique checks.
The final text names equivalent sources. Stored NormGraph norms are not deleted.

### Сводка и архив

summary.md и compliance_<scenario_id|files>_<YYYYMMDD-HHMM>.zip заменяют старый
одиночный отчёт. Архив содержит summary.json со всеми нормами, manifest.json с
SHA-256, общий layers.geojson; для запущенных проверок — папки violated/partial/passed
с report.md, result.json, result.geojson и входными layers/. Эквивалентные нормы
ссылаются на папку представителя. Inventory содержит общие файлы без папок проверок.

/files требует Bearer и доступен только владельцу один час. Ссылки относительны
к корню agents; используйте fetch с Authorization. Чужой/истёкший/неизвестный файл —
404. Ошибка сохранения файла может оставить результат без ссылок. Кэш прогона
очищается отдельно по mtime: 4200 секунд, GMART_COMPLIANCE_CACHE_TTL_SECONDS.

### Получение геометрии

GetServicesByGeometry получает источники по территории с максимальным радиусом
норм, затем цели по объединению нужных зон для каждого типа сервиса. Физические
объекты — GetPhysicalObjects(centers_only=false), функциональные зоны — GetFunctionalZones.
Provision использует ObjectEffects, пользовательские файлы — UserLayerSource.
Точные свойства, split 413/размерных 422, checkpoint, ошибки и TODO —
[compliance-status-contract.md](compliance-status-contract.md).
