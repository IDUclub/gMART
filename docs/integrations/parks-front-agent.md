# ICII-234 — контракт интеграции парков для фронтенд-агента

Это спецификация реализованного gMART-адаптера. Существующий общий SSE-конверт
`{type, content}` сохранён. Старые endpoints и их DTO не меняются.

## Источники и включение

- Upstream OpenAPI 1.0.1: https://api.1.urban.actcognitive.org/v3/api-docs
- Веб-клиент: https://parks.urban.itmo.ru
- Проверено 31 октября 2026 года. Снимок OpenAPI лежит в
  [contracts/parks-upstream-openapi.json](contracts/parks-upstream-openapi.json).
- OpenAPI задаёт `status` как integer без перечисления. Семантика `1=готово`,
  `-1=ошибка` подтверждена веб-клиентом `/js/app.668173da.js`; остальные значения
  означают продолжение ожидания. Числовой процент прогресса из них не вычисляется.
- В конфигурации backend задаётся `PARKS_API_URL`. Без него REST отвечает 503,
  агент исключён из каталога оркестратора.

## REST

Все пользовательские endpoints требуют `Authorization: Bearer <token>`.
Нельзя передавать токен в URL. Прямое обращение фронта к upstream не требуется.

| Endpoint | Назначение |
| --- | --- |
| `GET /parks/generate/stream` | Новое сообщение или восстановление SSE |
| `POST /parks/generate/stream` | Эквивалент с JSON-телом для большого GeoJSON |
| `GET /parks/requests/{request_id}` | Снимок состояния владельца запроса |
| `GET /parks/.well-known/agent-card.json` | Публичная A2A-карточка |
| `POST /parks/a2a` | A2A JSON-RPC 2.0, обычный JSON или SSE |

GET принимает стандартные `request` (обязательный), `model`, `temperature`,
`scenario_id`, `chat_id`, `request_id`; дополнительно `inputs` (JSON-строка),
`conversation_id` (запасной ключ диалога), `after` (целое >=0, по умолчанию 0).
POST принимает стандартные поля и объект `inputs` в теле; `conversation_id` и `after`
остаются в query. `Last-Event-ID` имеет приоритет над `after`.

```ts
type ParkTag = 'ART' | 'EXHIBITION' | 'TRADE' | 'FOOD' | 'REST' | 'CHILDREN'
  | 'BICYCLE' | 'SPORTS' | 'PARKING' | 'WATER' | 'ANIMALS' | 'AMUSEMENT' | 'GREENERY';

interface ParksInputs {
  park_geojson?: GeoJSON.FeatureCollection;
  tags?: ParkTag[]; // повторы значимы: ['REST', 'REST'] = два объекта
  location_requests?: {lat: number; lon: number; text: string}[];
  confirmed_revision?: number; // положительное целое, версия согласования
}
```

Лимиты: до 200 тегов и 200 локальных пожеланий; текст локального пожелания до 4000
символов. `lat` в [-90,90], `lon` в [-180,180]. Геометрия валидируется, нужен ровно
один boundary Polygon/MultiPolygon. Для запуска обязательны минимум две разные
entry Point (различаются по longitude/latitude). Черновик без входов принимается,
но возвращает `ready=false` и вопрос о входах; подтверждение не запускает upstream.
Сохранённые ранее черновики также перепроверяются перед подтверждением. Другие типы и
неизвестные поля `inputs` отклоняются. Пустой список тегов снимает пожелания и
требует нового согласования. Не передавайте поля, которые не хотите менять.

## Разделение нового сообщения и восстановления

- Новая реплика: прежний `chat_id`, новый текст, `request_id` отсутствует.
- Восстановление: прежний `request_id`, текст игнорируется; параметры и модель
  берутся из Redis. Без курсора журнал выдаётся с начала.
- Подтверждение кнопкой: новая реплика, `inputs.confirmed_revision` равен последнему
  `revision`. Изменённые параметры одновременно с подтверждением создают новый черновик.
- Подтверждение текстом: отдельное недвусмысленное «Подтверждаю», «Да», «Запускай».
  «Да, но добавь спорт» обрабатывается как правка.
- Подтверждение устаревшей версии не запускает задачу, возвращает текущий черновик.
- Повторное подтверждение уже запущенного диалога возвращает `active_request_id`.
  Для нового независимого парка создайте новый чат/ключ диалога.

`conversation_id` из `clarification` можно использовать при работе без ChatStorage.
Обычно достаточно сохранять `chat_id` из стандартного `service_event`:

```json
{"type":"service_event","content":{"event_type":"storage_event","event":{"storage_event_type":"chat_created","chat_id":"UUID","chat_title":"Парк"},"event_id":2}}
```

## События и действия интерфейса

Каждое сохранённое событие имеет SSE `id` — положительное целое, также находящееся
в `content.event_id`. Оно уникально в пределах одного `request_id`. Обычный
heartbeat приходит без `id` и не записывается в журнал.

| `type` | Поля `content` | Действие |
| --- | --- | --- |
| `pipeline_started` | `request_id`, `event_id` | Сохранить запрос |
| `service_event` | стандартный `storage_event` | Сохранить чат |
| `clarification` | `question`, `parameters`, `revision`, `ready`, `conversation_id`, иногда `active_request_id` | Показать параметры/вопрос, ждать следующей реплики |
| `status` | `status`, `text`, иногда `task_id`, `request_id`, `description` | Показать текущий этап |
| `status` heartbeat | `status="heartbeat"`, `text`, `phase`, `task_id`, `request_id` | Обновить время последней активности |
| `status` ТЭП | `status="economy"`, `text`, `economy` | Показать показатели из `economy.economyParams` |
| `feature_collection` | `name`, `feature_collection`, `task_id` | Добавить/обновить GeoJSON-слой |
| `chunk` | `text`, `done` | Добавить текст, закрыть ответ при `done=true` |
| `warning` | `code`, `message` | Показать неблокирующее предупреждение |
| `error` | `message`, `traceback=""`, иногда `request_id` | Показать ошибку; технических трассировок нет |

Жизненный цикл: `planning → awaiting_confirmation`; после отдельного подтверждения
`planning → submitting → generating → analysis → done`. `status` состояния запроса:
`running`, `awaiting_confirmation`, `done`, `failed`, `submission_unknown`.
`phase` является более подробным этапом, а не признаком завершения.

Генерация опрашивается каждые 3 секунды, ожидание до 600 секунд; heartbeat — каждые
10 секунд, в том числе во время LLM-вызовов. Обрыв SSE не отменяет worker.
После перезапуска новый подписчик подхватывает checkpoint, когда освободится
30-секундная Redis-блокировка. Redis TTL — 86400 секунд с последней записи.

Успешный порядок: статусы → GeoJSON → доступные ТЭП → статус анализа → текст
анализа и рекомендаций → пустой `chunk(done=true)`. Вопрос согласования и ошибка
также заканчиваются `chunk(done=true)`. EOF сам по себе не доказывает успех.

Если анализ LLM упал, уже доставленный GeoJSON остаётся пригодным; восстановление
с `request_id` повторяет анализ без повторной генерации. Отсутствие ТЭП даёт
`warning(code=economy_unavailable)` и не блокирует анализ. Анализ использует
GeoJSON и ТЭП; если результат содержит `images` с inline PNG/JPEG/WebP data URLs,
до четырёх изображений передаются vision-модели. Текущий upstream-контракт
изображений не обещает. Контекст анализа — валидный JSON до 60000 символов:
из первых 150 объектов выбираются только целые объекты, помещающиеся в бюджет.
Поля `included_features`, `omitted_features`, `context_truncated`, `omitted_fields`
явно описывают усечение; крупные параметры/ТЭП при необходимости исключаются целиком.
Полный GeoJSON фронту не обрезается. Вывод LLM не является проверкой нормативного
соответствия всех объектов.

## Ошибки и повторные попытки

400/422 — исправить входные данные/курсор; 404 — неизвестный, истёкший или чужой
`request_id`; 503 — интеграция выключена. После начала SSE ошибки передаются в потоке.
Не создавайте новую задачу автоматически при сетевой ошибке — сначала восстановите
существующий `request_id`.

Upstream не документирует идемпотентность `/map/generate`. Перед POST gMART
сохраняет намерение отправки; после ответа сохраняет `taskId`. Разрыв между ними
даёт `submission_unknown`. Автоматическая повторная генерация в этом состоянии
запрещена; нужно проверить запуск на стороне сервиса. Это ограничение upstream,
а не сигнал о гарантированном отсутствии результата.

## Оркестратор и A2A

Оркестратор принимает `parks_inputs`: JSON в query GET или объект в теле POST
`/orchestrator/route/stream`. Agent key — `parks`. Вложенный конверт:

```json
{"type":"step_event","content":{"step":1,"agent":"parks","event":{"type":"clarification","content":{"question":"Подтвердите параметры","revision":1,"ready":true,"parameters":{}}}}}
```

Сохраните внешний `request_id` и `step_started.content.step_request_id`. Первый
восстанавливает поток оркестратора; второй восстанавливает сам парковой запрос.
Продолжение согласования — новое сообщение того же чата, без внешнего `request_id`.
ChatStorage обслуживает оркестратор; внутренний агент не создаёт отдельного чата.
После успешного паркового шага оркестратор сохраняет полный текст анализа и те же
`data`-конверты `feature_collection`/`parks_economy`, что прямой REST. Они сохраняются
и при восстановлении шага из Redis; replay завершённого запроса повторную запись
не создаёт. Краткий digest для последующих шагов не заменяет полный текст в истории.
Прямой REST-агент сохраняет пользовательские реплики и вопросы согласования;
готовый результат сохраняется до вызова LLM как сообщение с `data`-частями
`feature_collection` и `parks_economy`. Анализ и рекомендации сохраняются
отдельным текстовым сообщением после успешного ответа модели. При чтении истории используйте их
для восстановления карты и показателей. Ошибка ChatStorage даёт предупреждение;
продолжение без чата возможно по возвращённому `conversation_id`.

В основном `docker-compose.yaml` для Redis включены AOF и том `/data`.
При отдельном развёртывании Redis также необходимо сохранять данные на постоянном
диске: восстановление backend опирается на сохранность checkpoint в Redis.

A2A поддерживает `message/send`, `message/stream`, `tasks/get` и их существующие
в gMART aliases `SendMessage`, `SendStreamingMessage`, `GetTask`, `tasks/send`,
`tasks/sendSubscribe`. Отмена и push notifications не заявлены.
Финальный streaming-статус читается из Redis, а не определяется оставшимися событиями:
replay с курсором после последнего события сохраняет `input-required` для согласования
и `failed` для неопределённого запуска, не объявляя их завершёнными.

```json
{
  "jsonrpc":"2.0","id":1,"method":"message/stream",
  "params":{
    "contextId":"park-conversation-1",
    "message":{"role":"user","parts":[{"kind":"text","text":"Создай парк"}]},
    "metadata":{"inputs":{"tags":["REST","GREENERY"]}}
  }
}
```

В первый запрос добавьте `metadata.inputs.park_geojson`. Параметры можно передавать
также в DataPart `data.inputs`. Для ответа на согласование оставьте прежний
`contextId`, отправьте новое сообщение «Подтверждаю» и при необходимости
`inputs.confirmed_revision`. Для восстановления передайте `metadata.request_id`.
A2A `taskId` совпадает с парковым `request_id`; задачи и артефакты читаются из Redis
с проверкой владельца. Согласование — `input-required`, генерация — `working`,
успех — `completed`; GeoJSON и анализ — `artifact-update`. A2A не пишет в ChatStorage.

## Приёмка фронта

- [ ] Читаются SSE-кадры, разбитые на произвольные сетевые чанки; UTF-8 не теряется.
- [ ] Граница, entry points, пожелания и их количества видны до подтверждения.
- [ ] Правки не используют прежний `request_id` и обновляют `revision`.
- [ ] Кнопка подтверждения отправляет только номер версии и текст подтверждения.
- [ ] Heartbeat виден во время ожидания, без выдуманного процента прогресса.
- [ ] Результат на карте появляется до анализа; ТЭП и рекомендации отображаются.
- [ ] После обрыва/перезагрузки используются прежний `request_id` и курсор.
- [ ] Replay не дублирует слои и текст; heartbeat не сдвигает курсор.
- [ ] `submission_unknown` не запускает автоматическую новую генерацию.
- [ ] В оркестраторе те же обработчики применяются к `step_event.content.event`.

Проверки backend с fakeredis, fake LLM и HTTP mock находятся в
`tests/unit/test_parks_integration.py` и `tests/unit/test_parks_regressions.py`.
Сохранение результатов оркестратора и восстановление истории покрыты
`tests/unit/test_orchestrator_parks_persistence.py`.
Выполнен живой прогон gMART → ParkPlanner → Qwen: с двумя входами получены GeoJSON,
ТЭП, анализ и финальный `done`; проверена блокировка запуска без входов.
ChatStorage и авторизация на стенде были заглушками. Реальные изображения не
передавались: vision-путь покрыт mock-тестом, но не живой проверкой.
