# Frontend integration guide

Документ описывает контракт фронтенда с сервисом `agents` в gMART: запуск, базовые URL, авторизацию, REST-эндпоинты, SSE-потоки и форматы событий.

> Помимо построения ограничений, сервис `agents` предоставляет **агента вопросов по нормативной документации** (RAG поверх IDU_DVD). Его эндпоинт (`/documents/qa/stream`, `/documents/a2a`) и события описаны в [frontend-document-qa.md](frontend-document-qa.md).

## Назначение сервиса

`agents` — FastAPI-сервис, который даёт фронтенду HTTP-интерфейс к LLM-агентам и геопространственному пайплайну ограничений. Для построения слоёв сервис обращается к `idu_mcp`, а `idu_mcp` уже работает с Urban API и геометрическими инструментами.

Основной пользовательский сценарий для фронта:

1. Пользователь вводит текстовый запрос на построение ограничений.
2. Фронт открывает SSE-поток `GET /restrictions/generate_restrictions/stream`.
3. Первым событием приходит `pipeline_started` с уникальным `request_id` — его нужно сохранить.
4. Сервис постепенно присылает статусы, текстовые чанки, GeoJSON-слои и служебные события.
5. Фронт отображает прогресс, дописывает текст ответа и добавляет полученные GeoJSON-слои на карту.

## Базовые URL

При запуске через `docker-compose.yaml`:

- `agents`: `http://localhost` или `http://localhost:80`
- Swagger UI `agents`: `http://localhost/docs`
- health-check `agents`: `http://localhost/ping`
- `idu_mcp`: `http://localhost:8000`
- документация `idu_mcp`: `http://localhost:8000/docs`

Для dev-окружения используется `docker-compose-dev.yaml`, порты те же.

## Запуск локально

```bash
make build-dev
```

или для обычной сборки:

```bash
make build
```

Сервису `agents` нужны переменные окружения:

- `OLLAMA_API_URL` — URL Ollama API.
- `IDU_MCP_SERVER` — URL MCP-сервера, например `http://idu_mcp:8000/mcp`.
- `CHAT_STORAGE` — URL сервиса хранения чатов.
- `REDIS_URL` — URL Redis, например `redis://:password@pipeline_storage:6379`.

Сервису `idu_mcp` нужна переменная:

- `URBAN_API_URL` — URL Urban API.

## Авторизация

Эндпоинты, которым нужен доступ к Urban API через MCP, требуют заголовок:

```http
Authorization: Bearer <access_token>
```

Токен не валидируется внутри `agents`: сервис извлекает Bearer-токен и передаёт его дальше в `idu_mcp`/Urban API. Если заголовка нет, сервис вернёт `401`.

Токен **короткоживущий**: он может истечь прямо в процессе выполнения пайплайна. В этом случае сервис пришлёт событие `token_expired` и продолжит ждать новый токен через отдельный эндпоинт (см. [Обновление токена](#обновление-токена-token_expired)).

Обязательно передавайте токен для:

- `GET /restrictions/generate_restrictions/stream`
- `POST /a2a`

## Общие эндпоинты

### `GET /ping`

Проверка доступности сервиса.

Ответ:

```json
{ "status": "ok" }
```

### `GET /system/logs`

Скачивает лог-файл текущего запуска приложения. Используется для диагностики.

## LLM endpoints

### `GET /llm/available_models`

Возвращает список моделей Ollama.

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `only_active` | `boolean` | `false` | Если `true`, вернуть только модели, загруженные в VRAM. |

Ответ:

```json
["gpt-oss:20b"]
```

### `GET /llm/message`

Одноразовый запрос к модели без стриминга.

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `request` | `string` | обязательный | Текст запроса. |
| `model` | `string` | `gpt-oss:20b` | Имя модели Ollama. |
| `temperature` | `number` | `1.0` | Температура генерации. |

Если указанная `model` не загружена в Ollama, эндпоинт возвращает `404` с сообщением и списком доступных моделей. Если недоступен зависимый сервис (Urban API и т.п.), возвращается `502`.

### `GET /llm/message/stream`

SSE-версия простого LLM-запроса. Параметры те же, что у `/llm/message`.

Каждое SSE-событие:

```json
{ "type": "Text", "content": "фрагмент ответа" }
```

---

## Restrictions endpoint

### `GET /restrictions/generate_restrictions/stream`

Основной эндпоинт. Запускает пайплайн построения ограничений и возвращает результат через Server-Sent Events.

Headers:

```http
Authorization: Bearer <access_token>
```

Query-параметры:

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `request` | `string` | обязательный | Текст запроса, например `Построй зону ограничения вокруг школ 200 метров`. |
| `scenario_id` | `number` | обязательный | ID сценария в Urban API. |
| `model` | `string` | `gpt-oss:20b` | Имя модели Ollama. |
| `temperature` | `number` | `1.0` | Температура генерации. |
| `chat_id` | `string \| null` | `null` | UUID чата из Chat Storage. Если не передан, сервис создаст новый чат и пришлёт `service_event`. |
| `request_id` | `string \| null` | `null` | UUID пайплайна для переподключения. Передаётся только при повторном подключении. |

### `POST /restrictions/{request_id}/token`

Эндпоинт для обновления токена в работающем пайплайне. Вызывается после получения события `token_expired`.

```http
POST /restrictions/550e8400-e29b-41d4-a716-446655440001/token
Content-Type: application/json

{ "token": "<новый_access_token>" }
```

Ответ:

```json
{ "status": "ok", "request_id": "550e8400-e29b-41d4-a716-446655440001" }
```

Если пайплайн не ожидает токен (нет активной подписки) — возвращает `404`.

---

## Provision endpoint

### `GET /provision/calculate_effects/stream`

Запускает пайплайн анализа обеспеченности сервисами и возвращает результат через Server-Sent Events. Параметры и заголовки те же, что у `/restrictions/generate_restrictions/stream` (`request`, `scenario_id`, `model`, `temperature`, `chat_id`, `request_id`).

Агент сам определяет тип запроса (интент) первым LLM-вызовом и дальше выполняет детерминированный пайплайн:

| Интент | Пример запроса | Что приходит в поток |
| --- | --- | --- |
| Список сервисов | «Какие сервисы есть в проекте?» | `chunk` с текстовым списком доступных сервисов |
| Сводка по сервисам | «Дай сводку по обеспеченности», «Какими сервисами меньше всего обеспечен проект?» | `tool_call` (`CalculateServicesProvision`), `table` (`provision_summary`), `chunk` с LLM-комментарием. Слои **не** приходят, если пользователь их явно не попросил |
| Текущая обеспеченность одним сервисом | «Какая обеспеченность школами?» | `tool_call`, `feature_collection` (buildings/services/links), `table` (`provision_metrics`), `chunk` |
| Эффекты проекта по сервису | «Как проект повлияет на обеспеченность школами?» | `tool_call` (`GetServiceTypeIdByName`, `CalculateObjectEffects`), `feature_collection` (до/после/эффекты), `table` (`effects_pivot`), `chunk` |

Если сервис не найден в каталоге сценария или запрос неоднозначен, придёт `chunk` с уточняющим вопросом и перечнем доступных сервисов (`done: true`), после чего поток завершится.

**Численность населения.** Пользователь может указать целевую численность населения прямо в тексте запроса (например: «сводка по обеспеченности при населении 25 000 человек») — она будет передана в расчёт (`target_population`) и переопределит население сценария, восстановленное из Urban API. Если население не указано, в конце ответа агент добавляет подсказку о такой возможности. Агент учитывает историю чата: параметры (сервис, режим, население) можно уточнять в последующих сообщениях, при противоречиях приоритет у более поздних.

Обновление токена — через общий эндпоинт `POST /pipelines/{request_id}/token`. Табличные данные всегда приходят событием `table` со строгими, фиксированными в коде колонками — LLM не участвует в формировании таблицы и пишет только текстовый комментарий.

---

## Типы SSE-событий

События приходят в поле `data` стандартного SSE-сообщения. Значение `data` — JSON одного из типов ниже.

### `pipeline_started`

**Первое событие в каждом новом запросе.** Содержит `request_id`, который необходимо сохранить для переподключения и обновления токена.

```json
{
  "type": "pipeline_started",
  "content": {
    "request_id": "550e8400-e29b-41d4-a716-446655440001"
  }
}
```

> ⚠️ Сохраните `request_id` сразу при получении этого события.

### `status`

Прогресс пайплайна. Используйте для индикатора текущего шага.

```json
{
  "type": "status",
  "content": {
    "status": "data_retrievement",
    "text": "Получаю каталоги сервисов и физических объектов"
  }
}
```

Возможные значения `content.status`:

| Значение | Описание |
| --- | --- |
| `data_retrievement` | Получение каталогов и слоёв из Urban API |
| `plan_explanation` | LLM объясняет выбранные параметры |
| `buffer_creation` | Построение буферных зон |
| `restriction_formation` | Извлечение нормативных ограничений |
| `context_preparation` | Подготовка контекста для финального ответа |

### `chunk`

Текстовая часть ответа ассистента. Конкатенируйте `content.text` по мере поступления.

```json
{
  "type": "chunk",
  "content": {
    "text": "Для запроса выбраны школы...",
    "done": false
  }
}
```

При `content.done === true` текстовый ответ завершён.

### `feature_collection`

GeoJSON-слой, который нужно добавить на карту.

```json
{
  "type": "feature_collection",
  "content": {
    "name": "schools_buffer_200m",
    "feature_collection": {
      "type": "FeatureCollection",
      "features": []
    }
  }
}
```

`content.name` можно использовать как название слоя. Пустой `features` — валидный результат ("объектов не найдено").

### `table`

Строгая таблица с данными расчёта (пока только в provision-пайплайне). Колонки формируются кодом сервиса, а не LLM: ключи (`columns[].key`) стабильны между запросами, подписи (`columns[].label`) — человекочитаемые русские названия. Рендерите таблицу до/рядом с текстовым комментарием.

```json
{
  "type": "table",
  "content": {
    "name": "provision_summary",
    "title": "Сводка обеспеченности сервисами",
    "columns": [
      { "key": "service", "label": "Сервис" },
      { "key": "capacity", "label": "Вместимость (чел)" },
      { "key": "demand", "label": "Спрос (чел)" },
      { "key": "deficit", "label": "Дефицит (чел)" },
      { "key": "surplus", "label": "Профицит (чел)" },
      { "key": "balance", "label": "Баланс (чел)" }
    ],
    "rows": [
      { "service": "Школы", "capacity": 1200, "demand": 1450, "deficit": 250, "surplus": 0, "balance": -250 }
    ]
  }
}
```

Известные таблицы (`content.name`):

| `name` | Пайплайн | Колонки |
| --- | --- | --- |
| `provision_summary` | Сводка по сервисам | `service`, `capacity`, `demand`, `deficit`, `surplus`, `balance` (строки отсортированы по убыванию дефицита) |
| `provision_metrics` | Обеспеченность одним сервисом | `metric`, `value` |
| `effects_pivot` | Эффекты проекта по сервису | `metric`, `value` |

В историю чата таблица сохраняется частью сообщения `kind: "table"` с тем же payload (см. документацию ChatStorage).

### `tool_call`

Информация о MCP-инструментах, вызванных в текущем шаге. Можно использовать для расширенного отображения прогресса.

```json
{
  "type": "tool_call",
  "content": {
    "execution_mode": "data_retrievement",
    "tool_calls": [
      { "tool_name": "GetServices", "arguments": { "scenario_id": 772 } }
    ]
  }
}
```

### `service_event`

Служебное событие о действии в Chat Storage. Сейчас используется только для создания нового чата.

```json
{
  "type": "service_event",
  "content": {
    "event_type": "storage_event",
    "event": {
      "storage_event_type": "chat_created",
      "chat_id": "550e8400-e29b-41d4-a716-446655440000",
      "chat_title": "Ограничения вокруг школ"
    }
  }
}
```

Сохраните `chat_id` и передавайте его в следующие запросы этого диалога.

### `token_expired`

Токен истёк в процессе выполнения пайплайна. **Не закрывайте SSE-соединение.** Вместо этого получите новый токен и отправьте его через `POST /restrictions/{request_id}/token`.

```json
{
  "type": "token_expired",
  "content": {
    "request_id": "550e8400-e29b-41d4-a716-446655440001",
    "message": "Токен истёк. Пожалуйста, обновите токен."
  }
}
```

### `pipeline_suspended`

Токен так и не был обновлён в течение отведённого времени (360 секунд / 6 минут). Пайплайн сохранил чекпоинт и завершил работу. SSE-поток закрывается.

```json
{
  "type": "pipeline_suspended",
  "content": {
    "request_id": "550e8400-e29b-41d4-a716-446655440001",
    "message": "Выполнение приостановлено: токен не был обновлён вовремя. Переподключитесь с тем же request_id, чтобы продолжить."
  }
}
```

После получения этого события переподключитесь с тем же `request_id` и новым токеном.

### `error`

Необработанная ошибка внутри SSE-пайплайна.

```json
{
  "type": "error",
  "content": {
    "message": "Internal stream exception",
    "traceback": "..."
  }
}
```

Показывайте пользователю дружелюбное сообщение. После `error` сервис дополнительно отправляет завершающий `chunk` с `done: true`.

Типичные причины, которые теперь **гарантированно** приходят как событие `error` (а не подвешивают поток):

- **Запрошенная модель не загружена в Ollama** — сервис отдаёт ошибку с упоминанием модели и списком доступных. Загрузите модель (`ollama pull <model>`) или выберите доступную из `GET /llm/available_models`.
- **Внешний сервис недоступен** (Urban API и т.п.) — запрос повторяется ограниченное число раз с задержкой, после чего сервис сообщает о недоступности зависимого сервиса, а не зацикливается.

---

## Основные флоу

### Флоу 1 — Обычный запрос (новый чат)

```
Фронт                                    Сервер
  |                                         |
  |-- GET /stream?request=...&scenario_id=772 -->|
  |                                         |
  |<-- pipeline_started { request_id } -----|  ← сохранить request_id
  |<-- service_event { chat_created, chat_id } -|  ← сохранить chat_id
  |<-- status { data_retrievement } --------|
  |<-- status { plan_explanation } ---------|
  |<-- chunk { text, done:false } ----------|  ← накапливать текст
  |<-- chunk { text, done:false } ----------|
  |<-- chunk { text, done:true } -----------|  ← текст завершён
  |<-- status { data_retrievement } --------|
  |<-- tool_call { ... } -------------------|
  |<-- feature_collection { name, geojson } |  ← добавить на карту
  |<-- status { buffer_creation } ----------|
  |<-- feature_collection { name, geojson } |  ← добавить на карту
  |<-- chunk { text, done:true } -----------|  ← финальный ответ
  |                                         |
  |  [поток завершён]                       |
```

### Флоу 2 — Продолжение диалога (существующий чат)

Передайте `chat_id` из предыдущего запроса:

```
GET /stream?request=...&scenario_id=772&chat_id=550e8400-...
```

Сервис загрузит историю чата и учтёт её в контексте LLM. Событие `service_event/chat_created` в этом случае не придёт.

### Флоу 3 — Обновление токена

Если токен истёк в процессе выполнения:

```
Фронт                                    Сервер
  |                                         |
  |  [SSE-поток открыт]                     |
  |<-- token_expired { request_id } --------|  ← НЕ закрывать поток
  |                                         |
  |-- POST /restrictions/{request_id}/token -->|
  |   { "token": "<новый_токен>" }          |
  |<-- { "status": "ok" } ------------------|
  |                                         |
  |  [пайплайн продолжается]                |
  |<-- status { ... } ----------------------|
  |<-- chunk / feature_collection / ... ----|
```

### Флоу 4 — Переподключение после разрыва

Если соединение разорвалось и `request_id` был сохранён:

```
Фронт                                    Сервер
  |                                         |
  |  [разрыв соединения]                    |
  |                                         |
  |-- GET /stream?request=...               |
  |       &scenario_id=772                  |
  |       &request_id=550e8400-... -------->|  ← тот же request_id
  |                                         |
  |<-- [повтор всех событий из буфера] -----|  ← все предыдущие события
  |<-- [продолжение с последнего чекпоинта]-|  ← уже выполненные шаги пропускаются
  |<-- chunk / feature_collection / ... ----|
```

> **Важно:** если разрыв произошёл *до* получения `pipeline_started`, `request_id` неизвестен — начинайте новый запрос без `request_id`.

### Флоу 5 — Пайплайн приостановлен (`pipeline_suspended`)

Если токен не был обновлён вовремя (> 360 сек / 6 мин):

```
Фронт                                    Сервер
  |                                         |
  |<-- token_expired { request_id } --------|
  |  [нет ответа 60 секунд]                 |
  |<-- pipeline_suspended { request_id } ---|  ← поток завершится
  |                                         |
  |  [получить новый токен]                 |
  |                                         |
  |-- GET /stream?request=...               |
  |       &scenario_id=772                  |
  |       &request_id=550e8400-... -------->|  ← переподключение
  |  { Authorization: Bearer <новый_токен> }|
  |                                         |
  |<-- [повтор буфера + продолжение] -------|
```

---

## TypeScript-типы и пример клиента

```ts
type PipelineStartedEvent = {
  type: "pipeline_started";
  content: { request_id: string };
};

type StatusEvent = {
  type: "status";
  content: {
    status:
      | "data_retrievement"
      | "plan_explanation"
      | "buffer_creation"
      | "restriction_formation"
      | "context_preparation";
    text: string;
  };
};

type ChunkEvent = {
  type: "chunk";
  content: { text: string; done: boolean };
};

type FeatureCollectionEvent = {
  type: "feature_collection";
  content: { name: string; feature_collection: GeoJSON.FeatureCollection };
};

type ToolCallEvent = {
  type: "tool_call";
  content: { execution_mode: string; tool_calls: unknown[] };
};

type ServiceEvent = {
  type: "service_event";
  content: {
    event_type: "storage_event";
    event: {
      storage_event_type: "chat_created";
      chat_id: string;
      chat_title: string;
    };
  };
};

type TokenExpiredEvent = {
  type: "token_expired";
  content: { request_id: string; message: string };
};

type PipelineSuspendedEvent = {
  type: "pipeline_suspended";
  content: { request_id: string; message: string };
};

type ErrorEvent = {
  type: "error";
  content: { message: string; traceback?: string };
};

type RestrictionEvent =
  | PipelineStartedEvent
  | StatusEvent
  | ChunkEvent
  | FeatureCollectionEvent
  | ToolCallEvent
  | ServiceEvent
  | TokenExpiredEvent
  | PipelineSuspendedEvent
  | ErrorEvent;
```

```ts
interface StreamRestrictionParams {
  baseUrl: string;
  request: string;
  scenarioId: number;
  getToken: () => string;           // вызывается перед каждым подключением
  chatId?: string;
  model?: string;
  temperature?: number;
  requestId?: string;               // передать при переподключении
  onEvent: (event: RestrictionEvent) => void;
  onRequestId?: (requestId: string) => void;  // вызывается при pipeline_started
  onChatId?: (chatId: string) => void;        // вызывается при chat_created
}

async function streamRestrictions(params: StreamRestrictionParams): Promise<void> {
  const url = new URL("/restrictions/generate_restrictions/stream", params.baseUrl);
  url.searchParams.set("request", params.request);
  url.searchParams.set("scenario_id", String(params.scenarioId));
  url.searchParams.set("model", params.model ?? "gpt-oss:20b");
  url.searchParams.set("temperature", String(params.temperature ?? 1));
  if (params.chatId)   url.searchParams.set("chat_id", params.chatId);
  if (params.requestId) url.searchParams.set("request_id", params.requestId);

  const response = await fetch(url.toString(), {
    headers: {
      Authorization: `Bearer ${params.getToken()}`,
      Accept: "text/event-stream",
    },
  });

  if (!response.ok || !response.body) {
    throw new Error(`Stream failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const messages = buffer.split("\n\n");
    buffer = messages.pop() ?? "";

    for (const message of messages) {
      const dataLine = message.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;

      const event = JSON.parse(dataLine.slice("data:".length).trim()) as RestrictionEvent;
      params.onEvent(event);

      // Служебная обработка ключевых событий
      if (event.type === "pipeline_started") {
        params.onRequestId?.(event.content.request_id);
      }

      if (
        event.type === "service_event" &&
        event.content.event.storage_event_type === "chat_created"
      ) {
        params.onChatId?.(event.content.event.chat_id);
      }

      if (event.type === "token_expired") {
        // Получите свежий токен (например, через refresh) и отправьте его
        const newToken = params.getToken();
        await fetch(
          new URL(`/restrictions/${event.content.request_id}/token`, params.baseUrl).toString(),
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: newToken }),
          }
        );
        // SSE-поток продолжается сам — ничего закрывать не нужно
      }
    }
  }
}
```

---

## Рекомендации по обработке событий

| Событие | Что делать |
| --- | --- |
| `pipeline_started` | Сохранить `request_id` (нужен для переподключения и обновления токена) |
| `service_event/chat_created` | Сохранить `chat_id` для следующих запросов в этом диалоге |
| `status` | Обновить индикатор прогресса |
| `chunk` | Дописать `text` к ответу; при `done: true` — завершить генерацию |
| `feature_collection` | Немедленно добавить слой на карту, не ждать конца потока |
| `table` | Отрисовать таблицу по `columns`/`rows`; порядок колонок задаёт порядок отображения |
| `tool_call` | Опционально: показать, какие инструменты вызываются |
| `token_expired` | **Не закрывать поток.** Получить новый токен и отправить `POST /restrictions/{request_id}/token` |
| `pipeline_suspended` | Показать уведомление. Переподключиться с тем же `request_id` и новым токеном |
| `error` | Показать пользователю дружелюбное сообщение; технические детали — в лог |

---

## Ошибки

Обычные HTTP-ошибки возвращаются как JSON:

```json
{
  "message": "Authorization header missing",
  "input": null
}
```

Для непойманных исключений middleware возвращает:

```json
{
  "message": "Internal server error",
  "error_type": "ValueError",
  "request": { "method": "GET", "url": "...", "query_params": {} },
  "detail": "...",
  "traceback": []
}
```

> Для SSE-потоков ошибки приходят как событие `type: "error"` внутри потока. Не полагайтесь только на HTTP status: поток может стартовать с `200 OK`, а затем завершиться ошибкой на одном из шагов.

---

## A2A endpoint

### `GET /.well-known/agent-card.json`

Карточка A2A-агента для автообнаружения.

### `POST /a2a`

JSON-RPC endpoint для A2A-клиентов. Для обычного веб-фронта проще использовать `/restrictions/generate_restrictions/stream`.

Headers:

```http
Authorization: Bearer <access_token>
Content-Type: application/json
```

Пример streaming-запроса:

```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "method": "SendStreamingMessage",
  "params": {
    "message": {
      "role": "user",
      "parts": [{ "type": "text", "text": "Построй зону ограничения вокруг школ 200 метров" }],
      "metadata": { "scenario_id": 772, "model": "gpt-oss:20b", "temperature": 0.7 }
    }
  }
}
```

Поддерживаемые методы: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`, `GetExtendedAgentCard`.
