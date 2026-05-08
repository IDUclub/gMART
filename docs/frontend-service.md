# Frontend integration guide

Документ описывает контракт фронтенда с сервисом `agents` в gMART: запуск, базовые URL, авторизацию, REST-эндпоинты, SSE-потоки и форматы событий.

## Назначение сервиса

`agents` - FastAPI-сервис, который дает фронтенду HTTP-интерфейс к LLM-агентам и геопространственному пайплайну ограничений. Для построения слоев сервис обращается к `idu_mcp`, а `idu_mcp` уже работает с Urban API и геометрическими инструментами.

Основной пользовательский сценарий для фронта:

1. Пользователь вводит текстовый запрос на построение ограничений.
2. Фронт открывает SSE-поток `GET /restrictions/generate_restrictions/stream`.
3. Сервис постепенно присылает статусы, текстовые чанки, GeoJSON-слои и служебные события.
4. Фронт отображает прогресс, дописывает текст ответа и добавляет полученные GeoJSON-слои на карту.

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

- `OLLAMA_API_URL` - URL Ollama API.
- `IDU_MCP_SERVER` - URL MCP-сервера, например `http://idu_mcp:8000/mcp`.
- `CHAT_STORAGE` - URL сервиса хранения чатов.

Сервису `idu_mcp` нужна переменная:

- `URBAN_API_URL` - URL Urban API.

## Авторизация

Эндпоинты, которым нужен доступ к Urban API через MCP, требуют заголовок:

```http
Authorization: Bearer <access_token>
```

Токен не валидируется внутри `agents`: сервис извлекает Bearer-токен и передает его дальше в `idu_mcp`/Urban API. Если заголовка нет, FastAPI вернет ошибку авторизации.

Обязательно передавайте токен для:

- `GET /restrictions/generate_restrictions/stream`
- `POST /a2a`

## Общие эндпоинты

### `GET /ping`

Проверка доступности сервиса.

Ответ:

```json
{
  "status": "ok"
}
```

### `GET /system/logs`

Скачивает лог-файл текущего запуска приложения. Используется для диагностики, в пользовательском интерфейсе обычно не нужен.

## LLM endpoints

### `GET /llm/available_models`

Возвращает список моделей Ollama.

Query-параметры:

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `only_active` | `boolean` | `false` | Если `true`, вернуть только модели, загруженные в VRAM сервера. |

Пример:

```http
GET /llm/available_models?only_active=false
```

Ответ:

```json
[
  "gpt-oss:20b"
]
```

### `GET /llm/message`

Одноразовый запрос к модели без стриминга.

Query-параметры:

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `request` | `string` | обязательный | Текст пользовательского запроса. |
| `model` | `string` | `gpt-oss:20b` | Имя модели Ollama. |
| `temperature` | `number` | `1.0` | Сейчас поле есть в DTO, но в реализации этого эндпоинта не используется. |

Пример:

```http
GET /llm/message?model=gpt-oss:20b&request=Почему%20небо%20синее%3F
```

Ответ проксирует объект ответа Ollama.

### `GET /llm/message/stream`

SSE-версия простого LLM-запроса.

Query-параметры такие же, как у `/llm/message`.

Каждое SSE-событие содержит JSON:

```json
{
  "type": "Text",
  "content": "фрагмент ответа"
}
```

## Restrictions endpoint

### `GET /restrictions/generate_restrictions/stream`

Основной эндпоинт для фронта. Запускает пайплайн построения ограничений и возвращает результат через Server-Sent Events.

Headers:

```http
Authorization: Bearer <access_token>
```

Query-параметры:

| Параметр | Тип | По умолчанию | Описание |
| --- | --- | --- | --- |
| `request` | `string` | обязательный | Пользовательский запрос, например `Построй зону ограничения вокруг школ 200 метров`. |
| `scenario_id` | `number` | обязательный | ID сценария в Urban API. |
| `chat_id` | `string \| null` | `null` | UUID чата из Chat Storage. Если не передан, сервис создаст новый чат и пришлет `service_event`. |
| `model` | `string` | `gpt-oss:20b` | Имя модели Ollama. |
| `temperature` | `number` | `1.0` | Температура генерации. |

Пример URL:

```http
GET /restrictions/generate_restrictions/stream?scenario_id=772&model=gpt-oss:20b&temperature=0.7&request=%D0%9F%D0%BE%D1%81%D1%82%D1%80%D0%BE%D0%B9%20%D0%B7%D0%BE%D0%BD%D1%83%20%D0%BE%D0%B3%D1%80%D0%B0%D0%BD%D0%B8%D1%87%D0%B5%D0%BD%D0%B8%D1%8F%20%D0%B2%D0%BE%D0%BA%D1%80%D1%83%D0%B3%20%D1%88%D0%BA%D0%BE%D0%BB%20200%20%D0%BC%D0%B5%D1%82%D1%80%D0%BE%D0%B2
```

### Типы SSE-событий

События приходят в поле `data` стандартного SSE-сообщения. Значение `data` - JSON одного из типов ниже.

#### `status`

Прогресс пайплайна. Используйте для индикатора статуса.

```json
{
  "type": "status",
  "content": {
    "status": "data_retrievement",
    "text": "Получаю каталоги сервисов и физических объектов"
  }
}
```

Возможные `content.status`:

- `data_retrievement`
- `plan_explanation`
- `buffer_creation`
- `restriction_formation`
- `context_preparation`

#### `chunk`

Текстовая часть ответа ассистента.

```json
{
  "type": "chunk",
  "content": {
    "text": "Для запроса выбраны школы...",
    "done": false
  }
}
```

Фронт должен конкатенировать `content.text`. Когда `content.done === true`, текстовый ответ завершен.

#### `feature_collection`

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

`content.feature_collection` соответствует GeoJSON `FeatureCollection`. `content.name` можно использовать как название слоя в UI.

#### `service_event`

Служебное событие. Сейчас используется для создания нового чата, если фронт не передал `chat_id`.

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

После получения `chat_created` сохраните `chat_id` и передавайте его в следующие запросы этого диалога.

#### `error`

Ошибка внутри SSE-пайплайна.

```json
{
  "type": "error",
  "content": {
    "message": "Internal stream exception",
    "traceback": "..."
  }
}
```

Фронту лучше показать пользователю дружелюбное сообщение, а технические детали отправить в клиентский лог или систему мониторинга. После `error` сервис дополнительно отправляет завершающий `chunk` с `done: true`.

### Пример клиента для SSE

`EventSource` не позволяет добавить `Authorization` header. Поэтому для защищенного потока используйте `fetch` и читайте `ReadableStream`.

```ts
type RestrictionEvent =
  | {
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
    }
  | { type: "chunk"; content: { text: string; done: boolean } }
  | {
      type: "feature_collection";
      content: { name: string; feature_collection: GeoJSON.FeatureCollection };
    }
  | {
      type: "service_event";
      content: {
        event_type: "storage_event";
        event: {
          storage_event_type: "chat_created";
          chat_id: string;
          chat_title: string;
        };
      };
    }
  | { type: "error"; content: { message: string; traceback?: string } };

async function streamRestrictions(params: {
  baseUrl: string;
  token: string;
  request: string;
  scenarioId: number;
  chatId?: string;
  model?: string;
  temperature?: number;
  onEvent: (event: RestrictionEvent) => void;
}) {
  const url = new URL("/restrictions/generate_restrictions/stream", params.baseUrl);
  url.searchParams.set("request", params.request);
  url.searchParams.set("scenario_id", String(params.scenarioId));
  url.searchParams.set("model", params.model ?? "gpt-oss:20b");
  url.searchParams.set("temperature", String(params.temperature ?? 1));
  if (params.chatId) url.searchParams.set("chat_id", params.chatId);

  const response = await fetch(url, {
    headers: {
      Authorization: `Bearer ${params.token}`,
      Accept: "text/event-stream",
    },
  });

  if (!response.ok || !response.body) {
    throw new Error(`Restrictions stream failed: ${response.status}`);
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
      const dataLine = message
        .split("\n")
        .find((line) => line.startsWith("data:"));

      if (!dataLine) continue;

      const event = JSON.parse(dataLine.slice("data:".length).trim()) as RestrictionEvent;
      params.onEvent(event);
    }
  }
}
```

Рекомендуемая обработка на фронте:

- `status`: обновить текущий шаг прогресса.
- `chunk`: добавить `text` к уже показанному ответу; при `done: true` завершить состояние генерации.
- `feature_collection`: добавить слой на карту, используя `name` как заголовок.
- `service_event/chat_created`: сохранить `chat_id`.
- `error`: показать ошибку и остановить интерактивные индикаторы.

## A2A endpoint

### `GET /.well-known/agent-card.json`

Возвращает карточку A2A-агента. Нужна для клиентов, которые умеют обнаруживать A2A-агентов автоматически.

### `GET /.well-known/agent.json`

Legacy alias для карточки A2A-агента.

### `POST /a2a`

JSON-RPC endpoint для A2A-клиентов. Для обычного веб-фронта проще использовать `/restrictions/generate_restrictions/stream`, но `/a2a` полезен для интеграции с агентскими платформами.

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
      "parts": [
        {
          "type": "text",
          "text": "Построй зону ограничения вокруг школ 200 метров"
        }
      ],
      "metadata": {
        "scenario_id": 772,
        "model": "gpt-oss:20b",
        "temperature": 0.7
      }
    }
  }
}
```

Streaming-методы возвращают SSE, где `data` содержит JSON-RPC envelope:

```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "result": {
    "artifactUpdate": {
      "taskId": "...",
      "contextId": "...",
      "artifact": {
        "artifactId": "geojson-schools-buffer-200m",
        "name": "schools_buffer_200m",
        "parts": [
          {
            "type": "data",
            "data": {
              "type": "FeatureCollection",
              "features": []
            },
            "mediaType": "application/vnd.geo+json"
          }
        ]
      },
      "append": false,
      "lastChunk": true
    }
  }
}
```

Поддерживаемые методы:

- `SendMessage`, `message/send`, `tasks/send` - выполнить задачу и вернуть финальный task.
- `SendStreamingMessage`, `message/stream`, `tasks/sendSubscribe` - выполнить задачу в SSE-режиме.
- `GetTask`, `tasks/get` - получить task по `id` или `taskId`.
- `ListTasks`, `tasks/list` - получить список task.
- `CancelTask`, `tasks/cancel` - отменить task.
- `GetExtendedAgentCard`, `agent/getAuthenticatedExtendedCard` - получить расширенную agent card.

## Ошибки

Обычные HTTP-ошибки приходят в JSON-ответе. Для непойманных исключений middleware возвращает:

```json
{
  "message": "Internal server error",
  "error_type": "ValueError",
  "request": {
    "method": "GET",
    "url": "...",
    "query_params": {}
  },
  "detail": "...",
  "traceback": []
}
```

Для SSE-потоков ошибки приходят как событие `type: "error"` внутри потока. Не полагайтесь только на HTTP status: поток может стартовать с `200 OK`, а затем завершиться ошибкой на одном из шагов пайплайна.

## Замечания для UI

- Показывайте пользователю шаги пайплайна из `status`, потому что построение слоев может занимать заметное время.
- Не блокируйте карту до окончания всего потока: `feature_collection` можно отображать сразу по мере получения.
- Сохраняйте `chat_id` после `service_event/chat_created`, иначе каждый новый запрос будет создавать новый чат.
- Обрабатывайте отключение пользователя: при закрытии страницы или отмене запроса прерывайте `fetch` через `AbortController`.
- Для текстовых ответов используйте append-логику: один ответ приходит множеством `chunk`.
- Для GeoJSON проверяйте `feature_collection.features.length`: пустой слой тоже может быть валидным результатом и должен отображаться как "нет объектов", а не как ошибка.
