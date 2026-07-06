# gMART API documentation

> Bilingual reference for the gMART **agents** service, focused on the **A2A**
> (agent-to-agent) protocol surface. English first, **[Русская версия ниже](#русская-версия)**.

---

## English version

### 1. Overview

gMART ships two deployable apps from one codebase:

| App | Port | Purpose |
|---|---|---|
| `src/agents` | `80` | Agents REST API + **A2A** endpoints + SSE streaming |
| `src/idu_mcp` | `8000` | IDU MCP server (geometry + Urban API tools) |

This document covers the **A2A** surface of the agents app. The REST/SSE surface for the
frontend is documented in [`frontend-service.md`](frontend-service.md) and
[`frontend-document-qa.md`](frontend-document-qa.md).

The agents app exposes **three A2A agents**, each as a JSON-RPC 2.0 endpoint with an
A2A AgentCard for discovery:

| Agent | Card `name` | JSON-RPC endpoint | Needs `scenario_id` |
|---|---|---|---|
| Restriction creation | `restriction-creation-agent` | `POST /restriction/a2a` | **required** |
| Provision effects | `provision-effects-agent` | `POST /provision/a2a` | **required** |
| Document QA (RAG) | `document-qa-agent` | `POST /documents/a2a` | optional |

Protocol: **A2A 0.3.0**, transport **JSONRPC**. The agents also accept the 1.0 method
binding names (`SendMessage`, `GetTask`, …) as aliases, but responses are serialized in the
0.3 shape.

### 2. Discovery — the AgentCard

Each agent publishes a card (no auth required):

| Agent | Card URL |
|---|---|
| Restriction | `GET /restriction/.well-known/agent-card.json` |
| Provision | `GET /provision/.well-known/agent-card.json` |
| Document QA | `GET /documents/.well-known/agent-card.json` |

Legacy aliases also resolve: `GET /.well-known/agent-card.json`,
`GET /.well-known/agent.json`, `GET /restriction/agent.json`.

Relevant fields:

```jsonc
{
  "name": "restriction-creation-agent",
  "protocolVersion": "0.3.0",
  "preferredTransport": "JSONRPC",
  "url": "http://<host>/restriction/a2a",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateTransitionHistory": true,
    "extensions": [
      {
        "uri": "https://github.com/IDUclub/gMART/a2a/extensions/scenario-context/v1",
        "required": true,
        "params": {
          "type": "object",
          "required": ["scenario_id"],
          "properties": { "scenario_id": { "type": "integer" } }
        }
      }
    ]
  },
  "defaultInputModes": ["text/plain", "application/json"],
  "defaultOutputModes": ["text/plain", "application/vnd.geo+json", "application/geo+json"],
  "skills": [
    {
      "id": "create-geospatial-restrictions",
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["text/plain", "application/vnd.geo+json", "application/geo+json"]
    }
  ]
}
```

#### The `scenario-context` extension

The restriction and provision agents require a project **`scenario_id`** on every request.
This requirement is declared as an A2A **Profile Extension** so it is discoverable from the
card rather than learned from a runtime error:

- `uri`: `https://github.com/IDUclub/gMART/a2a/extensions/scenario-context/v1`
  (an opaque, stable identifier — it is never fetched over HTTP).
- `required: true` — clients must supply `scenario_id`.
- `params` — a JSON Schema describing the expected `scenario_id` (integer).

Clients may activate it via the `A2A-Extensions: <uri>` HTTP header and list the URI in
`message.extensions`, but activation is **not required** for the request to work — the agent
reads `scenario_id` from the structured channels below regardless.

The document-QA agent does **not** declare this extension (`scenario_id` is optional there).

### 3. Supported JSON-RPC methods

| Method (0.3) | 1.0 alias | Behavior |
|---|---|---|
| `message/send` | `SendMessage`, `tasks/send` | Run the pipeline to completion; return the final `Task`. |
| `message/stream` | `SendStreamingMessage`, `tasks/sendSubscribe` | Run the pipeline; stream events over SSE. |
| `tasks/get` | `GetTask` | Fetch a stored task by id. |
| `tasks/list` | `ListTasks` | List stored tasks (`includeArtifacts` param). |
| `tasks/cancel` | `CancelTask` | Mark a non-terminal task canceled. |
| `agent/getAuthenticatedExtendedCard` | `GetExtendedAgentCard` | Return the agent card. |

### 4. Passing `scenario_id` (and other parameters)

`scenario_id` is resolved from the **first** source that provides it, in this order:

1. **DataPart** — the portable, recommended channel:
   ```json
   "parts": [
     { "kind": "data", "data": { "scenario_id": 772 } },
     { "kind": "text", "text": "построй зону вокруг школ 200 м" }
   ]
   ```
2. **`message.metadata`** — `{ "scenario_id": 772 }`.
3. **`params.metadata`** — `{ "scenario_id": 772 }`.
4. **Inline in the message text** — `"scenario_id=772 ..."` (backward-compatible fallback,
   kept for migration; the inline id is stripped from the query forwarded to the LLM).

If `scenario_id` is supplied through a structured channel **and** inline in text, the
**structured value wins**. Both `scenario_id` and `scenarioId` keys are accepted.

Optional request fields (same merge order — DataPart / metadata / params): `model`,
`temperature`. The document-QA agent additionally accepts `chat_id` / `chatId`.

Incoming parts may use either `{"kind": ...}` (0.3) or `{"type": ...}` (legacy) — both are
accepted on input.

#### Example — `message/send`

```bash
curl -X POST http://<host>/restriction/a2a \
  -H 'Content-Type: application/json' \
  -H 'A2A-Extensions: https://github.com/IDUclub/gMART/a2a/extensions/scenario-context/v1' \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "messageId": "client-001",
        "parts": [
          { "kind": "data", "data": { "scenario_id": 772 } },
          { "kind": "text", "text": "построй зону вокруг школ 200 м" }
        ]
      },
      "configuration": { "historyLength": 0 }
    }
  }'
```

### 5. Response format

`message/send` returns a JSON-RPC envelope whose `result` is an A2A **Task**:

```jsonc
{
  "jsonrpc": "2.0",
  "id": "1",
  "result": {
    "kind": "task",
    "id": "…",
    "contextId": "…",
    "status": {
      "state": "completed",                       // working | completed | failed | canceled
      "timestamp": "2026-06-26T11:48:08.022614Z", // RFC3339, always with a 'Z' offset
      "message": { "kind": "message", "messageId": "…", "role": "agent", "parts": [...] }
    },
    "history": [
      { "kind": "message", "messageId": "client-001", "role": "user",  "parts": [...] },
      { "kind": "message", "messageId": "…",          "role": "agent", "parts": [...] }
    ],
    "artifacts": [
      {
        "artifactId": "restriction-agent-text",
        "name": "restriction-agent-response",
        "parts": [ { "kind": "text", "text": "…" } ]
      },
      {
        "artifactId": "geojson-schools",
        "name": "schools",
        "parts": [
          {
            "kind": "data",
            "data": { "type": "FeatureCollection", "features": [ ... ] },
            "metadata": { "mediaType": "application/vnd.geo+json" }
          }
        ],
        "metadata": { "layerName": "schools", "mediaType": "application/vnd.geo+json" }
      }
    ]
  }
}
```

Guarantees (verified against the official `a2a-sdk` v0.3 strict models):

- **Every `Message` carries `messageId`** — including the echoed user message in `history[0]`.
  The client's original `messageId` is preserved when present, otherwise generated.
- **`status.timestamp` is RFC3339 with a timezone** (`…Z`).
- **Message/artifact parts use the `kind` discriminator** (`{"kind": "text"}`,
  `{"kind": "data"}`). GeoJSON payloads under a data part keep their own `type`
  (`FeatureCollection` / `Feature`).
- Spatial layers are GeoJSON `FeatureCollection` in WGS84 (EPSG:4326).

`configuration.historyLength` (on `message/send`) and `historyLength` (on `tasks/get`) trim
the returned `history` to the most recent N messages; `0` drops it entirely.

### 6. Streaming (`message/stream`)

Returns an SSE stream (`Content-Type: text/event-stream`). Each SSE `data:` frame is a
JSON-RPC envelope whose `result` is one incremental event:

| `result` shape | Meaning |
|---|---|
| `{ "kind": "task", id, contextId, status, … }` | Initial task snapshot (first frame). |
| `{ "kind": "status-update", taskId, contextId, status, final }` | Status transition. |
| `{ "kind": "artifact-update", taskId, contextId, artifact, append, lastChunk }` | New / appended artifact (text chunks use `append: true`). |

Events follow the A2A 0.3 flat discriminated union (`kind` on the top level of `result`),
so each SSE frame validates against the official SDK's `SendStreamingMessageResponse`.

A terminal `status-update` (`final: true`) is **always** emitted — `completed` on success,
`failed` on error. This holds even when the request is rejected before the pipeline starts
(e.g. missing `scenario_id`): the stream emits a terminal `failed` status-update **and** a
JSON-RPC error frame, so a spec-compliant streaming client never hangs on an empty stream.

### 7. Error codes

Errors use standard JSON-RPC codes:

| Code | Meaning | When |
|---|---|---|
| `-32600` | Invalid Request | `jsonrpc != "2.0"`, non-object payload. |
| `-32601` | Method not found | Unknown method. |
| `-32602` | Invalid params | Missing/invalid `scenario_id`, missing message text, missing task id, non-object `params`. |
| `-32001` | Task not found (A2A `TaskNotFoundError`) | `tasks/get` / `tasks/cancel` on an unknown id. |
| `-32004` | Unsupported operation (A2A `UnsupportedOperationError`) | A streaming method sent to the non-streaming path. |
| `-32000` | Server error | Unexpected pipeline error. |

Missing `scenario_id` returns `-32602` with a message naming the field and the accepted
channels (DataPart / metadata / inline text).

### 8. Compatibility summary

Changes made for A2A 0.3 / official-SDK compatibility (all three agents):

- `scenario_id` read from DataPart / `message.metadata` / `params.metadata`, with inline-text
  fallback retained for backward compatibility.
- Missing/invalid params → `-32602` (was `-32000`).
- `messageId` on every `Message`, including the echoed user message.
- `status.timestamp` in RFC3339 with a `Z` offset.
- Part discriminator `type` → `kind` on all outgoing messages/artifacts.
- Required, discoverable `scenario-context` extension on the restriction/provision cards.
- `application/json` advertised in `defaultInputModes` / skill `inputModes`.
- `configuration.historyLength` / `historyLength` honored.
- Streaming error path emits a terminal `failed` event (no empty streams).

---

## Русская версия

### 1. Обзор

gMART состоит из двух разворачиваемых приложений в одной кодовой базе:

| Приложение | Порт | Назначение |
|---|---|---|
| `src/agents` | `80` | REST API агентов + **A2A**-эндпоинты + SSE-стриминг |
| `src/idu_mcp` | `8000` | IDU MCP-сервер (геометрия + инструменты Urban API) |

Этот документ описывает **A2A**-интерфейс приложения agents. REST/SSE-контракт для фронтенда
описан в [`frontend-service.md`](frontend-service.md) и
[`frontend-document-qa.md`](frontend-document-qa.md).

Приложение agents предоставляет **три A2A-агента**, каждый — это эндпоинт JSON-RPC 2.0 с
карточкой агента (AgentCard) для обнаружения:

| Агент | `name` карточки | Эндпоинт JSON-RPC | Нужен `scenario_id` |
|---|---|---|---|
| Построение ограничений | `restriction-creation-agent` | `POST /restriction/a2a` | **обязателен** |
| Эффекты обеспеченности | `provision-effects-agent` | `POST /provision/a2a` | **обязателен** |
| QA по документам (RAG) | `document-qa-agent` | `POST /documents/a2a` | опционален |

Протокол: **A2A 0.3.0**, транспорт **JSONRPC**. Агенты также принимают имена методов из
биндинга 1.0 (`SendMessage`, `GetTask`, …) как алиасы, но ответы сериализуются в формате 0.3.

### 2. Обнаружение — AgentCard

Каждый агент публикует карточку (без авторизации):

| Агент | URL карточки |
|---|---|
| Ограничения | `GET /restriction/.well-known/agent-card.json` |
| Обеспеченность | `GET /provision/.well-known/agent-card.json` |
| QA по документам | `GET /documents/.well-known/agent-card.json` |

Также работают устаревшие алиасы: `GET /.well-known/agent-card.json`,
`GET /.well-known/agent.json`, `GET /restriction/agent.json`.

Значимые поля карточки — см. JSON-пример в английской части (раздел 2). Ключевые элементы:
`protocolVersion: "0.3.0"`, `preferredTransport: "JSONRPC"`, `capabilities.streaming: true`,
объявление расширения `scenario-context` и `defaultInputModes`, включающий `application/json`.

#### Расширение `scenario-context`

Агенты ограничений и обеспеченности требуют идентификатор сценария проекта **`scenario_id`**
в каждом запросе. Это требование объявлено как **Profile Extension** протокола A2A, чтобы оно
было **обнаруживаемым** из карточки, а не узнавалось из рантайм-ошибки:

- `uri`: `https://github.com/IDUclub/gMART/a2a/extensions/scenario-context/v1`
  (непрозрачный стабильный идентификатор; по HTTP не запрашивается).
- `required: true` — клиент обязан передать `scenario_id`.
- `params` — JSON Schema, описывающая `scenario_id` (целое число).

Клиент может активировать расширение заголовком `A2A-Extensions: <uri>` и перечислить URI в
`message.extensions`, но активация **не обязательна** для работы запроса — агент в любом случае
читает `scenario_id` из структурированных каналов ниже.

Агент QA по документам это расширение **не** объявляет (там `scenario_id` опционален).

### 3. Поддерживаемые методы JSON-RPC

| Метод (0.3) | Алиас 1.0 | Поведение |
|---|---|---|
| `message/send` | `SendMessage`, `tasks/send` | Выполнить пайплайн целиком; вернуть финальный `Task`. |
| `message/stream` | `SendStreamingMessage`, `tasks/sendSubscribe` | Выполнить пайплайн; стримить события по SSE. |
| `tasks/get` | `GetTask` | Получить сохранённую задачу по id. |
| `tasks/list` | `ListTasks` | Список задач (параметр `includeArtifacts`). |
| `tasks/cancel` | `CancelTask` | Отменить незавершённую задачу. |
| `agent/getAuthenticatedExtendedCard` | `GetExtendedAgentCard` | Вернуть карточку агента. |

### 4. Передача `scenario_id` (и других параметров)

`scenario_id` берётся из **первого** источника, где он найден, в таком порядке:

1. **DataPart** — переносимый, рекомендуемый канал:
   ```json
   "parts": [
     { "kind": "data", "data": { "scenario_id": 772 } },
     { "kind": "text", "text": "построй зону вокруг школ 200 м" }
   ]
   ```
2. **`message.metadata`** — `{ "scenario_id": 772 }`.
3. **`params.metadata`** — `{ "scenario_id": 772 }`.
4. **Внутри текста сообщения** — `"scenario_id=772 ..."` (обратная совместимость на период
   миграции; инлайн-id вырезается из запроса, передаваемого в LLM).

Если `scenario_id` передан и структурно, и в тексте — **приоритет у структурного значения**.
Принимаются ключи `scenario_id` и `scenarioId`.

Опциональные поля запроса (тот же порядок слияния — DataPart / metadata / params): `model`,
`temperature`. Агент QA дополнительно принимает `chat_id` / `chatId`.

Во **входящих** частях допускается и `{"kind": ...}` (0.3), и `{"type": ...}` (legacy) —
оба принимаются.

Пример запроса `message/send` — см. `curl` в английской части (раздел 4).

### 5. Формат ответа

`message/send` возвращает JSON-RPC-конверт, где `result` — это A2A **Task** (полный пример
JSON см. в английской части, раздел 5). Гарантии (проверены по строгим моделям v0.3
официального `a2a-sdk`):

- **У каждого `Message` есть `messageId`** — включая эхо пользовательского сообщения в
  `history[0]`. Исходный `messageId` клиента сохраняется, иначе генерируется новый.
- **`status.timestamp` — RFC3339 с таймзоной** (`…Z`).
- **Части сообщений/артефактов используют дискриминатор `kind`** (`{"kind": "text"}`,
  `{"kind": "data"}`). GeoJSON внутри data-части сохраняет собственный `type`
  (`FeatureCollection` / `Feature`).
- Геослои — GeoJSON `FeatureCollection` в WGS84 (EPSG:4326).

`configuration.historyLength` (для `message/send`) и `historyLength` (для `tasks/get`)
обрезают возвращаемую `history` до последних N сообщений; `0` убирает её полностью.

### 6. Стриминг (`message/stream`)

Возвращает SSE-поток (`Content-Type: text/event-stream`). Каждый кадр `data:` — это
JSON-RPC-конверт, где `result` — одно инкрементальное событие:

| Форма `result` | Значение |
|---|---|
| `{ "kind": "task", id, contextId, status, … }` | Начальный снимок задачи (первый кадр). |
| `{ "kind": "status-update", taskId, contextId, status, final }` | Смена статуса. |
| `{ "kind": "artifact-update", taskId, contextId, artifact, append, lastChunk }` | Новый/дополненный артефакт (текстовые чанки — с `append: true`). |

События следуют плоскому дискриминированному союзу A2A 0.3 (`kind` на верхнем уровне
`result`), поэтому каждый SSE-кадр проходит валидацию официальным SDK
(`SendStreamingMessageResponse`).

Терминальный `status-update` (`final: true`) эмитится **всегда** — `completed` при успехе,
`failed` при ошибке. Это верно даже когда запрос отклонён до старта пайплайна (например, нет
`scenario_id`): поток отдаёт терминальный `failed`-статус **и** кадр с JSON-RPC-ошибкой, так
что спецификационный стриминг-клиент не зависает на пустом потоке.

### 7. Коды ошибок

| Код | Значение | Когда |
|---|---|---|
| `-32600` | Invalid Request | `jsonrpc != "2.0"`, не-объектный payload. |
| `-32601` | Method not found | Неизвестный метод. |
| `-32602` | Invalid params | Нет/некорректный `scenario_id`, нет текста сообщения, нет id задачи, `params` не объект. |
| `-32001` | Task not found (A2A `TaskNotFoundError`) | `tasks/get` / `tasks/cancel` по неизвестному id. |
| `-32004` | Unsupported operation (A2A `UnsupportedOperationError`) | Стриминговый метод отправлен в нестриминговый путь. |
| `-32000` | Server error | Непредвиденная ошибка пайплайна. |

Отсутствие `scenario_id` возвращает `-32602` с сообщением, называющим поле и принимаемые
каналы (DataPart / metadata / текст).

### 8. Сводка изменений совместимости

Изменения для совместимости с A2A 0.3 / официальным SDK (во всех трёх агентах):

- `scenario_id` читается из DataPart / `message.metadata` / `params.metadata`, с фолбэком на
  текст для обратной совместимости.
- Отсутствие/некорректность параметров → `-32602` (было `-32000`).
- `messageId` у каждого `Message`, включая эхо пользовательского сообщения.
- `status.timestamp` в RFC3339 с офсетом `Z`.
- Дискриминатор частей `type` → `kind` во всех исходящих сообщениях/артефактах.
- Обязательное, обнаруживаемое расширение `scenario-context` на карточках restriction/provision.
- `application/json` объявлен в `defaultInputModes` / `inputModes` навыка.
- Учитывается `configuration.historyLength` / `historyLength`.
- Путь ошибок стриминга эмитит терминальное `failed`-событие (нет пустых потоков).
