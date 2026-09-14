# Интеграция frontend с Synapse через gMART

Документ описывает публичный контракт frontend с Synapse-шлюзом gMART: получение
доступных workflow и run configuration, создание нового диалога, отправку follow-up,
чтение событий, восстановление соединения, отмену и загрузку истории.

Архитектура и серверная часть интеграции описаны в
[synapse-integration.md](../synapse-integration.md). Общий REST/SSE-контракт gMART —
в [frontend-service.md](../frontend-service.md).

## 1. Основной принцип

Frontend работает только с gMART и ChatStorage. Браузер не должен обращаться к
Synapse напрямую и не должен получать email, password или access token технического
пользователя Synapse.

```text
Browser ── user Bearer token ──► gMART /synapse/*
                                  │
                                  └── technical user ──► Synapse /api/*

Browser ── user Bearer token ──► ChatStorage /api/v1/*?space=synapse
```

Каталог workflow и run configuration ограничен tenant технического пользователя
gMART. Сейчас это общий каталог для всех пользователей gMART, а не персональный
каталог Synapse каждого пользователя.

Инварианты диалога:

- новый gMART chat создаёт новый Synapse project;
- один chat в `space=synapse` связан ровно с одним Synapse project;
- workflow и run configuration выбираются до первого запроса;
- для существующего chat выбор закреплён и не меняется;
- follow-up продолжает существующий Synapse project.

## 2. Эндпоинты

| Метод | Путь | Назначение |
| --- | --- | --- |
| `GET` | `/synapse/available` | Проверить, включена ли интеграция |
| `GET` | `/synapse/configurations` | Получить доступные workflow и run configuration |
| `POST` | `/synapse/runs` | Создать project или отправить follow-up |
| `GET` | `/synapse/runs/{request_id}` | Получить состояние запуска |
| `GET` | `/synapse/runs/{request_id}/events?after={cursor}` | Читать нормализованные события через SSE |
| `POST` | `/synapse/runs/{request_id}/cancel` | Остановить выполнение |

Все эндпоинты, кроме `/synapse/available`, требуют пользовательский токен:

```http
Authorization: Bearer <user-access-token>
```

`request_id` принадлежит пользователю, который запустил запрос. Для чужого или
неизвестного `request_id` API возвращает `404`, не раскрывая наличие запуска.

## 3. Проверка доступности

При загрузке приложения вызовите:

```http
GET /synapse/available
```

Ответ:

```json
{ "enabled": true }
```

Если `enabled=false` или запрос недоступен, не показывайте Synapse среди доступных
оркестраторов. Этот запрос не подтверждает доступность самого Synapse — только то,
что интеграция включена в конфигурации gMART.

## 4. Загрузка workflow и run configuration

После авторизации пользователя и выбора режима Synapse вызовите:

```http
GET /synapse/configurations
Authorization: Bearer <user-access-token>
```

Пример ответа:

```json
{
  "workflows": [
    {
      "id": "01991d22-workflow",
      "name": "idu_orchestrator",
      "display_name": "IDU orchestrator",
      "description": "Проверка территории агентами gMART",
      "execution_mode": "dynamic",
      "is_default": true
    }
  ],
  "run_configurations": [
    {
      "id": "01991d22-run-config",
      "name": "IDU default",
      "description": "Основная конфигурация моделей",
      "is_default": true
    }
  ],
  "default_workflow_id": "01991d22-workflow",
  "default_run_config_id": "01991d22-run-config"
}
```

Алгоритм выбора начального значения:

1. Использовать ранее сохранённый пользователем ID, если он всё ещё есть в каталоге.
2. Иначе использовать `default_workflow_id` / `default_run_config_id`.
3. Если default отсутствует — выбрать первый элемент списка.
4. Если один из списков пуст, заблокировать создание нового чата и показать ошибку.

Для подписи workflow используйте `display_name`, затем `name`. В запрос всегда
передавайте поле `id`, а не отображаемое имя.

Выбор можно сохранить в `localStorage` как предпочтение для следующего нового
диалога. Это не источник истины для существующего диалога: его значения нужно
восстанавливать из ChatStorage metadata.

## 5. Создание нового диалога

Для нового диалога `chat_id` должен быть `null`, а workflow и run configuration
передаются явно:

```http
POST /synapse/runs
Authorization: Bearer <user-access-token>
Idempotency-Key: 91c585f7-b4d4-4a91-b56d-fc73cf630d72
Content-Type: application/json
```

```json
{
  "request": "Проверь ограничения для выбранной территории",
  "chat_id": null,
  "scenario_id": 772,
  "project_id": 42,
  "workflow_id": "01991d22-workflow",
  "run_config_id": "01991d22-run-config",
  "metadata": {
    "selected_object_ids": [1001, 1002],
    "selected_layer_ids": [15]
  }
}
```

Поля запроса:

| Поле | Тип | Обязательность | Описание |
| --- | --- | --- | --- |
| `request` | `string` | да | Текст пользователя, от 1 до 50 000 символов |
| `chat_id` | `string \| null` | нет | `null` для нового chat; UUID ChatStorage для follow-up |
| `scenario_id` | `integer` | да | ID сценария Urban API |
| `project_id` | `integer \| null` | нет | ID проекта Urban API |
| `workflow_id` | `string \| null` | для нового chat | Выбранный ID workflow |
| `run_config_id` | `string \| null` | для нового chat | Выбранный ID run configuration |
| `metadata.selected_object_ids` | `array` | нет | Выбранные объекты интерфейса |
| `metadata.selected_layer_ids` | `array` | нет | Выбранные слои интерфейса |

Backend принимает отсутствие `workflow_id`/`run_config_id`, только если для них
заданы серверные defaults. Frontend не должен зависеть от этого fallback и должен
отправлять пользовательский выбор явно.

Ответ `202 Accepted`:

```json
{
  "request_id": "74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e",
  "chat_id": "e98ea3f9-9c75-40ba-86cf-26655f16cd8e",
  "synapse_project_id": "01991d22-project",
  "run_id": "01991d22-run",
  "workflow_id": "01991d22-workflow",
  "run_config_id": "01991d22-run-config",
  "status": "running",
  "events_url": "/synapse/runs/74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e/events"
}
```

Сразу сохраните `request_id`, `chat_id` и выбранные ID. После ответа откройте SSE
по `events_url` с cursor `0-0`.

## 6. Idempotency-Key

`Idempotency-Key` обязателен для каждого `POST /synapse/runs`.

Правила frontend:

- генерировать новый UUID для нового логического сообщения;
- сохранить ключ до получения определённого результата запуска;
- при сетевой ошибке повторять тот же body с тем же ключом;
- при `status=start_unknown` не генерировать новый ключ для того же body;
- после успешного запуска удалить сохранённый ключ;
- никогда не использовать один ключ для разных body — сервер вернёт `409`.

Удобно хранить `{ signature, idempotencyKey }`, где `signature` — стабильная
сериализация всего body. Если body изменился, требуется новый ключ.

## 7. SSE-поток событий

Для чтения событий используйте `fetch`, а не стандартный `EventSource`: необходимо
передавать заголовок `Authorization`.

```http
GET /synapse/runs/{request_id}/events?after=0-0
Authorization: Bearer <user-access-token>
Accept: text/event-stream
```

Пример frame:

```text
id: 1724508000123-0
event: synapse_event
data: {"type":"synapse_event","source_type":"phase.planning.started","source_event_id":"01991d22-event","stream_id":"1724508000123-0","request_id":"74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e","synapse_project_id":"01991d22-project","run_id":"01991d22-run","timestamp":"2026-09-08T12:00:00Z","content":{}}
```

Нормализованное событие:

```ts
type SynapseEvent = {
  type: "synapse_event";
  source_type: string;
  source_event_id: string;
  stream_id?: string | null;
  request_id: string;
  synapse_project_id: string;
  run_id?: string | null;
  timestamp?: string | null;
  content: Record<string, unknown>;
};
```

Не смешивайте идентификаторы:

- SSE `id` / `data.stream_id` — cursor Redis Stream для reconnect;
- `source_event_id` — ID исходного события Synapse для дедупликации;
- `request_id` — ID операции gMART;
- `run_id` — ID выполнения внутри Synapse.

Сохраняйте последний полностью обработанный SSE `id` после каждого события.
Строки heartbeat вида `: heartbeat` не содержат `data` и должны игнорироваться.

`source_type` расширяемый: frontend не должен падать на неизвестном значении.
Минимально рекомендуется обрабатывать:

| `source_type` | Действие frontend |
| --- | --- |
| `*.text.delta` | Дописать текст из `content` к текущему ответу |
| `message_appended`, `a2a_message` | Показать итоговое сообщение, кроме `role=user` |
| `agent.delegation.started`, `a2a_agent_started` | Показать передачу задачи агенту gMART |
| `agent.delegation.completed`, `a2a_agent_completed` | Завершить статус делегирования |
| `phase.*` | Показать текущую фазу без жёсткого списка фаз |
| `project_completed` | Завершить run успешно |
| `project_failed` | Завершить run с ошибкой |
| `project_stopped` | Отметить run отменённым |

GeoJSON `FeatureCollection` может находиться на любой вложенности `content`.
Frontend может рекурсивно искать объекты с `type="FeatureCollection"` и добавлять
их на карту. Не интерпретируйте произвольный внутренний payload как HTML.

## 8. Восстановление после разрыва или перезагрузки

Во время активного запуска сохраните локально:

```ts
type SynapseResumeState = {
  requestId: string;
  cursor: string;
  chatId?: string;
  question?: string;
};
```

После перезагрузки:

1. Получить свежий пользовательский access token.
2. Вызвать `GET /synapse/runs/{requestId}`.
3. Если status активный, открыть SSE с сохранённым `cursor`.
4. Если локального cursor нет, использовать `last_stream_id` из status response.
5. На сетевом разрыве повторять подключение с последним обработанным cursor.
6. После terminal status удалить локальное resume-состояние.

Активные статусы: `starting`, `running`, `start_unknown`.
Terminal-статусы: `done`, `failed`, `cancelled`.

Рекомендуемый backoff для SSE: `400 ms`, затем удваивать до `5 s`. Перед каждым
переподключением обновляйте access token. Повторное подключение не создаёт новый
Synapse project и не требует нового `POST /synapse/runs`.

Status response дополнительно содержит:

```json
{
  "last_event_id": "01991d22-event",
  "last_stream_id": "1724508000123-0",
  "error": null,
  "started_at": "2026-09-08T12:00:00Z",
  "finished_at": null
}
```

Если сохранённый browser cursor старше `last_stream_id`, используйте именно
browser cursor: так frontend повторно получит ещё не обработанные им события.

## 9. Follow-up существующего диалога

История chat содержит metadata:

```json
{
  "agent_id": "synapse",
  "provider": "synapse",
  "synapse_project_id": "01991d22-project",
  "synapse_workflow_id": "01991d22-workflow",
  "synapse_run_config_id": "01991d22-run-config"
}
```

При открытии chat восстановите selectors из `synapse_workflow_id` и
`synapse_run_config_id` и заблокируйте их. Для follow-up отправьте `chat_id`:

```json
{
  "request": "Теперь сравни результат с соседним участком",
  "chat_id": "e98ea3f9-9c75-40ba-86cf-26655f16cd8e",
  "scenario_id": 772,
  "project_id": 42,
  "workflow_id": "01991d22-workflow",
  "run_config_id": "01991d22-run-config",
  "metadata": {}
}
```

Можно не передавать два configuration ID в follow-up, но если они переданы, то
должны совпадать с metadata chat. Попытка изменить их вернёт `409`.

Если сохранённая конфигурация больше не присутствует в каталоге, покажите её ID
как недоступное закреплённое значение. Это не мешает отправить follow-up в уже
существующий Synapse project.

## 10. Отмена

```http
POST /synapse/runs/{request_id}/cancel
Authorization: Bearer <user-access-token>
```

После ответа закройте локальный SSE, сбросьте active state и удалите resume-запись.
Ожидаемый status — `cancelled`. Позднее дублирующее событие `project_stopped`
следует обрабатывать идемпотентно.

## 11. История ChatStorage

Synapse-история отделена от обычной истории gMART параметром `space=synapse`.

```http
GET {CHAT_STORAGE}/api/v1/chat_history/chats?limit=100&offset=0&space=synapse
Authorization: Bearer <user-access-token>
```

```http
GET {CHAT_STORAGE}/api/v1/chat_history/{chat_id}?space=synapse&message_limit=50
Authorization: Bearer <user-access-token>
```

```http
DELETE {CHAT_STORAGE}/api/v1/chat_history/{chat_id}?space=synapse
Authorization: Bearer <user-access-token>
```

Live-ответ строится из SSE. После terminal event обновите список чатов, а после
перезагрузки используйте ChatStorage как устойчивый источник истории. Delta-события
в ChatStorage не сохраняются по одному; там находятся итоговые сообщения и
устойчивые структурированные события.

## 12. Минимальный TypeScript-клиент

```ts
async function api<T>(path: string, token: string, init: RequestInit = {}) {
  const agentsUrl = AGENTS_URL.replace(/\/+$/, "");
  const response = await fetch(`${agentsUrl}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...init.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

const configurations = await api<SynapseConfigurationOptions>(
  "/synapse/configurations",
  token,
);

const run = await api<SynapseRun>("/synapse/runs", token, {
  method: "POST",
  headers: { "Idempotency-Key": crypto.randomUUID() },
  body: JSON.stringify({
    request,
    chat_id: null,
    scenario_id: 772,
    project_id: 42,
    workflow_id: selectedWorkflowId,
    run_config_id: selectedRunConfigId,
    metadata: {},
  }),
});
```

Для SSE используйте потоковый parser, который сохраняет неполный frame между
сетевыми chunks. Нельзя считать, что один `reader.read()` содержит ровно одно
SSE-событие.

## 13. Ошибки

| HTTP status | Причина | Реакция frontend |
| --- | --- | --- |
| `401` | Токен отсутствует или невалиден | Обновить токен или открыть login |
| `404` | Run не существует или принадлежит другому пользователю | Удалить resume-state |
| `409` | Активный run на chat, конфликт idempotency или смена конфигурации | Не повторять с новым ключом автоматически; показать конфликт |
| `422` | Ошибка body либо не выбраны workflow/run configuration | Подсветить форму |
| `502` | Synapse/ChatStorage временно недоступен | Сохранить idempotency state и предложить безопасный retry |

Не показывайте пользователю raw credentials, внутренние stack traces или содержимое
технического токена. Для неизвестного события сохраняйте техническую запись в
диагностике, но не завершайте весь UI с ошибкой.

## 14. Чек-лист frontend

- Synapse скрыт при `enabled=false`.
- Каталог загружается только после пользовательской авторизации.
- Новый chat нельзя запустить без двух выбранных ID.
- В API передаются ID, а не display name.
- Selectors заблокированы для существующего chat.
- `Idempotency-Key` переживает неопределённый результат POST.
- SSE открывается через `fetch` с Bearer token.
- Cursor обновляется после каждого полностью обработанного frame.
- Неизвестные `source_type` не ломают поток.
- Terminal event закрывает SSE и обновляет историю.
- После reload активный run восстанавливается через status + cursor.
- История читается только с `space=synapse`.
