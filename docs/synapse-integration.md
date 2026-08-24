# Интеграция Synapse как альтернативного оркестратора

## 1. Цель и ограничения

Добавить в gMART альтернативную точку входа, через которую frontend:

1. отправляет пользовательский запрос и городской контекст;
2. запускает или продолжает проект в Synapse;
3. получает события Synapse в реальном времени;
4. видит сохранённую историю в ChatStorage в пространстве `synapse`.

Ограничения решения:

- оркестратор, workflow и A2A-серверы уже настроены в Synapse;
- код Synapse изменять нельзя;
- Synapse вызывает A2A endpoints gMART с сервисным OAuth2-токеном;
- пользовательские и сервисные токены нельзя помещать в prompt, события, ChatStorage
  или логи;
- существующий оркестратор gMART и его endpoint должны продолжить работать без изменений.

## 2. Итоговая архитектура

```text
Frontend
   │ 1. User Bearer
   ▼
gMART /synapse/runs
   │
   ├── извлекает user_id и формирует prompt
   ├── создаёт/продолжает Synapse project через технического пользователя Synapse
   ├── сохраняет связь request/chat/project/user в Redis
   └── запускает фоновый relay
            │
            ├── Synapse SSE ──► Redis Stream ──► Frontend SSE
            └── нормализованные события ──────► ChatStorage space=synapse

Synapse orchestrator
   │ OAuth2 client_credentials
   ▼
gMART A2A endpoints
   └── gMART service auth ──► MCP / Urban API / другие IDU-сервисы
```

В этой схеме исходный пользовательский access token не должен жить дольше входного
HTTP-запроса. gMART сохраняет только стабильный `user_id`. Для обращения к ChatStorage
и другим IDU-сервисам используется сервисный токен gMART вместе с `X-User-Id`.

В MVP сами A2A-агенты выполняются с машинной identity сервисного клиента Synapse. Если
Urban API или другой downstream обязан проверять права исходного пользователя, это
оформляется отдельным расширением: gMART разрешает `metadata.project_id` в исходный
`user_id` и вызывает downstream своим сервисным токеном с этим `X-User-Id`. Передавать
исходный user access token через Synapse для этого не требуется.

## 3. Авторизация

### 3.1. Frontend → gMART

Frontend продолжает передавать пользовательский Keycloak-токен:

```http
Authorization: Bearer <user-access-token>
```

gMART получает из него `sub` и использует это значение как `user_id`. В production
подпись, issuer и audience токена должны проверяться либо ingress, либо отдельной
Keycloak-aware dependency. Текущая `verify_bearer_token` только извлекает Bearer и не
проверяет его подпись.

### 3.2. gMART → Synapse API

Текущий Synapse не принимает Keycloak `client_credentials` как собственную
аутентификацию API. Поэтому без изменения Synapse нужно завести технического пользователя
в tenant, где находится настроенный оркестратор.

gMART использует штатные endpoints Synapse:

```text
POST /api/auth/login
POST /api/auth/refresh
```

`SynapseApiClient` должен кэшировать access/refresh token, обновлять access token до
истечения срока и один раз повторять запрос после `401`. Пароль, refresh token и access
token не должны попадать в `/system/config`, `repr`, логи или ChatStorage.

Если перед Synapse уже расположен ingress, который выдаёт подходящий технический токен,
клиент может принимать этот токен из secret/env вместо login/password.

### 3.3. Synapse → gMART A2A

В конфигурации каждого gMART A2A server в Synapse используется существующая поддержка
OAuth2 `client_credentials`:

```json
{
  "auth": {
    "type": "oauth2",
    "token_url": "https://keycloak.example/realms/idu/protocol/openid-connect/token",
    "client_id": "synapse-gmart",
    "client_secret_env": "SYNAPSE_GMART_CLIENT_SECRET",
    "grant_type": "client_credentials"
  }
}
```

Keycloak client `synapse-gmart` должен:

- иметь разрешение на A2A endpoints gMART;
- иметь audience, ожидаемый gMART/ingress;
- не иметь административных прав, не нужных для выполнения агентов;
- хранить secret только в secret manager или переменной окружения Synapse.

Для production нельзя полагаться только на текущую `verify_bearer_token`: она проверяет
наличие заголовка, но не валидность сервисного JWT. Проверку выполняет доверенный ingress
или отдельная dependency gMART с проверкой Keycloak JWKS, issuer и audience.

### 3.4. gMART → ChatStorage и другие IDU-сервисы

В gMART уже есть `KeycloakTokenClient`, автоматическое обновление сервисного токена и
заголовок `X-User-Id` в [service_auth.py](../src/common/service_auth.py).

Для фонового relay нельзя требовать исходный пользовательский access token. Нужно
расширить `JsonApiHandler` и `ChatStorageApiClient`, чтобы они принимали явный `user_id`:

```python
await chat_storage.add_parts_message(
    user_id=run.user_id,
    space="synapse",
    chat_id=run.chat_id,
    role="system",
    parts=parts,
    source_event_id=event.event_id,
)
```

На HTTP-границе клиент формирует:

```http
Authorization: Bearer <gmart-service-token>
X-User-Id: <original-user-id>
```

## 4. Публичный API gMART

Рекомендуется разделить запуск и чтение событий. Это позволяет продолжать relay после
закрытия браузера и безопасно переподключаться.

### 4.1. Запуск или продолжение

```http
POST /synapse/runs
Authorization: Bearer <user-token>
Content-Type: application/json
```

```json
{
  "request": "Проверь ограничения для выбранной территории",
  "chat_id": null,
  "scenario_id": 772,
  "project_id": 42,
  "metadata": {
    "selected_object_ids": [1001, 1002],
    "selected_layer_ids": [15]
  }
}
```

Ответ `202 Accepted`:

```json
{
  "request_id": "74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e",
  "chat_id": "e98ea3f9-9c75-40ba-86cf-26655f16cd8e",
  "synapse_project_id": "synapse-project-id",
  "status": "running",
  "events_url": "/synapse/runs/74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e/events"
}
```

Правила:

- `chat_id == null`: создать новый Synapse project, затем создать ChatStorage chat с
  `space=synapse` и metadata, содержащей `synapse_project_id`;
- `chat_id != null`: загрузить chat из `space=synapse`, получить
  `metadata.synapse_project_id` и отправить follow-up через Synapse messages API;
- существующий chat без `synapse_project_id` возвращает `409 synapse_mapping_missing`;
- одновременно допускается только один активный run на chat;
- повторный запрос с тем же idempotency key не создаёт второй проект/run.

### 4.2. Поток событий

```http
GET /synapse/runs/{request_id}/events?after=<redis-stream-id>
Authorization: Bearer <user-token>
Accept: text/event-stream
```

Каждый SSE frame содержит `id` и нормализованный JSON в `data`:

```text
id: 1724508000123-0
event: synapse_event
data: {"type":"synapse_event","source_event_id":"01991d22-7a04-7d93-9900-cc95d8db4f47","source_type":"agent.delegation.started","content":{...}}
```

SSE `id` — это cursor Redis Stream, используемый frontend при reconnect.
`source_event_id` — исходный UUIDv7 Synapse, используемый для дедупликации и
идемпотентной записи в ChatStorage.

Минимальный контракт нормализованного события:

```json
{
  "type": "synapse_event",
  "source_type": "agent.delegation.started",
  "source_event_id": "01991d22-7a04-7d93-9900-cc95d8db4f47",
  "stream_id": "1724508000123-0",
  "request_id": "74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e",
  "synapse_project_id": "synapse-project-id",
  "run_id": "synapse-run-id",
  "timestamp": "2026-08-24T14:00:00Z",
  "content": {}
}
```

Терминальные события:

- `project_completed` → `status=done`;
- `project_failed` → `status=failed`;
- `project_stopped` → `status=cancelled`.

### 4.3. Состояние run

```http
GET /synapse/runs/{request_id}
```

Endpoint нужен для восстановления UI после reload и должен возвращать status, chat id,
Synapse project/run id, последний event id и terminal error, если он есть.

## 5. Prompt contract

Prompt должен формироваться детерминированно, отдельно от HTTP-клиента. Рекомендуемый
формат:

```text
[IDU_CONTEXT_V1]
request_id=74cd1ed4-64a5-41c7-b29c-83a90a4d7c2e
scenario_id=772
urban_project_id=42
selected_object_ids=[1001,1002]
selected_layer_ids=[15]
[/IDU_CONTEXT_V1]

[USER_REQUEST]
Проверь ограничения для выбранной территории
[/USER_REQUEST]
```

`scenario_id` должен присутствовать явно: Synapse использует текст инструкции для
подготовки обязательного A2A extension DataPart.

На первом ходе `chat_id` ещё неизвестен: сначала создаётся Synapse project, затем
ChatStorage chat с `metadata.synapse_project_id`. Поэтому `chat_id` не требуется
оркестратору и не включается в первый prompt. На follow-up его можно добавить только как
технический correlation field, если это действительно нужно настроенному workflow.

Запрещено добавлять в prompt:

- `Authorization`;
- пользовательские или сервисные access/refresh tokens;
- Keycloak client secret;
- технические пароли Synapse;
- данные, не требующиеся оркестратору для выполнения задачи.

## 6. Клиент Synapse

Создать `src/agents/api_clients/synapse_client.py` со следующими операциями:

```python
class SynapseApiClient:
    async def create_project(self, prompt: str) -> SynapseProject: ...
    async def send_message(
        self, project_id: str, content: str, metadata: dict
    ) -> None: ...
    async def get_project(self, project_id: str) -> SynapseProjectState: ...
    async def stream_events(
        self,
        project_id: str,
        *,
        run_id: str | None,
        last_event_id: str | None,
    ) -> AsyncIterator[SynapseEvent]: ...
```

Используемые Synapse endpoints:

```text
POST /api/projects
POST /api/projects/{project_id}/messages?run_id=...
GET  /api/projects/{project_id}
GET  /api/projects/{project_id}/events?run_id=...
```

Требования к клиенту:

- единый `httpx.AsyncClient` на lifespan приложения;
- отдельные connect/read timeouts для обычных запросов и SSE;
- корректный разбор полей SSE `event`, `id`, `data`;
- `Last-Event-ID` при повторном подключении;
- bounded exponential backoff с jitter;
- один refresh/retry после `401`;
- отсутствие body/headers в логах при auth-ошибках;
- закрытие клиента в lifespan.

Можно добавить прямую зависимость `httpx-sse`, даже если она уже присутствует в lock-файле
как транзитивная зависимость.

## 7. Redis и фоновый relay

Создать отдельный `SynapseRunStore`, не смешивая Synapse-состояние с существующими
pipeline checkpoints.

Рекомендуемые ключи:

```text
synapse:run:{request_id}:state       HASH
synapse:run:{request_id}:events      STREAM
synapse:run:{request_id}:seen        SET
synapse:project:{project_id}         STRING -> request_id
synapse:chat:{chat_id}:active        STRING -> request_id
```

State содержит только:

```json
{
  "user_id": "keycloak-sub",
  "chat_id": "chat-uuid",
  "synapse_project_id": "project-id",
  "run_id": "run-id",
  "last_event_id": "uuidv7",
  "last_stream_id": "1724508000123-0",
  "status": "running",
  "started_at": "...",
  "finished_at": null,
  "error": null
}
```

Access/refresh tokens в Redis state не сохраняются.

Relay выполняет следующий цикл:

1. открывает Synapse event stream с последним cursor;
2. нормализует событие;
3. атомарно проверяет `source_event_id` через Redis `SET`/Lua или транзакцию;
4. добавляет событие в Redis Stream;
5. сохраняет представление события в ChatStorage;
6. обновляет `last_event_id` только после успешной обработки;
7. завершает run на terminal event;
8. при обрыве соединения переподключается с `Last-Event-ID`.

Redis Stream предпочтительнее комбинации list + pub/sub: он одновременно обеспечивает
буфер, cursor и ожидание новых сообщений через `XREAD`.

Для MVP relay может быть background task внутри Agents API. Для нескольких workers и
восстановления после рестарта необходимо добавить owner lock (`SET NX EX`) и startup
recovery активных run либо вынести relay в отдельный worker.

## 8. Запись в ChatStorage

Все операции выполняются с `space=synapse`:

```text
POST /api/v1/chat_history/create_chat
GET  /api/v1/chat_history/{chat_id}?space=synapse
POST /api/v1/chat_history/{chat_id}/message?space=synapse
```

При создании chat сохранить:

```json
{
  "space": "synapse",
  "title": "...",
  "scenario_id": 772,
  "project_id": 42,
  "metadata": {
    "provider": "synapse",
    "synapse_project_id": "project-id",
    "synapse_workflow_id": "configured-workflow"
  }
}
```

Не следует создавать отдельное пользовательское сообщение на каждый токен/delta LLM.
Frontend получает все live events, а ChatStorage хранит устойчивые события:

| Synapse event | ChatStorage representation |
|---|---|
| исходный запрос | `role=user`, `kind=text` |
| phase/status/delegation | `role=system`, `kind=status` или `kind=data` |
| tool call/result | `kind=tool_call` / `kind=data` |
| assistant message | `role=assistant`, `kind=text` |
| artifact | `kind=artifact_ref` |
| project failure | `kind=failure` |

В metadata каждого сохранённого сообщения добавить:

```json
{
  "provider": "synapse",
  "source_event_id": "uuidv7",
  "source_event_type": "agent.delegation.completed",
  "synapse_project_id": "project-id",
  "synapse_run_id": "run-id"
}
```

Redis `seen` защищает от повторов при обычном reconnect. Для строгой идемпотентности
после рестарта между записью в ChatStorage и подтверждением Redis необходимо расширить
ChatStorage полем `source_event_id` и уникальным индексом
`(user_id, space, chat_id, source_event_id)`. Без этого невозможно обеспечить exactly-once
между двумя независимыми системами; допустима только at-least-once доставка.

## 9. Изменения в gMART

### Новые файлы

```text
src/agents/api_clients/synapse_client.py
src/agents/dto/synapse_request_dto.py
src/agents/schema/synapse_response.py
src/agents/services/synapse_gateway_service.py
src/agents/services/synapse_run_store.py
src/agents/routers/synapse_controller.py
```

### Изменяемые файлы

1. `src/agents/common/config/app_config.py`
   - добавить Synapse settings;
   - секреты не включать в `to_dict()`.
2. `src/agents/common/config/app_config_loader.py`
   - загрузить новые env;
   - проверять обязательные значения только при `SYNAPSE_ENABLED=true`.
3. `src/agents/dependencies/init_dependencies.py`
   - создать Synapse client, run store и gateway service.
4. `src/agents/dependencies/dependencies.py`
   - добавить typed getters.
5. `src/agents/main.py`
   - подключить `synapse_router`;
   - открыть/закрыть Synapse HTTP client в lifespan.
6. `src/agents/api_clients/chat_storage_client/chat_storage_client.py`
   - добавить `space` ко всем операциям;
   - поддержать явный `user_id` для M2M вызовов;
   - передавать `source_event_id` в metadata.
7. `src/agents/common/api_handlers/json_api_handler.py`
   - разрешить service auth с явным `user_id`, без исходного user token.
8. `frontend/src/api.ts`
   - добавить start/status/events методы;
   - разбирать SSE `id` и сохранять cursor.
9. `frontend/src/types.ts`
   - добавить DTO run и нормализованных событий.
10. `frontend/src/App.tsx`
    - переключатель gMART/Synapse;
    - reconnect по `request_id` и cursor.
11. env examples, compose и CI/deploy configuration
    - передать только несекретные параметры через обычный env;
    - пароли/client secrets — через secret storage.

### A2A Agent Cards

До подключения live Synapse прогнать все Agent Cards через его строгую валидацию.
Текущие карточки gMART содержат нестандартные поля `google_a2a_compatible` и
`parts_array_format` внутри `capabilities`. Если настроенный Synapse использует эти
карточки, поля нужно убрать из сериализуемого Agent Card. Изменение выполняется только
в gMART; код Synapse не затрагивается.

## 10. Переменные окружения gMART

```dotenv
SYNAPSE_ENABLED=true
SYNAPSE_API_URL=http://synapse:8000
SYNAPSE_SERVICE_EMAIL=gmart@service.local
SYNAPSE_SERVICE_PASSWORD=change-via-secret-manager
SYNAPSE_WORKFLOW_ID=idu-orchestrator
SYNAPSE_RUN_CONFIG_ID=idu-default
SYNAPSE_APPROVAL_MODE=auto
SYNAPSE_HTTP_TIMEOUT=30
SYNAPSE_SSE_RECONNECT_MAX_SECONDS=30
SYNAPSE_RUN_TTL_SECONDS=86400
```

Если gMART получает готовый технический токен от ingress, email/password заменяются
соответствующей настройкой token provider. Статический короткоживущий access token в env
использовать нельзя.

## 11. Обработка ошибок

| Ситуация | Поведение gMART |
|---|---|
| Synapse login/refresh отклонён | `502 synapse_auth_failed`, без утечки response body |
| Synapse недоступен при создании | bounded retry, затем `502 synapse_unavailable` |
| SSE соединение оборвалось | reconnect с `Last-Event-ID` |
| повторно пришёл event | пропустить по `source_event_id` |
| ChatStorage временно недоступен | retry; cursor не подтверждать до сохранения |
| A2A service token отклонён | Synapse получает A2A auth error; gMART пишет security log без token |
| неизвестный chat/project mapping | `409 synapse_mapping_missing` |
| frontend отключился | relay продолжает работу |
| terminal failure | сохранить `failure`, опубликовать событие и завершить run |

Необходимо ограничить размеры prompt, event payload и Redis Stream. Большие artifacts
нужно сохранять как ссылки/файлы, а не дублировать в каждом событии.

## 12. Тестирование

### Unit tests

- `SynapseApiClient`: login, refresh, retry-once, create, follow-up, SSE parsing;
- prompt builder: обязательные metadata и отсутствие любых token/secret;
- `SynapseRunStore`: single-flight, event ordering, deduplication, cursor;
- event normalizer: все terminal states и неизвестный event type;
- ChatStorage client: `space=synapse`, service auth + `X-User-Id`;
- frontend SSE parser: `id`, multiline `data`, partial network chunks;
- A2A cards проходят строгий Synapse validator.

### Integration tests

1. frontend contract → gMART → fake Synapse → SSE → Redis → fake ChatStorage;
2. reconnect после нескольких событий не создаёт дубликаты;
3. browser disconnect не останавливает relay;
4. follow-up использует тот же Synapse project;
5. два одновременных запроса в один chat дают `409` одному из запросов;
6. истечение Synapse access token приводит к refresh и одному retry;
7. Synapse вызывает каждый gMART A2A endpoint с сервисным токеном;
8. ChatStorage история видна только в `space=synapse` исходного пользователя.

### Live smoke test

1. создать новый Synapse chat из UI;
2. убедиться, что в network нет user token в query/body/SSE;
3. проверить событие делегирования в UI;
4. проверить A2A вызов gMART;
5. дождаться `project_completed`;
6. перезагрузить UI и восстановить историю из ChatStorage;
7. отправить follow-up и проверить сохранение прежнего `synapse_project_id`.

## 13. Порядок внедрения

1. Подготовить технического пользователя Synapse и Keycloak client `synapse-gmart`.
2. Проверить A2A Agent Cards и service-token доступ к gMART.
3. Реализовать config и `SynapseApiClient`.
4. Реализовать `SynapseRunStore` и background relay.
5. Расширить ChatStorage client для `space` и явного `user_id`.
6. Добавить `/synapse/runs` и events/status endpoints.
7. Добавить frontend переключатель и reconnect.
8. Прогнать unit/integration tests.
9. Включить `SYNAPSE_ENABLED` только в testing environment.
10. После smoke test включить feature flag для ограниченной группы пользователей.

Откат выполняется выключением `SYNAPSE_ENABLED`; существующие gMART endpoints и данные
`space=main` не затрагиваются.

## 14. Критерии приёмки

- код Synapse не изменён;
- Synapse вызывает gMART по A2A с автоматически обновляемым сервисным токеном;
- пользовательские и сервисные tokens отсутствуют в prompt, Redis state, ChatStorage и логах;
- browser disconnect не останавливает выполнение и сохранение истории;
- reconnect не дублирует события;
- новый chat создаётся в `space=synapse` и связан с одним Synapse project;
- follow-up продолжает тот же Synapse project;
- события появляются на frontend в исходном порядке;
- terminal status одинаков в Synapse, Redis, frontend и ChatStorage;
- существующий `/orchestrator/route/stream` продолжает работать без изменений.
