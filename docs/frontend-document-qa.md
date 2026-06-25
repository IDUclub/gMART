# Document-QA agent (RAG по нормативной документации)

Агент `document-qa` отвечает на вопросы по **нормативной документации** (градостроительство и
городское планирование): извлекает фрагменты из **IDU_DVD** (векторная БД документов, через MCP
по `DVD_MCP_SERVER`) и формирует ответ, обоснованный источниками.

Общие механики — авторизация (`Authorization: Bearer <token>`), формат SSE, переподключение по
`request_id` — те же, что в [frontend-service.md](frontend-service.md). Ниже только специфика
этого агента.

## Как работает

Итеративный цикл (до 3 итераций):

1. **retrieval_planning** — LLM подбирает параметры поиска: запрос, тип (`text` / `table` / `all`),
   число фрагментов и ширину соседнего контекста.
2. **searching** — детерминированный поиск в IDU_DVD по выбранным параметрам.
3. **answer_drafting** — ответ стримится чанками; каждый чанк помечен номером черновика `iteration`.
4. **self_review** — критик-LLM проверяет черновик по найденным фрагментам. Если ответ не устроил —
   формируется уточнённый поисковый запрос, поиск и ответ **переписываются**. Последняя итерация
   принимается без проверки.

Геометрии в этом пайплайне нет — событие `feature_collection` агент не присылает.

## Эндпоинты

### `GET /documents/qa/stream` — SSE

```http
Authorization: Bearer <access_token>
```

| Параметр | Тип | По умолч. | Описание |
| --- | --- | --- | --- |
| `request` | `string` | обязательный | Вопрос пользователя. |
| `model` | `string` | `gpt-oss:20b` | Имя модели Ollama. |
| `temperature` | `number` | `1.0` | Температура генерации. |
| `scenario_id` | `number \| null` | `null` | ID сценария Urban API. Если `chat_id` не передан — создаётся чат с этим `scenario_id` и выведенным из него `project_id`. |
| `chat_id` | `string \| null` | `null` | UUID чата из Chat Storage для продолжения диалога. |
| `request_id` | `string \| null` | `null` | UUID пайплайна для переподключения после разрыва. |

### `POST /documents/a2a` — A2A JSON-RPC

Карточка агента: `GET /documents/.well-known/agent-card.json` (имя `document-qa-agent`).
Методы: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`,
`GetExtendedAgentCard`. `scenario_id` / `chat_id` передаются в `params.message.metadata`.

## Типы SSE-событий

Конверт стандартный (`{ "type", "content" }`). Набор событий этого агента:

| Тип | Назначение |
| --- | --- |
| `pipeline_started` | первое событие; сохранить `content.request_id`. |
| `status` | прогресс; `content.status` ∈ `retrieval_planning`, `searching`, `answer_drafting`, `self_review`, `finalizing`. |
| `chunk` | текст ответа; `content = { text, done, iteration }`. |
| `tool_call` | вызов поиска в IDU_DVD (`content.mcp_source = "DVD_MCP_URL"`). |
| `service_event` | создан чат (`chat_created` с `chat_id` / `chat_title`). |
| `warning` | неблокирующее предупреждение (см. ниже). |
| `error` | ошибка; следом приходит завершающий `chunk` с `done: true`. |

> Этот агент **не** использует `feature_collection`, `token_expired`, `pipeline_suspended`:
> обновление токена не применяется (IDU_DVD MCP без авторизации; токен нужен только для истории
> в Chat Storage).

### `chunk` — итеративные черновики

```json
{ "type": "chunk", "content": { "text": "Согласно [1]…", "done": false, "iteration": 2 } }
```

Группируйте чанки по `iteration`. При **росте** `iteration` отклонённый черновик заменяется новым —
покажите свежий вместо предыдущего. Финальный ответ — последний `chunk` с `done: true`.

### `warning`

```json
{
  "type": "warning",
  "content": {
    "code": "project_id_unavailable",
    "scenario_id": 772,
    "message": "Не удалось получить идентификатор проекта (project_id) по scenario_id=772. Фильтр проекта не будет сохранён, выполнение запроса продолжается."
  }
}
```

Сейчас единственный код — `project_id_unavailable`. Чат всё равно создаётся (только с `scenario_id`),
выполнение запроса продолжается. Покажите ненавязчивое уведомление.

### `self_review` — самопроверка

Статус `self_review` с текстом вида «Модель не удовлетворена ответом: … Переформулирую запрос и
переписываю ответ…» означает, что текущий черновик отклонён и начинается новая итерация. Используйте
для индикации того, что модель улучшает ответ.

## Переподключение и история

Механика `request_id` (буфер событий + резюм с последнего чекпоинта) и `chat_id` — как в
[frontend-service.md](frontend-service.md) (флоу 1, 2, 4). Окно переподключения — 360 секунд
(`PIPELINE_TTL`); после истечения состояние в Redis удаляется и запрос с тем же `request_id`
стартует заново.
