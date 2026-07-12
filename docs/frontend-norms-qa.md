# Norms-QA agent (граф-RAG по нормативным ограничениям, NormGraph)

Агент `norms-qa` отвечает на вопросы по **нормативным ограничениям** (градостроительство,
СП/СНиП/ГОСТ/СанПиН): запрашивает граф ограничений в **NormGraph** (через MCP по
`NORM_GRAPH_MCP_SERVER`) и формирует ответ, обоснованный источниками (документ, редакция,
номер пункта, `restriction_id`).

Общие механики — авторизация (`Authorization: Bearer <token>`), формат SSE, переподключение по
`request_id` — те же, что в [frontend-service.md](frontend-service.md). Ниже только специфика
этого агента.

## Как работает

Итеративный цикл (до 3 итераций):

1. **retrieval_planning** — LLM выбирает инструмент NormGraph: `search_restrictions` для открытых
   вопросов или `restrictions_applicable` для вопросов вида «какие ограничения действуют на X»,
   плюс фильтры, глубину раскрытия графа (`neighbors_depth`) и нужна ли проверка противоречий.
2. **executing** — детерминированный вызов выбранного инструмента NormGraph.
3. **conflict_check** *(опционально)* — если план запросил проверку, для найденных ограничений
   вызывается `list_conflicts`; найденные противоречия попадают в контекст.
4. **answer_drafting** — ответ стримится чанками; каждый чанк помечен номером черновика
   `iteration`. Ответ обязан ссылаться на источники по каждому приведённому ограничению.
5. **self_review** — критик-LLM проверяет черновик: обоснованность, наличие ссылок на источники,
   упоминание найденных противоречий. Если ответ не устроил — формируется уточнённый запрос,
   и ответ **переписывается**. Последняя итерация принимается без проверки.

Геометрии в этом пайплайне нет — событие `feature_collection` агент не присылает.

## Эндпоинты

### `GET /norms/qa/stream` — SSE

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

### `POST /norms/a2a` — A2A JSON-RPC

Карточка агента: `GET /norms/.well-known/agent-card.json` (имя `norms-qa-agent`).
Методы: `SendMessage`, `SendStreamingMessage`, `GetTask`, `ListTasks`, `CancelTask`,
`GetExtendedAgentCard`. `scenario_id` / `chat_id` передаются в `params.message.metadata`.

## Типы SSE-событий

Конверт стандартный (`{ "type", "content" }`). Набор событий этого агента:

| Тип | Назначение |
| --- | --- |
| `pipeline_started` | первое событие; сохранить `content.request_id`. |
| `status` | прогресс; `content.status` ∈ `retrieval_planning`, `executing`, `conflict_check`, `answer_drafting`, `self_review`, `finalizing`. |
| `chunk` | текст ответа; `content = { text, done, iteration }`. |
| `tool_call` | вызов(ы) NormGraph за раунд (`content.mcp_source = "NORM_GRAPH_MCP_URL"`); может включать и основной вызов, и вызовы `list_conflicts`. |
| `service_event` | создан чат (`chat_created` с `chat_id` / `chat_title`). |
| `warning` | неблокирующее предупреждение (см. ниже). |
| `error` | ошибка; следом приходит завершающий `chunk` с `done: true`. |

> Этот агент **не** использует `feature_collection`, `token_expired`, `pipeline_suspended`:
> обновление токена не применяется (NormGraph MCP без авторизации; токен нужен только для истории
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
    "message": "Не удалось получить идентификатор проекта…"
  }
}
```
