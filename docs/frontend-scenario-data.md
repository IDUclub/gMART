# Агент городских данных

Агент `scenario-data-agent` отвечает на фактические вопросы по Urban API через
сгруппированный внешний Urban MCP. Он работает строго в режиме чтения и может
возвращать текст, таблицы и GeoJSON-слои.

## Контекст сценария

`scenario_id` необязателен:

- если он передан, агент использует общие и сценарные инструменты;
- если его нет, инструменты, где `scenario_id` является обязательным параметром,
  исключаются из каталога до планирования;
- если вопрос относится к конкретному сценарию, агент просит пользователя выбрать
  сценарий и не пытается угадать идентификатор.

Все шесть Urban MCP-групп загружаются с bearer-токеном пользователя. Загрузка
выполняется в режиме fail-fast: недоступность любой группы завершает запрос явной
ошибкой, чтобы агент не сформировал неполный ответ.

## REST/SSE

### `GET /scenario-data/qa/stream`

Параметры запроса:

| Параметр | Тип | Обязателен | Назначение |
|---|---|---|---|
| `request` | `string` | да | Вопрос пользователя |
| `model` | `string` | нет | Модель LLM |
| `temperature` | `number` | нет | Температура генерации |
| `scenario_id` | `integer` | нет | Контекст сценария Urban API |
| `chat_id` | `string` | нет | Продолжение истории ChatStorage |
| `request_id` | `string` | нет | Повтор буферизованных событий пайплайна |

Основные SSE-события: `pipeline_started`, `status`, `tool_call`,
`feature_collection`, `table`, `chunk`, `token_expired`, `pipeline_suspended` и
`error`.

## A2A

### Обнаружение

```http
GET /scenario-data/.well-known/agent-card.json
```

Имя карточки: `scenario-data-agent`. Протокол: A2A 0.3.0, транспорт JSON-RPC.
Расширение `scenario-context/v1` объявлено в карточке как необязательное.

### JSON-RPC

```http
POST /scenario-data/a2a
Authorization: Bearer <token>
Content-Type: application/json
```

Поддерживаются методы `SendMessage`/`message/send`,
`SendStreamingMessage`/`message/stream`, `GetTask`/`tasks/get`,
`ListTasks`/`tasks/list`, `CancelTask`/`tasks/cancel` и
`GetExtendedAgentCard`.

`scenario_id` и `chat_id` можно передать через `params.message.metadata` или
DataPart. Для совместимости `scenario_id=772` также распознаётся в тексте.

```json
{
  "jsonrpc": "2.0",
  "id": "request-1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Какие типы сервисов доступны?"}],
      "metadata": {"scenario_id": 772}
    }
  }
}
```

A2A-запуски выполняются с `persist_history=False`: они могут читать переданный
`chat_id` как контекст, но не создают и не изменяют историю ChatStorage.

Результаты преобразуются в артефакты:

- текст — `scenario-data-agent-text` (`text/plain`);
- GeoJSON — `geojson-<layer>` (`application/vnd.geo+json`);
- таблица — `table-<name>` (`application/json`).
