# Agents SDK: проверка интеграции с dev, 12 сентября 2026

Текущий локальный код после миграции на `openai-agents==0.22.2` работает с dev-моделью и MCP-сервисами. Полностью зелёной проверку считать нельзя: ObjectEffects возвращает ошибку расчёта, а завершённые restriction/provision/compliance-пайплайны при reconnect дописывают события. В выполненных проверках отдельной регрессии SDK не обнаружено; это не доказательство отсутствия всех регрессий.

## Условия

- Рабочая копия `refactor/agents-sdk`, базовый commit `bc46e66`, с незакоммиченными изменениями миграции.
- Проверялся локальный Python-код; развёрнутый Agents API не обновлялся.
- Модель `gpt-oss-20b`, OpenAI-compatible API `http://10.32.11.27:8001/v1`, reasoning effort `low`.
- IDU MCP `http://10.32.11.90:31051/mcp`; DVD `:31002/mcp`; NormGraph `:31003/mcp`; ObjectEffects `:31006/effects/mcp`; Urban MCP `:31054/mcp/{group}/`; Urban API `:31001/api`.
- Данные сценария 772; действующая M2M-аутентификация из существующего локального файла конфигурации. Токены и секреты в отчёт не включены.
- Отдельный временный Redis `127.0.0.1:16389`, без persistence. История ChatStorage не записывалась, внешние данные не изменялись.
- REST/SSE: настоящее FastAPI-приложение, штатная инициализация зависимостей и lifespan, реальные dev-сервисы; HTTP-клиент использовал ASGI transport внутри процесса. Это проверка роутов и сериализации, не сетевого прокси/ingress.

## Результаты

| Проверка | Результат |
|---|---|
| SDK → dev-модель, structured output | Pydantic-ответ `value=42`, успешно |
| SDK → dev-модель, streaming | `SDK_OK`, терминальный `stop` |
| MCP discovery | IDU: 12 инструментов; ObjectEffects: 3; Urban: 77 в шести группах |
| DVD retrieval | Прямой поиск вернул 3 фрагмента |
| Document-QA, service pipeline | `done`, 10 событий, точный replay; для выбранного вопроса ответил о недостаточности найденных источников |
| NormGraph QA | `done`, 13 событий, точный replay; поиск ограничений вернул пустой набор |
| Restriction: буфер 50 м вокруг школ | `done`, реальные GeoJSON-слои, 18 событий; reconnect возвращает 25 |
| Compliance: нормы из NormGraph | `done`, 8 событий, но доступных норм нет; reconnect возвращает 9 |
| Геометрический исполнитель | Реальные IDU MCP и геоданные: 70/70 объектов проверены; 2 нарушения тестового условия, 68 соответствий |
| Provision: список доступных сервисов | `done`, 4 события; reconnect возвращает 7 |
| Provision: расчёт обеспеченности школами | `failed`, ObjectEffects: `KeyError: 'service_type'` |
| Scenario-data: типы сервисов сценария | `done`, таблица, 11 событий, точный replay |
| Orchestrator: буферы + список сервисов | Два последовательных шага (`restriction`, `scenario_data`) завершены; таблица на 27 строк, 28 событий, точный replay |
| `/llm/available_models` | HTTP 200, содержит `gpt-oss-20b` |
| `/llm/message` | HTTP 200, `SDK_HTTP_OK` |
| `/llm/message/stream` | HTTP 200, `text/event-stream`, 4 события, `SDK_SSE_OK` |
| `/documents/qa/stream`, anonymous | HTTP 200, 24 события, штатная M2M-аутентификация к DVD, финальный `done`, точный HTTP replay |

Геометрическая проверка использовала контрактную fixture T1, а не норму, полученную из dev NormGraph. Числа 2/68 относятся только к этому тестовому условию и не являются нормативным заключением.

Проверены Pydantic-схемы 95 событий восьми первоначальных сервисных прогонов: ошибок сериализации нет. Дополнительный scenario-data каталог и HTTP document-QA также проверены по своим схемам.

## Выявленные проблемы

### ObjectEffects возвращает ошибку независимо от SDK

Минимальный запрос к `CalculateServicesProvision`:

```json
{"scenario_id":772,"services":{"22":{"name":"Школа","as_layer":true}}}
```

Тип 22 получен из dev-каталога при исходном прогоне. Одинаковые аргументы дважды проверены через SDK-обёртку gMART и напрямую через `fastmcp.Client.call_tool`. Результаты полностью совпадают:

```json
{"services":{"22":{"name":"Школа","summary":null,"layers":null,"error":"KeyError: 'service_type'"}}}
```

Таким образом, конкретная ошибка приходит из ObjectEffects; SDK не меняет результат. Внутренняя причина в ObjectEffects/его входных данных не установлена. Dev-сервис не исправлялся и не перевыкатывался.

### Reconnect дописывает события

У restriction, compliance и provision после replay выполнение продолжается, хотя состояние уже `done`/`failed`. Повторный прогон не возвращает ровно исходную последовательность. Для упавшего расчёта provision также получено 9 событий вместо 5.

Такая структура reconnect-веток уже присутствует в `HEAD` до незакоммиченной миграции SDK. Для compliance проблема также описана в диагностике от 10 сентября. Повторный запуск старой версии целиком не выполнялся. Для DVD, NormGraph, scenario-data и orchestrator в этой проверке replay совпадает.

### Ограничения данных и смыслового результата

- `NormGraph.search_restrictions(limit=10)` вернул `count=0`. Поэтому выполнение compliance по реальным извлечённым нормам не подтверждено.
- Запрос scenario-data «Покажи название и основные сведения о сценарии 772» завершился технически, но ответил уточнением о типах объектов вместо сведений о сценарии. Технический `done` здесь не считается успешным выполнением пользовательского запроса. Каталог сервисов в отдельном прогоне отработал.
- Качество и применимость нормативного ответа DVD отдельно не подтверждены: HTTP-прогон сослался на СП 308.1325800.2017; оценка релевантности корпуса и юридической применимости не входила в эту smoke-проверку.
- Urban MCP во время запроса общих сведений вывел два предупреждения о завершении сессии (`second argument (exceptions) must be a non-empty sequence`), не прервав pipeline. При отдельном запросе каталога предупреждения не повторились.

## Что не проверялось

Запись/чтение истории через ChatStorage, A2A и Synapse целиком, token refresh при истечении токена, рестарты процессов, сетевой disconnect через ingress, Linux workspace worker и локальная Ollama. Эти результаты не заменяют проверку установленной в dev версии после deployment. Запрос конфигурации развёрнутого Agents API вернул 401 с локальным system password; его runtime-конфигурация не подтверждена.

## Артефакты и воспроизведение

- `output/agents-sdk-dev/`: JSON-сводки, события, replay, сырые SSE-ответы и результат сравнения ObjectEffects.
- `tmp/agents_sdk_dev_preflight.py`: проверка доступности endpoints.
- `tmp/agents_sdk_dev_check.py`: диагностический harness, явно использующий перечисленные dev-endpoints и отдельный Redis. Секреты читает из существующего `env/.env.benchmark.local`.

После запуска отдельного Redis на `127.0.0.1:16389` из корня репозитория:

```powershell
.venv\Scripts\python.exe tmp/agents_sdk_dev_check.py runtime
.venv\Scripts\python.exe tmp/agents_sdk_dev_check.py http
.venv\Scripts\python.exe tmp/agents_sdk_dev_check.py scenario-catalog
.venv\Scripts\python.exe tmp/agents_sdk_dev_check.py orchestrator
.venv\Scripts\python.exe tmp/agents_sdk_dev_check.py effects-direct
```

Последняя команда воспроизводит ошибку ObjectEffects за примерно 3 секунды после инициализации и возвращает exit code 1 (`AssertionError: ObjectEffects returns per-service error`). Диагностические файлы в `tmp/` и `output/` являются локальными артефактами, не гарантированно доступными в другом checkout.
