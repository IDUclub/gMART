# Агентный слой: Google ADK и pydantic-ai

Агентная логика gMART построена на двух фреймворках. Каждый используется там, где
он сильнее:

| Что | Фреймворк | Где в коде |
|---|---|---|
| Оркестратор: выбирает агентов и решает, что делать дальше | **Google ADK** (`google-adk`) | `src/agents/services/orchestrator/orchestrator_adk.py`, `src/agents/llm_agents/adk_model.py` |
| Типизированные LLM-этапы внутри пайплайнов: планы, маппинги, критики, ответы, сводки | **pydantic-ai** | `src/agents/llm_agents/stage.py`, `src/agents/llm_agents/pydantic_model.py` |
| Протокол A2A (agent card, состояния задач) | **a2a-sdk** (официальный SDK протокола) | `src/agents/a2a/a2a_types.py` |

Пайплайны под-агентов (restriction, compliance, provision, scenario_data, documents,
norms, PZZ, project_files) остаются детерминированными программами. Модель строит план
или отвечает, а MCP-инструменты по этому плану вызывает код. Автономен только
оркестратор: для него под-агенты — это инструменты.

Внешние контракты не изменились: REST/SSE-события, A2A v0.3 (JSON-RPC, agent card),
ChatStorage и реконнект через Redis.

## Транспорт к модели

Оба фреймворка обращаются к модели через один и тот же `BaseLlmAdapter`
(`src/agents/model_clients`). Это OpenAI-совместимый сервер (vLLM) или Ollama. В адаптере
сосредоточено всё, что нужно серверу: приоритеты запросов (`request_kind`), pacing,
способ передать `think` конкретному серверу, ограниченный повтор обрезанного JSON.

* `GmartChatModel` — модель pydantic-ai поверх адаптера (на базе `FunctionModel`).
* `GmartAdkLlm` — `BaseLlm` ADK поверх адаптера. Переводит `genai`-контенты в
  chat-сообщения, function declarations в OpenAI `tools`, и `tool_calls` обратно.

Поэтому тестовые двойники и захват запросов бенчмарками (`benchmarks/harness/capture.py`),
которые подменяют адаптер, работают без изменений.

## LLM-этапы на pydantic-ai

Каждое обращение пайплайна к модели — это `LlmStage`:

```python
from src.agents.llm_agents.stage import FIX_JSON_PROMPT, LlmStage, StageOutputError

_PLAN_STAGE = LlmStage(
    "provision.plan",                      # request_kind: приоритет, логи, бенчмарки
    output_type=ProvisionPlan,             # ответ валидируется pydantic-ai (NativeOutput)
    retries=2,                             # невалидный ответ возвращается модели
    retry_prompt=FIX_JSON_PROMPT,          # ...с этой инструкцией (на русском)
    think=False,
    options={"temperature": 0, "num_predict": 1024},
)

result = await _PLAN_STAGE.run(llm_client, model, messages, schema=schema)
result.output       # ProvisionPlan
result.text         # сырой ответ
result.done_reason  # "stop" / "length" / ...
```

* `messages` — те же chat-сообщения, что строили сервисы. Промпты и схемы уходят на
  сервер без изменений: по умолчанию схема — это `output_type.model_json_schema()`.
* Повтор после невалидного ответа отправляет прежнюю последовательность: исходные
  сообщения, ответ модели, инструкция этапа (`retry_prompt`). Если валидатор этапа бросил
  `ModelRetry`/`ValueError`, его объяснение дописывается к инструкции.
* Если ответ так и не прошёл валидацию, поднимается `StageOutputError` (подкласс
  `ValueError`) с сырым текстом, `done_reason` и исходной ошибкой (`.error`).
* `parser=` — для ответов, которым мало валидации по типу: восстановить id, сверить с
  данными запроса. Парсер получает сырой текст, его результат становится выходом этапа.
* `stage.stream(...)` стримит текстовые этапы по дельтам. Итог (`done_reason`, `terminal`)
  можно получить через `result=[]`.
* Пустой текстовый ответ — не ошибка: этап возвращает `""`, и пайплайн решает сам.

Там, где на каждой попытке меняются бюджет вывода, схема или `reasoning_effort` (DVD
`_request_json`, планер и маппер scenario_data), политика попыток осталась в пайплайне.
Каждая попытка — отдельный запуск этапа.

Генерация названия чата (`BaseLlmService.generate_chat_title`) намеренно осталась на
`llm_client.generate`. Это служебная генерация одной строки через отдельный API
адаптера, а не шаг агента.

## Автономный оркестратор на Google ADK

`ORCHESTRATOR_ENGINE` выбирает движок `/orchestrator`:

* `adk` (по умолчанию) — ADK `LlmAgent`. Его инструменты: `run_<ключ агента>(task)`
  для каждого доступного агента и `ask_clarification(question)`. Правила маршрутизации те
  же, что у JSON-планера (`OrchestratorPlanBuilder.build_tool_instruction`): вместо формата
  JSON — правила работы с инструментами.
* `planner` — прежний JSON-планер (`OrchestratorPlanBuilder.build_plan`).

Как ADK-агент ложится на контракт SSE:

1. Вызовы первого хода — это план. Они проходят те же детерминированные поправки, что и
   план JSON-планера (`refine_plan`): сравнение документов уходит в documents, обзор
   регламентов получает norms и documents, недоступный агент или больше
   `MAX_PLAN_STEPS` шагов превращают план в уточнение. После поправок план объявляется
   событием `plan`.
2. Каждый вызов инструмента — шаг с событиями `step_started` / `step_event` /
   `step_finished`. Его выполняет тот же код, что и шаг плана
   (`OrchestratorService._execute_step`). Инструмент возвращает модели статус и выжимку.
3. Шаги первого плана идут по порядку, и каждый получает выжимки предыдущих.
   Параллельно идёт только план из одних norms/documents, как и раньше. События всегда
   приходят в порядке шагов (`OrderedStepBus`).
4. Увидев результаты, агент может вызвать ещё агентов: нумерация шагов продолжается, всего
   не больше `MAX_PLAN_STEPS`. Шаг, который ждёт пользователя (уточнение, приостановка),
   завершает работу. Неуспешный шаг работу не завершает.
5. Финал — как раньше: `orchestrator_final`, файлы, сохранение в ChatStorage.

Ответы на вопросы, заданные шагом ранее (выбор документа compliance, вопрос
project_files), направляются обратно детерминированно, без модели.

### Требование к серверу модели

ADK-оркестратору нужен вызов инструментов (tool calling) на сервере модели. Для vLLM
с gpt-oss это:

```bash
vllm serve openai/gpt-oss-20b --enable-auto-tool-choice --tool-call-parser openai
```

Если сервер отклоняет запрос с инструментами, а ни один шаг ещё не начался, оркестратор
пишет предупреждение в лог и обрабатывает этот запрос JSON-планером. Ответ 400/422/501
(«не поддерживается») запоминается до перезапуска процесса: следующие запросы сразу идут
в планер, без заведомо неудачного вызова. Временные сбои (например, 503) так не
запоминаются. Ollama поддерживает инструменты у моделей с tool calling.

## A2A на a2a-sdk

Карточки агентов собираются из моделей `a2a.compat.v0_3.types` (`AgentCard`,
`AgentSkill`), состояния задач — `TaskState` из a2a-sdk. Сериализация карточек и задач
совпадает с прежней (на python-a2a) байт в байт. JSON-RPC-контроллеры и executors не
изменились.

## Тесты

* `tests/unit/test_llm_stage.py` — слой этапов: запросы доходят до адаптера без
  изменений, повторы, парсеры, стриминг, пустые ответы, проброс исключений.
* `tests/unit/test_orchestrator_adk.py` — ADK-оркестратор: контракт событий, запросы к
  адаптеру с `tools`, уточнение, добавленные шаги, остановка, параллельные шаги, откат на
  планер.
* Тесты JSON-планера закрепляют `ORCHESTRATOR_ENGINE=planner`.
* Бенчмарк-этап `orchestrator.route` замораживает первый запрос ADK-агента.
