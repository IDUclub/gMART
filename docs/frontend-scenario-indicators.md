# Показатели сценария: гайд для фронтенда

`GET /scenario-data/indicators/stream` — режим «Показатели сценария»: показатели выбранного
сценария и их сравнение с базовым сценарием его проекта. Ответ приходит таблицей и коротким
текстом-сводкой. Контракт сверен с кодом на 2026-09-11.

Поведение агента подробно описано в
[frontend-scenario-data.md](frontend-scenario-data.md#get-scenario-dataindicatorsstream), общие
механики SSE — в [frontend-service.md](frontend-service.md). Здесь — что сделать фронту.

## Что сделать

1. Добавить в меню «+» у поля ввода режим «Показатели сценария». В этом режиме сообщение уходит
   на `GET /scenario-data/indicators/stream`, а не в оркестратор.
2. Передавать `scenario_id` выбранного в интерфейсе сценария. Без выбранного сценария режим
   недоступен. ID базового сценария фронт не знает и не передаёт — сервер находит его сам.
3. Рисовать событие `table` универсальным компонентом по `columns`/`rows`, с сортировкой.
4. Рисовать текст из `chunk` с сохранением переносов строк.
5. Сохранять `chat_id` из `chat_created` и блокировать отправку, пока в чате идёт запрос.

## Этот режим и обычный чат

Обычный чат идёт через оркестратор (`/orchestrator/route/stream`), для данных сценария он вызывает
общий `/qa`-пайплайн. Вопросы о значениях показателей там отвечают тем же форматом, что и здесь:
та же широкая таблица и та же сводка. Базовый сценарий оркестратор находит сам и ID у пользователя
не спрашивает.

Разница — в том, как запрос доходит до ответа:

- здесь URL сразу ведёт в ветку показателей; сравнение с базовым включено по умолчанию, запрос без
  названия показателя возвращает все показатели;
- в обычном чате запрос сначала разбирает планировщик оркестратора, затем классификация `/qa`;
  базовый подставляется, только когда пользователь просит сравнение («Сравни с базовым
  сценарием»), а показатели без кавычек выбирает модель. Это дольше и зависит от модели.

## Запрос

```http
GET /scenario-data/indicators/stream?request=Сравни%20с%20базовым%20сценарием&scenario_id=848
Authorization: Bearer <access_token>
Accept: text/event-stream
```

На стенде сервис опубликован за префиксом `/gmart`, как и
`/gmart/orchestrator/route/stream`, которым пользуется текущий UI.

| Параметр | Тип | Обязателен | Назначение |
|---|---|---|---|
| `request` | `string` | да | Текст пользователя как есть |
| `scenario_id` | `integer` | да для этого режима | Сценарий, выбранный в интерфейсе |
| `chat_id` | `string` (UUID, 36 символов) | нет | Чат для продолжения; без него создаётся новый |
| `request_id` | `string` (UUID, 36 символов) | нет | Только для переподключения к уже запущенному запросу |
| `model` | `string` | нет | Не передавайте: сервер возьмёт модель провайдера |
| `temperature` | `number` | нет | Не влияет: таблицу строит код, сводку модель пишет с температурой 0 |

Без `Authorization` сервер отвечает `401`. `chat_id` или `request_id` не той длины — `422` ещё до
открытия потока.

Нативный `EventSource` не умеет передавать заголовок `Authorization`, поэтому поток читается через
`fetch` (пример ниже) или библиотеку вроде `@microsoft/fetch-event-source`.

### Что может написать пользователь

Сервер решает по тексту, что вернуть. ID сценариев вводить не нужно, ими оперирует только
интерфейс. Фразы ниже подойдут для кнопок-подсказок.

| Текст | Что вернётся |
|---|---|
| «Сравни с базовым сценарием», «Что изменилось?», «Покажи показатели», «Покажи все показатели» | Все показатели, сравнение с базовым, сводка |
| «Покажи «Земли жилой застройки»», «Покажи численность населения» | Один показатель: одна строка таблицы и одна строка текста |
| То же, но со словами «без сравнения», «не сравнивай», «без базового», «только мой сценарий» | Только выбранный сценарий, без колонок разницы |
| «Почему такое значение?», «Откуда эти данные?», расчёт плотности | Прежний формат `/qa` — см. [Другие ответы](#другие-ответы) |

## Поток событий

Каждое SSE-сообщение — `data: {"type": …, "content": …}`. Различайте события по полю `type`
внутри JSON, а не по SSE-полю `event:`.

Живой прогон: сценарий 848, проект 633, базовый сценарий 846.

```
pipeline_started   { request_id }                          ← сохранить
service_event      chat_created { chat_id, chat_title }    ← только в новом чате
status             tool_discovery «Загружаю инструменты Urban MCP…»
tool_call          projects/GetScenarioById          { scenario_id: 848 }
tool_call          projects/GetProjectById           { project_id: 633 }
tool_call          projects/GetScenarioById          { scenario_id: 846 }
tool_call          indicators/GetScenarioIndicatorsValues { scenario_id: 846 }
tool_call          indicators/GetScenarioIndicatorsValues { scenario_id: 848 }
status             response_analysis «Готовлю сводку…»     ← если сравниваются несколько показателей
table              scenario_analytics
chunk × N          текст кусками до 280 символов, done: false
chunk              { text: "", done: true }                ← ответ завершён
```

| `type` | Что делать |
|---|---|
| `pipeline_started` | Сохранить `content.request_id`: он нужен для отмены и переподключения |
| `service_event` | При `content.event.storage_event_type === "chat_created"` сохранить `chat_id` |
| `status` | Показать `content.text`. В этом режиме приходят `tool_discovery`, при временном сбое Urban MCP `tool_retry` и перед сводкой `response_analysis`; не завязывайтесь на список значений |
| `tool_call` | По желанию — в раскрываемой трассе. `content.tool_calls[]` = `{group, tool_name, arguments}`, `execution_mode: "sequential"`, `mcp_source: "URBAN_MCP/<group>"` |
| `table` | Нарисовать таблицу (см. ниже) |
| `chunk` | Склеивать `content.text`; `done: true` — конец ответа |
| `error` | Показать дружелюбное сообщение. Перед `error` уже приходит `chunk` с текстом ошибки, после — `chunk` с `done: true` |

`table` приходит до текста. Таблицы может не быть: например, если данные не подтвердились, придёт
только текст «Не удалось подтвердить данные для ответа. Уточните сценарии, названия показателей и
условия запроса.»

## Таблица

```json
{
  "type": "table",
  "content": {
    "name": "scenario_analytics",
    "title": "Показатели и сравнение сценариев",
    "columns": [
      { "key": "indicator", "label": "Показатель" },
      { "key": "unit", "label": "Ед." },
      { "key": "scenario_846", "label": "Базовый сценарий «Исходный пользовательский сценарий»" },
      { "key": "scenario_848", "label": "Ваш сценарий «Сценарий от Рокета, вариант 1»" },
      { "key": "difference", "label": "Разница" },
      { "key": "change_percent", "label": "Изменение, %" }
    ],
    "rows": [
      { "indicator": "Земли жилой застройки", "unit": "% (разница — п. п.)",
        "scenario_846": 23.41, "scenario_848": 97.6, "difference": 74.19, "change_percent": null },
      { "indicator": "Стоимость рекультивации территории", "unit": "руб",
        "scenario_846": 6949748051, "scenario_848": 2257190106,
        "difference": -4692557945, "change_percent": -67.5 },
      { "indicator": "Средняя этажность", "unit": "этажей",
        "scenario_846": 14, "scenario_848": null, "difference": null, "change_percent": null },
      { "indicator": "Население", "unit": null,
        "scenario_846": 5, "scenario_848": 5, "difference": 0, "change_percent": 0 }
    ]
  }
}
```

Правила рендера:

- **Колонки** — в порядке `columns`, заголовки — из `label`. Ключи `scenario_<id>` динамические, не
  хардкодьте их. Колонка базового сценария идёт первой.
- **Числа** приходят числами без форматирования, чтобы их можно было сортировать. Форматируйте на
  клиенте: `Intl.NumberFormat("ru-RU")`, без округления сверх того, что прислал сервер (бывает
  `0.1385`). Для `difference` и `change_percent` удобно показывать знак `+`.
- **`null`**: в колонках значений, разницы и процента — «—»; в `unit` — пустая ячейка. В колонке
  значения это «нет значения в сценарии», в разнице и проценте — «не вычисляется».
- **`unit`**: для процентных показателей — `% (разница — п. п.)`, разница у них в процентных пунктах,
  а `change_percent` всегда `null`. При разных единицах в сценариях — `км² / га`, разница `null`.
- **Режим без сравнения** — нет колонки `difference`, есть `indicator`, `unit` и одна колонка
  значений. Проверка: `columns.some(c => c.key === "difference")`.
- **Сортировка** — по клику на заголовок; `null` всегда в конце, в обе стороны.
- Всех показателей около 45 строк — таблице нужна своя прокрутка и закреплённая шапка.
- Не выбирайте компонент по `name`: то же имя `scenario_analytics` бывает у «длинной» таблицы из
  [других ответов](#другие-ответы). Рисуйте универсально по `columns`/`rows`.
- Во всех таблицах агента подписи колонок русские, а колонок с ID нет — см.
  [Таблицы](frontend-scenario-data.md#таблицы). `key` пользователю не показывайте.

### Где показывать

Сводка короткая и заканчивается фразой «Все значения — в таблице.» Рекомендация — текст, сразу под
ним таблица с видимым заголовком `title`. В текущем UI таблица стоит над текстом в контейнере
высотой 384 px, и при длинном ответе её не замечают — так было в жалобе «таблицы нет» на проде.

## Текст

Текст — plain text с `\n`: абзацы разделены пустой строкой, пункты списка начинаются с «•» и идут
через одиночный `\n`.

Когда сравниваются несколько показателей, средний абзац пишет модель: 2–4 предложения о главном.
Модель получает только посчитанные кодом значения, а сервер сверяет с ними каждое число её текста
(округление допустимо: 97,6 → 98). Если число не сошлось, модель не ответила или ответ обрезан,
приходит сводка, посчитанная кодом. Первый абзац «Сравниваются: …», строки о ненайденных
показателях и финальную фразу всегда пишет код. Один названный показатель — одна строка текста,
модель не вызывается.

Пример со сводкой модели:

```
Сравниваются: базовый сценарий «Исходный пользовательский сценарий» → ваш сценарий «Сценарий от Рокета, вариант 1».

Доля земель жилой застройки выросла с 23,41 % до 97,6 %, стоимость рекультивации территории снизилась на 67,5 %. Изменились 8 показателей, у 7 нет значения в вашем сценарии.

Все значения — в таблице.
```

Сводка, посчитанная кодом:

```
Сравниваются: базовый сценарий «Исходный пользовательский сценарий» → ваш сценарий «Сценарий от Рокета, вариант 1».

Показателей: в базовом — 44, в вашем — 37. Изменились — 8, без изменений — 29, нет значения в вашем — 7.

Изменения:
• Земли жилой застройки: 23,41 % → 97,6 % (+74,19 п. п.)
• Срок рекультивации территории: 34 175 → 8 366 дней (-75,5 %)

Нет значения в вашем сценарии: Средняя этажность, Численность населения.

Все значения — в таблице.
```

Рисуйте с `white-space: pre-wrap`. Если текст идёт через Markdown-рендер, включите жёсткие переносы
(`breaks: true` / `remark-breaks`), иначе строки с «•» склеятся в один абзац.

Когда сравнивать не с чем, первый абзац объясняет причину. Отдельного события для этого нет,
специально обрабатывать не нужно:

| Ситуация | Первый абзац |
|---|---|
| Выбран сам базовый сценарий | «Этот сценарий — базовый сценарий проекта, сравнивать не с чем.» |
| У сценария не определён проект | «Проект сценария не определён, поэтому показана только сводка без сравнения.» |
| У проекта не задан базовый сценарий | «У проекта не задан базовый сценарий, поэтому показана только сводка без сравнения.» |
| Нет прав на проект | «Базовый сценарий недоступен, поэтому показана только сводка без сравнения.» |

### Другие ответы

Вопросы «почему/откуда» и расчёт плотности отвечают другим форматом. Таблица, если придёт, будет
«длинной»: строка — пара «сценарий × показатель», колонки «Сценарий», «Показатель», «Значение»,
«Ед.», «Статус», без ID. Текст пишет код, модель не вызывается. Универсальный компонент нарисует и
эту таблицу.

## Чат и уточняющие вопросы

- Первый запрос без `chat_id` создаёт чат — придёт `chat_created`. Следующие сообщения отправляйте с
  этим `chat_id`, тогда они сохраняются в тот же чат.
- В одном чате одновременно выполняется один запрос. Второй запрос сейчас получает общий текст «Не
  удалось выполнить запрос из-за внутренней ошибки сервера. Повторите попытку позже.» и `error`:
  конкретная причина до клиента не доходит. Поэтому блокируйте поле ввода до `chunk` с `done: true`.
- Каждое сообщение этот режим разбирает самостоятельно, историю не учитывает. Сработают «Покажи
  численность населения» или «А без сравнения?» — всё нужное есть в самом тексте. Не сработают
  «А по второму?» и «А в процентах?».

## Отмена и переподключение

- **Отмена**: `POST /pipelines/{request_id}/cancel` с тем же `Authorization`. Закрытие соединения
  (`AbortController`) запрос **не** останавливает: сервер доводит его до конца и сохраняет в чат.
- **Переподключение** после разрыва: тот же `GET` с тем же `request` и сохранённым `request_id`.
  Сервер повторит все события запроса с начала, поэтому перед повтором очистите уже показанный
  ответ. Если разрыв случился до `pipeline_started`, `request_id` неизвестен — отправьте запрос
  заново.

## Токен

В этом режиме сервер сейчас **не** присылает `token_expired`: если токен истёк посреди запроса,
придёт общий `error`. Запрос короткий, несколько секунд, поэтому обновляйте токен перед отправкой
и повторяйте запрос с новым токеном, если пришёл `error`.

Общий эндпоинт `POST /pipelines/{request_id}/token` для других пайплайнов принимает новый токен в
заголовке `Authorization: Bearer <новый_токен>`, тело не нужно.

## История чата

При загрузке чата из ChatStorage ответ хранится частями сообщения ассистента: `kind: "table"` с
payload `{name, title, columns, rows, total_rows, complete}`, `kind: "text"` с текстом и
`kind: "tool_call"` с вызовами. Таблицу из истории рисуйте тем же компонентом, что и живую.

## Если ответ идёт через оркестратор

Таблица и сводка по показателям в обычном чате те же, что в этом режиме. События агента приходят
обёрнутыми в `step_event`:

```json
{
  "type": "step_event",
  "content": {
    "step": 1,
    "agent": "scenario_data",
    "event": { "type": "table", "content": { "name": "…", "title": "…", "columns": [], "rows": [] } }
  }
}
```

Разворачивайте `content.event` и отдавайте его тем же обработчикам, что и для прямого потока.
`pipeline_started` и `service_event` подагента оркестратор не пересылает, у него свои. Для
обновления токена у шага есть `step_request_id` из `step_started`.

## Пример клиента (TypeScript)

```ts
type Cell = string | number | null;
type TableRow = Record<string, Cell>;
type TableColumn = { key: string; label: string };
type TableContent = { name: string; title: string; columns: TableColumn[]; rows: TableRow[] };

type IndicatorsEvent =
  | { type: "pipeline_started"; content: { request_id: string } }
  | {
      type: "service_event";
      content: {
        event_type: "storage_event";
        event: { storage_event_type: "chat_created"; chat_id: string; chat_title: string };
      };
    }
  | { type: "status"; content: { status: string; text: string } }
  | {
      type: "tool_call";
      content: {
        execution_mode: string;
        tool_calls: { group: string; tool_name: string; arguments: Record<string, unknown> }[];
        mcp_source: string | null;
      };
    }
  | { type: "table"; content: TableContent }
  | { type: "chunk"; content: { text: string; done: boolean } }
  | { type: "error"; content: { message: string; traceback?: string } };

export function takeSseEvents(buffer: string): { events: IndicatorsEvent[]; rest: string } {
  const messages = buffer.replace(/\r\n/g, "\n").split("\n\n");
  const rest = messages.pop() ?? "";
  const events: IndicatorsEvent[] = [];
  for (const message of messages) {
    const data = message
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice("data:".length).trimStart())
      .join("\n");
    if (data) events.push(JSON.parse(data) as IndicatorsEvent);
  }
  return { events, rest };
}

export async function streamIndicators(params: {
  baseUrl: string; // с префиксом стенда, например "https://<host>/gmart"
  token: string;
  request: string;
  scenarioId: number;
  chatId?: string;
  requestId?: string;
  signal?: AbortSignal;
  onEvent: (event: IndicatorsEvent) => void;
}): Promise<void> {
  // Concatenate rather than new URL(path, base): an absolute path would drop the /gmart prefix.
  const url = new URL(`${params.baseUrl}/scenario-data/indicators/stream`);
  url.searchParams.set("request", params.request);
  url.searchParams.set("scenario_id", String(params.scenarioId));
  if (params.chatId) url.searchParams.set("chat_id", params.chatId);
  if (params.requestId) url.searchParams.set("request_id", params.requestId);

  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${params.token}`, Accept: "text/event-stream" },
    signal: params.signal,
  });
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const { events, rest } = takeSseEvents(buffer);
    buffer = rest;
    events.forEach(params.onEvent);
  }
}

const valueFormat = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 10 });
const deltaFormat = new Intl.NumberFormat("ru-RU", {
  maximumFractionDigits: 10,
  signDisplay: "exceptZero",
});

export function formatCell(key: string, value: Cell | undefined): string {
  if (value === null || value === undefined) return key === "unit" ? "" : "—";
  if (typeof value !== "number") return value;
  const delta = key === "difference" || key === "change_percent";
  return (delta ? deltaFormat : valueFormat).format(value);
}

export function isComparison(table: TableContent): boolean {
  return table.columns.some((column) => column.key === "difference");
}

export function sortRows(rows: TableRow[], key: string, direction: "asc" | "desc"): TableRow[] {
  const sign = direction === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const x = a[key] ?? null;
    const y = b[key] ?? null;
    if (x === null || y === null) return x === y ? 0 : x === null ? 1 : -1;
    if (typeof x === "number" && typeof y === "number") return sign * (x - y);
    return sign * String(x).localeCompare(String(y), "ru");
  });
}
```

Использование:

```ts
async function sendIndicatorsMessage(baseUrl: string, token: string, text: string, scenarioId: number, chatId?: string) {
  let answer = "";
  await streamIndicators({
    baseUrl,
    token,
    request: text,
    scenarioId,
    chatId,
    onEvent: (event) => {
      if (event.type === "pipeline_started") console.log("request_id", event.content.request_id);
      if (event.type === "service_event") console.log("chat_id", event.content.event.chat_id);
      if (event.type === "status") console.log(event.content.text);
      if (event.type === "table") console.log(event.content.title, isComparison(event.content));
      if (event.type === "chunk") answer += event.content.text;
    },
  });
  return answer;
}
```

## Проверка на стенде

| Действие | Ожидаемо |
|---|---|
| Сценарий 848, «Сравни с базовым сценарием» | Таблица на 6 колонок, 44 строки; базовый — первая колонка значений; сводка начинается с «Сравниваются: …» |
| Сценарий 848, «Покажи «Земли жилой застройки»» | Одна строка; текст «23,41 % → 97,6 % (+74,19 п. п.)» |
| Сценарий 846 (сам базовый), «Покажи все показатели» | Первый абзац «Этот сценарий — базовый…»; в таблице 3 колонки, значения подписаны «Базовый сценарий «…»» |
| Сценарий 848, «Покажи все показатели без сравнения» | 3 колонки, без разницы и процента |
| Сортировка по «Изменение, %» | Пустые ячейки внизу в обе стороны |
| Второе сообщение в том же чате | Без `chat_created`, сообщение попадает в тот же чат |
