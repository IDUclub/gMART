# Парки: подключение фронта

Задача: `feat/ICII-234-steve-green-parks-integration` (ICII-234).

Фронт обращается к gMART. Backend согласовывает параметры, запускает сервис Стива,
получает GeoJSON и ТЭП, затем выдаёт анализ и рекомендации. Включение на backend:
`PARKS_API_URL=https://api.1.urban.actcognitive.org`.

## Один пользовательский сценарий

1. Пользователь описывает парк, выделяет границу и отмечает минимум две разные точки входа на карте.
2. Отправьте `GET /parks/generate/stream`. Текст — в `request`, геометрия —
   в JSON-параметре `inputs`. Передавайте `Authorization: Bearer …`.
3. Сохраните `request_id` из `pipeline_started`, а после `chat_created` — `chat_id`.
4. Покажите `clarification.content.question` и `parameters`. Если `ready=true`,
   предложите подтвердить или написать правки. До подтверждения генерация не запускается.
5. Правки отправляются новым запросом с прежним `chat_id`, без `request_id`.
   Backend вернёт обновлённые параметры и новый `revision`.
6. Для кнопки подтверждения отправьте новый запрос с тем же `chat_id`,
   `request=Подтверждаю` и `inputs={"confirmed_revision": <revision>}`.
   Геометрию и теги повторять при подтверждении не нужно: их присутствие считается правкой.
7. Показывайте `status`; `status=heartbeat` означает, что backend продолжает работать.
8. На `feature_collection` добавьте слой на карту сразу. Следом придут ТЭП в
   `status` с `status=economy`, затем текст анализа в `chunk`.
   `chunk.content.done=true` закрывает текущий ответ, включая ответ с вопросом согласования.

У событий прежний конверт:

```json
{"type":"chunk","content":{"text":"Анализ и рекомендации…","done":false,"event_id":8}}
```

## Геометрия и параметры

`inputs` — JSON, закодированный через `URLSearchParams`. Для большого GeoJSON используйте
`POST /parks/generate/stream` с теми же полями в JSON-теле и тем же SSE-ответом.

```json
{
  "request": "Создай парк с двумя детскими площадками и озеленением",
  "inputs": {
    "park_geojson": {
      "type": "FeatureCollection",
      "features": [{
        "type": "Feature",
        "properties": {"type": "boundary"},
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[30,60],[30.01,60],[30.01,60.01],[30,60]]]
        }
      }, {
        "type": "Feature",
        "properties": {"type": "entry"},
        "geometry": {"type": "Point", "coordinates": [30.002, 60]}
      }, {
        "type": "Feature",
        "properties": {"type": "entry"},
        "geometry": {"type": "Point", "coordinates": [30.01, 60.005]}
      }]
    }
  }
}
```

Для генерации нужны один `boundary` типа Polygon/MultiPolygon и минимум две разные
точки входа Point с `properties.type=entry`. Координаты WGS84: `[longitude, latitude]`.
Без входов черновик принимается, но `ready=false`: backend просит отметить входы,
а подтверждение не запускает генерацию. Backend не выдумывает границу или входы.

Необязательные `tags` и `location_requests` можно передать явно. Без них пожелания
извлекает LLM. Повторение тега задаёт количество объектов. Геометрию и все правки
пользователь видит до запуска.

## Отображение результата

- `feature_collection.content.feature_collection` — полный GeoJSON, не картинка:
  стилизуйте объекты по `properties.type` (`road`, `greenery`, `grass`, `playground`,
  `calm_rest`, `multifunctional`, `fountain`, `border`, `entry_point` и другие).
- В `properties.tags` может лежать JSON-строка с атрибутами; разбирайте её отдельно
  и безопасно обрабатывайте некорректные значения.
- `status=economy`: показатели находятся в `content.economy.economyParams`.
  У каждого есть `name`, `primaryValue`, `secondaryValue` (может быть null).
  Для площадей используются м²; единица `road_density` не документирована — не
  добавляйте знак процента без уточнения контракта сервиса.
- Число GeoJSON-объектов не обязано совпадать с `secondaryValue` в ТЭП.
- Анализ приходит позже отдельными `chunk`. Слой показывайте сразу, не дожидаясь LLM.
- Общий frontend репозитория умеет добавлять GeoJSON-слой; отдельные карточки ТЭП
  и парковая легенда требуют реализации на фронте.
- При открытии истории чата восстанавливайте карту из `data`-части
  `payload.type=feature_collection`, а показатели — из `payload.type=parks_economy`.
  Прямой парковый REST и успешный шаг оркестратора сохраняют одинаковые конверты.

## Переподключение

При обрыве повторите GET с прежним `request_id` и `request` (может быть пустой строкой).
Передайте последний обработанный SSE `id` в `Last-Event-ID` или в `after`.
Backend отдаст оставшиеся события и продолжит отслеживать сохранённый `taskId`.
Генерация продолжает работать после закрытия вкладки. После перезапуска backend
восстанавливается при новом подключении; освобождение блокировки занимает до 30 секунд.

Без курсора события переигрываются с начала: обновляйте слой по `request_id`,
а события дедуплицируйте по `content.event_id`. Heartbeat не меняет курсор.
Хранение состояния и событий — 24 часа с последней записью.

`GET /parks/requests/{request_id}` возвращает `status`, `phase`, `task_id` и `event_seq`.
Чужой или истёкший идентификатор возвращает 404. Если связь оборвалась во время
самой отправки задачи и `taskId` не получен, статус — `submission_unknown`:
backend сообщает о неопределённом запуске и не создаёт вторую задачу автоматически.

Используйте `fetch` с чтением SSE, поскольку стандартный браузерный EventSource
не позволяет задать Bearer-заголовок. Обрабатывайте SSE по границам кадров,
а не по границам сетевых чанков.

## Оркестратор

Агент имеет ключ `parks`. Используйте существующий `GET /orchestrator/route/stream`,
передав `parks_inputs` как JSON в query, или POST с `parks_inputs` в теле.
События парка приходят внутри `step_event.content.event`.
В ответ на согласование отправьте новое сообщение с тем же `chat_id`.
`step_started.content.step_request_id` — идентификатор паркового запроса;
по нему также доступны восстановление через `/parks/generate/stream` и проверка состояния.

Точный контракт, A2A-примеры и критерии приёмки: [версия для агента](parks-front-agent.md).
