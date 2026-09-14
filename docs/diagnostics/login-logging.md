# Журнал входа во встроенном фронтенде

Каждый `POST /auth/token`, отправленный формой входа, создаёт две записи в обычных
логах agents: `login_attempt` при получении запроса и `login_result` после завершения.
Они связаны уникальным `attempt_id`. Повторный вход для обновления токена через
тот же endpoint также записывается. Проверка доступности `/auth/available` не
считается попыткой входа.

Успех записывается на INFO, отказ с 4xx и отмена — WARNING, ошибки 5xx — ERROR.
В записи результата есть `status`, `duration_ms`, `error_type`, `downstream_status`
и одна из причин:

| reason | Значение |
| --- | --- |
| `invalid_credentials` | Логин/пароль отклонены (`invalid_grant`), HTTP 401 |
| `invalid_request` | Неправильный JSON или отсутствующие/некорректные поля, HTTP 422 |
| `login_not_configured` | Не настроен auth helper, HTTP 404 |
| `auth_helper_unavailable` | Ошибка auth helper или транспорта, HTTP 502; исходный статус в `downstream_status`, при сетевом сбое — `null` |
| `invalid_helper_response` | Auth helper ответил без непустого строкового access_token, HTTP 502 |
| `internal_error` | Непредвиденная внутренняя ошибка, HTTP 500 |
| `request_cancelled` | Выполнение запроса отменено |
| `http_error` | Прочая HTTP-ошибка |

Пример (идентификатор и время условные):

```text
Login {"attempt_id": "12345678-1234-1234-1234-123456789abc", "event": "login_result", "outcome": "failure", "username": "-", "status": 401, "reason": "invalid_credentials", "duration_ms": 123.45, "error_type": "AgentsUnauthorizedException", "downstream_status": "-"}
```

Логин скрыт по умолчанию. `AUTH_LOG_USERNAME=true` включает его в записи результата
(до 128 символов; переносы строк экранируются). Настройка добавлена в пример env
и deploy compose. Пароли, токены, API-ключи, заголовки, тела запросов/ответов и
текст исключения в журнал попыток входа не включаются.

Сервер видит только запросы, дошедшие до `/auth/token`. Сетевые ошибки браузера
до обращения к agents и вход через прямой редирект в Keycloak/auth helper не
попадают в этот журнал. Для включения на develop нужно обновить образ и
пересоздать контейнер agents; отдельный сервис логирования не требуется.
