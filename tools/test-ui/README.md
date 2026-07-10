# gMART Test UI

Небольшой Vite/React-интерфейс для ручного тестирования `agents` SSE-пайплайнов, Keycloak-авторизации, GeoJSON-слоёв и MCP-инструментов.

## Запуск

```bash
cd tools/test-ui
npm install
npm run dev
```

## FastAPI mount

Agents FastAPI автоматически монтирует собранный UI под `/test-ui`, если существует папка `tools/test-ui/dist`.

```bash
cd tools/test-ui
npm install
npm run build
```

После запуска agents API страница будет доступна по адресу:

```text
http://127.0.0.1/test-ui/
```

Если `dist` отсутствует, API стартует как обычно и только пишет в лог подсказку.
В Docker-образе `docker/Dockerfile-agents` UI собирается отдельным Node.js stage и копируется в финальный agents image автоматически.
При запуске под `/test-ui` поле `Agents API URL` по умолчанию указывает на текущий origin страницы.

По умолчанию UI использует:

- Agents API: `http://127.0.0.1:80`
- MCP URL: `http://127.0.0.1:8000/mcp`
- Scenario ID: `772`
- Model: `gpt-oss:20b`

Все значения можно поменять в блоке настроек. Настройки сохраняются в `localStorage`.
Там же переключается светлая/тёмная тема интерфейса.

## Keycloak

В репозитории не найдено тестовых Keycloak-настроек, поэтому поля Keycloak пустые. Заполните:

- `Keycloak URL`
- `Realm`
- `Client ID`
- `Redirect URI`

Клиент ожидается public/browser client с Authorization Code + PKCE. Для работы login callback добавьте текущий Vite URL в разрешённые redirect URI клиента.

Если неверные Keycloak-настройки мешают вернуться в интерфейс, откройте UI с одним из параметров:

- `?skipAuth=1` — открыть страницу без автоматической инициализации Keycloak, сохранив настройки.
- `?resetSettings=1` — сбросить настройки UI в `localStorage`.

Например: `http://localhost:5173/?skipAuth=1`.

## Auth helper

Если для тестового стенда используется `https://idu-auth-helper.idulab.ru/`, можно не настраивать browser client в Keycloak:

1. Нажмите `Открыть auth helper`.
2. Получите токен в helper-е.
3. Вставьте token или строку `Bearer ...` в поле `Manual Bearer token`.
4. Нажмите `Применить токен`.

UI также пытается автоматически подхватить токен из URL, если helper вернёт его как `?access_token=...`,
`?token=...`, `#access_token=...` или `#token=...`.

## Proxy

Режим `Vite proxy` проксирует запросы через dev-сервер Vite (`/__gmart_proxy`) и помогает обходить CORS при тестировании MCP. Режим `Direct browser fetch` отправляет запросы напрямую из браузера.
Если видите `fetch failed` для локальных сервисов, сначала проверьте, что `agents`/`idu_mcp` запущены, и попробуйте `127.0.0.1` вместо `localhost` в URL.
