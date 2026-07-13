# gMART UI

Единый desktop-интерфейс для пользовательских агентов и системной диагностики gMART.

## Локальный запуск

```bash
cd frontend
npm install
npm run dev
```

Vite проксирует Agents API с `/api-agents` на `http://127.0.0.1:80` и ChatStorage с
`/api-chats` на `http://127.0.0.1:8010`. Адреса сервисов можно изменить в настройках UI.

## Production

```bash
cd frontend
npm ci
npm run build
```

Если существует `frontend/dist`, FastAPI раздаёт приложение по `/ui/`, а `/` перенаправляет
на него. Dockerfile agents собирает frontend автоматически отдельным Node.js stage.

## Авторизация

UI поддерживает Keycloak Authorization Code + PKCE. URL, realm и client ID задаются в
локальных настройках браузера. Если Keycloak не настроен, кнопка входа перенаправляет в
настроенный IDU auth helper и принимает возвращённый `access_token` из query или hash URL.

Тема, подложка карты, адреса сервисов, модель и температура сохраняются только в
`localStorage` браузера.
