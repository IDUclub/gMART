import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Keycloak from "keycloak-js";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./styles.css";

const STORAGE_KEY = "gmart-test-ui-settings";
const bootParams = new URLSearchParams(window.location.search);
const resetSettingsOnBoot = bootParams.has("resetSettings");
const skipAuthOnBoot = resetSettingsOnBoot || bootParams.has("skipAuth");
const runningUnderFastApiMount = window.location.pathname.startsWith("/test-ui");

const AGENTS = {
  restrictions: {
    label: "Ограничения",
    path: "/restrictions/generate_restrictions/stream",
    needsScenario: true,
  },
  provision: {
    label: "Обеспеченность",
    path: "/provision/calculate_effects/stream",
    needsScenario: true,
  },
  documents: {
    label: "Документы QA",
    path: "/documents/qa/stream",
    needsScenario: false,
  },
};

const EXAMPLES = {
  restrictions: [
    "Построй зону ограничения вокруг школ 200 метров",
    "Нельзя размещать магазины алкогольной продукции в непосредственной близости от школ. Какие магазины попадают в радиус действия школ в пределах 200 метров?",
    "Построй буфер 100 метров вокруг детских садов",
  ],
  provision: [
    "Какая обеспеченность школами?",
    "Дай сводку по обеспеченности сервисами",
    "Как проект повлияет на обеспеченность детскими садами?",
    "Какие сервисы есть в проекте?",
  ],
  documents: [
    "Какие нормативные требования влияют на размещение школ?",
    "Найди требования к радиусам доступности детских садов",
    "Объясни, какие ограничения могут быть у жилой застройки рядом с санитарными зонами",
  ],
};

const COLORS = [
  "#2563eb",
  "#dc2626",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#be123c",
  "#4d7c0f",
];

const defaultSettings = {
  agentsBaseUrl: runningUnderFastApiMount ? window.location.origin : "http://127.0.0.1:80",
  mcpUrl: "http://127.0.0.1:8000/mcp",
  requestMode: runningUnderFastApiMount ? "direct" : "proxy",
  theme: "light",
  authHelperUrl: "https://idu-auth-helper.idulab.ru/",
  keycloakUrl: "",
  keycloakRealm: "",
  keycloakClientId: "",
  redirectUri: window.location.origin + window.location.pathname,
  tileUrl: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
  tileAttribution: "© OpenStreetMap contributors",
};

function loadSettings() {
  if (resetSettingsOnBoot) {
    localStorage.removeItem(STORAGE_KEY);
    window.history.replaceState({}, "", window.location.pathname);
    return defaultSettings;
  }

  try {
    return { ...defaultSettings, ...JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "{}") };
  } catch {
    return defaultSettings;
  }
}

function saveSettings(settings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

function joinUrl(baseUrl, path) {
  return new URL(path, baseUrl.replace(/\/+$/, "") + "/");
}

function viaProxy(url, settings) {
  if (settings.requestMode !== "proxy") {
    return url.toString();
  }

  const proxied = new URL("/__gmart_proxy", window.location.origin);
  proxied.searchParams.set("url", url.toString());
  return proxied.toString();
}

function proxiedUrl(url) {
  const proxied = new URL("/__gmart_proxy", window.location.origin);
  proxied.searchParams.set("url", url.toString());
  return proxied.toString();
}

function asUrl(value, fallbackProtocol = "https:") {
  const trimmed = value.trim();
  if (!trimmed) {
    throw new Error("URL is empty");
  }

  if (/^https?:\/\//i.test(trimmed)) {
    return new URL(trimmed);
  }

  return new URL(`${fallbackProtocol}//${trimmed}`);
}

function mcpHealthUrl(mcpUrl) {
  const url = asUrl(mcpUrl, "http:");
  return new URL("/health", url.origin);
}

function describeFetchFailure(error, target, mode) {
  const message = error instanceof Error ? error.message : String(error);
  const cause = error?.cause instanceof Error ? `\nCause: ${error.cause.message}` : "";
  const hints = [
    "Проверьте, что URL открывается из браузера.",
    "Для локальных сервисов попробуйте 127.0.0.1 вместо localhost.",
    "Для чужого домена Direct mode может падать из-за CORS; проверьте тот же URL через Vite proxy.",
    "Если страница UI открыта по https, http API может быть заблокирован браузером как mixed content.",
  ].join("\n- ");

  return [
    `Fetch failed (${mode})`,
    `URL: ${target}`,
    `Error: ${message}${cause}`,
    `Hints:\n- ${hints}`,
  ].join("\n");
}

async function safeFetch(target, init, mode) {
  try {
    return await fetch(target, init);
  } catch (error) {
    throw new Error(describeFetchFailure(error, target, mode));
  }
}

function parseJwt(token) {
  if (!token) return null;
  try {
    const payload = token.split(".")[1];
    return JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
  } catch {
    return null;
  }
}

function formatTimeFromSeconds(seconds) {
  if (!seconds) return "—";
  return new Date(seconds * 1000).toLocaleString();
}

function parseSseMessages(buffer) {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  const events = [];

  for (const part of parts) {
    const data = part
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");

    if (!data) continue;
    events.push(JSON.parse(data));
  }

  return { events, rest };
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function getTokenFromLocation() {
  const search = new URLSearchParams(window.location.search);
  const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
  return (
    search.get("access_token") ||
    search.get("token") ||
    hash.get("access_token") ||
    hash.get("token") ||
    ""
  );
}

function App() {
  const [settings, setSettings] = useState(loadSettings);
  const [agent, setAgent] = useState("restrictions");
  const [requestText, setRequestText] = useState(EXAMPLES.restrictions[0]);
  const [scenarioId, setScenarioId] = useState("772");
  const [model, setModel] = useState("gpt-oss:20b");
  const [temperature, setTemperature] = useState("1");
  const [chatId, setChatId] = useState("");
  const [requestId, setRequestId] = useState("");
  const [answer, setAnswer] = useState("");
  const [events, setEvents] = useState([]);
  const [tables, setTables] = useState([]);
  const [layers, setLayers] = useState([]);
  const [streamStatus, setStreamStatus] = useState("idle");
  const [notice, setNotice] = useState("");
  const [keycloak, setKeycloak] = useState(null);
  const [token, setToken] = useState("");
  const [manualToken, setManualToken] = useState("");
  const [authState, setAuthState] = useState("not_configured");
  const [mcpToolName, setMcpToolName] = useState("GetServices");
  const [mcpArgs, setMcpArgs] = useState('{"scenario_id":772}');
  const [mcpSessionId, setMcpSessionId] = useState("");
  const [mcpResult, setMcpResult] = useState("");
  const [urlChecks, setUrlChecks] = useState([]);
  const [checkingUrls, setCheckingUrls] = useState(false);
  const abortRef = useRef(null);
  const authInitRef = useRef(false);
  const mapNodeRef = useRef(null);
  const mapRef = useRef(null);
  const layerGroupRef = useRef(null);
  const tileLayerRef = useRef(null);

  const tokenClaims = useMemo(() => parseJwt(token), [token]);

  useEffect(() => {
    saveSettings(settings);
  }, [settings]);

  useEffect(() => {
    const urlToken = getTokenFromLocation();
    if (!urlToken) return;

    setToken(urlToken);
    setManualToken(urlToken);
    setAuthState("manual_token");
    setNotice("Токен получен из URL и сохранён для запросов.");
    window.history.replaceState({}, "", window.location.pathname);
  }, []);

  useEffect(() => {
    if (skipAuthOnBoot || authInitRef.current || !hasKeycloakConfig()) return;
    authInitRef.current = true;
    initKeycloak().catch((error) => {
      setAuthState("error");
      setNotice(error.message);
    });
  }, []);

  useEffect(() => {
    setRequestText(EXAMPLES[agent][0]);
  }, [agent]);

  useEffect(() => {
    if (!mapNodeRef.current || mapRef.current) return;

    const map = L.map(mapNodeRef.current, { zoomControl: true }).setView([59.93, 30.31], 10);
    tileLayerRef.current = L.tileLayer(settings.tileUrl, {
      attribution: settings.tileAttribution,
      maxZoom: 20,
    }).addTo(map);
    layerGroupRef.current = L.layerGroup().addTo(map);
    mapRef.current = map;

    setTimeout(() => map.invalidateSize(), 0);
  }, []);

  useEffect(() => {
    if (!mapRef.current || !tileLayerRef.current) return;
    tileLayerRef.current.setUrl(settings.tileUrl);
    tileLayerRef.current.options.attribution = settings.tileAttribution;
  }, [settings.tileUrl, settings.tileAttribution]);

  useEffect(() => {
    if (!mapRef.current || !layerGroupRef.current) return;
    layerGroupRef.current.clearLayers();

    const bounds = [];
    layers
      .filter((layer) => layer.visible)
      .forEach((layer) => {
        const leafletLayer = L.geoJSON(layer.geojson, {
          style: {
            color: layer.color,
            weight: 2,
            opacity: 0.85,
            fillOpacity: 0.22,
          },
          pointToLayer: (_feature, latlng) =>
            L.circleMarker(latlng, {
              radius: 6,
              color: layer.color,
              fillColor: layer.color,
              fillOpacity: 0.75,
              weight: 2,
            }),
          onEachFeature: (feature, item) => {
            const props = feature?.properties ?? {};
            const body = Object.entries(props)
              .slice(0, 12)
              .map(([key, value]) => `<b>${key}</b>: ${String(value)}`)
              .join("<br />");
            item.bindPopup(`<b>${layer.name}</b>${body ? `<br />${body}` : ""}`);
          },
        });
        leafletLayer.addTo(layerGroupRef.current);
        const layerBounds = leafletLayer.getBounds?.();
        if (layerBounds?.isValid()) bounds.push(layerBounds);
      });

    if (bounds.length > 0) {
      const combined = bounds.reduce((acc, item) => acc.extend(item), bounds[0]);
      mapRef.current.fitBounds(combined.pad(0.15), { maxZoom: 16 });
    }
  }, [layers]);

  function patchSettings(patch) {
    setSettings((current) => ({ ...current, ...patch }));
  }

  function resetSettings() {
    localStorage.removeItem(STORAGE_KEY);
    authInitRef.current = false;
    setSettings(defaultSettings);
    setKeycloak(null);
    setToken("");
    setManualToken("");
    setAuthState("not_configured");
    setNotice("Настройки сброшены. Заполните Keycloak заново или продолжайте без авторизации.");
  }

  function applyManualToken() {
    const cleanToken = manualToken.trim().replace(/^Bearer\s+/i, "");
    setToken(cleanToken);
    setAuthState(cleanToken ? "manual_token" : "anonymous");
    setNotice(cleanToken ? "Ручной токен применён для запросов." : "Ручной токен очищен.");
  }

  function openAuthHelper() {
    const helperUrl = new URL(settings.authHelperUrl);
    helperUrl.searchParams.set("returnUrl", window.location.origin + window.location.pathname);
    window.open(helperUrl.toString(), "_blank", "noopener,noreferrer");
  }

  async function checkOneUrl(label, url, mode, init = {}) {
    const target = mode === "proxy" ? proxiedUrl(url) : url.toString();
    const started = performance.now();

    try {
      const response = await safeFetch(
        target,
        {
          method: "GET",
          headers: { Accept: "application/json, text/plain, */*" },
          ...init,
        },
        mode,
      );
      const text = await response.text();
      return {
        label,
        mode,
        url: url.toString(),
        ok: response.ok,
        status: response.status,
        ms: Math.round(performance.now() - started),
        detail: text.slice(0, 800) || response.statusText,
      };
    } catch (error) {
      return {
        label,
        mode,
        url: url.toString(),
        ok: false,
        status: "fetch failed",
        ms: Math.round(performance.now() - started),
        detail: error.message,
      };
    }
  }

  async function checkConfiguredUrls() {
    setCheckingUrls(true);
    setUrlChecks([]);

    try {
      const agentsPing = joinUrl(settings.agentsBaseUrl, "/ping");
      const mcpHealth = mcpHealthUrl(settings.mcpUrl);
      const authHelper = asUrl(settings.authHelperUrl);
      const checks = [];

      for (const mode of ["proxy", "direct"]) {
        checks.push(await checkOneUrl("Agents /ping", agentsPing, mode));
        checks.push(await checkOneUrl("MCP /health", mcpHealth, mode));
        checks.push(await checkOneUrl("Auth helper", authHelper, mode));
      }

      setUrlChecks(checks);
      const failed = checks.filter((check) => !check.ok);
      setNotice(
        failed.length
          ? `Проверка URL завершена: ${failed.length} проверок упало. Подробности в блоке "Проверка URL".`
          : "Все URL доступны в direct и proxy режимах.",
      );
    } catch (error) {
      setNotice(error.message);
      setUrlChecks([
        {
          label: "Config parse",
          mode: "local",
          url: "",
          ok: false,
          status: "invalid",
          ms: 0,
          detail: error.message,
        },
      ]);
    } finally {
      setCheckingUrls(false);
    }
  }

  function hasKeycloakConfig() {
    return Boolean(settings.keycloakUrl && settings.keycloakRealm && settings.keycloakClientId);
  }

  async function initKeycloak() {
    if (!hasKeycloakConfig()) {
      setAuthState("not_configured");
      setNotice("Keycloak не настроен. Заполните URL, realm и clientId.");
      return null;
    }

    const kc = new Keycloak({
      url: settings.keycloakUrl,
      realm: settings.keycloakRealm,
      clientId: settings.keycloakClientId,
    });

    setAuthState("initializing");
    const authenticated = await kc.init({
      onLoad: "check-sso",
      pkceMethod: "S256",
      checkLoginIframe: false,
      redirectUri: settings.redirectUri,
    });

    setKeycloak(kc);
    setToken(kc.token ?? "");
    setAuthState(authenticated ? "authenticated" : "anonymous");
    return kc;
  }

  async function login() {
    const kc = keycloak ?? (await initKeycloak());
    if (kc) {
      await kc.login({ redirectUri: settings.redirectUri });
    }
  }

  async function logout() {
    if (!keycloak) return;
    await keycloak.logout({ redirectUri: settings.redirectUri });
  }

  async function refreshAccessToken(minValidity = 60) {
    if (!keycloak) {
      const kc = await initKeycloak();
      if (!kc) throw new Error("Keycloak is not configured");
      await kc.updateToken(minValidity);
      setToken(kc.token ?? "");
      return kc.token ?? "";
    }

    await keycloak.updateToken(minValidity);
    setToken(keycloak.token ?? "");
    setAuthState(keycloak.authenticated ? "authenticated" : "anonymous");
    return keycloak.token ?? "";
  }

  function resetRun() {
    setAnswer("");
    setEvents([]);
    setTables([]);
    setLayers([]);
    setNotice("");
  }

  function appendEvent(event) {
    setEvents((current) => [{ at: new Date().toISOString(), event }, ...current].slice(0, 200));
  }

  async function postPipelineToken(id) {
    const freshToken = await refreshAccessToken(-1);
    const url = joinUrl(settings.agentsBaseUrl, `/pipelines/${id}/token`);
    const target = viaProxy(url, settings);
    const response = await safeFetch(target, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${freshToken}`,
        Accept: "application/json",
      },
    }, settings.requestMode);

    if (!response.ok) {
      throw new Error(`Token refresh endpoint failed: ${response.status} ${await response.text()}`);
    }
  }

  async function handleStreamEvent(event) {
    appendEvent(event);

    if (event.type === "pipeline_started") {
      setRequestId(event.content.request_id);
      return;
    }

    if (event.type === "service_event" && event.content?.event?.storage_event_type === "chat_created") {
      setChatId(event.content.event.chat_id);
      return;
    }

    if (event.type === "status") {
      setNotice(event.content.text ?? event.content.status ?? "Статус обновлён");
      return;
    }

    if (event.type === "chunk") {
      setAnswer((current) => current + (event.content.text ?? ""));
      if (event.content.done) setStreamStatus("done");
      return;
    }

    if (event.type === "feature_collection") {
      const index = layers.length;
      setLayers((current) => [
        ...current,
        {
          id: `${Date.now()}-${current.length}`,
          name: event.content.name ?? `Layer ${current.length + 1}`,
          geojson: event.content.feature_collection,
          color: COLORS[current.length % COLORS.length],
          visible: true,
          count: event.content.feature_collection?.features?.length ?? 0,
        },
      ]);
      setNotice(`Добавлен слой ${event.content.name ?? index + 1}`);
      return;
    }

    if (event.type === "table") {
      setTables((current) => [...current, event.content]);
      return;
    }

    if (event.type === "token_expired") {
      const id = event.content.request_id || requestId;
      setNotice("Токен истёк, обновляю через Keycloak и отправляю в pipeline...");
      await postPipelineToken(id);
      setNotice("Токен обновлён, pipeline продолжает работу.");
      return;
    }

    if (event.type === "pipeline_suspended") {
      setNotice(event.content.message ?? "Pipeline приостановлен.");
      setStreamStatus("suspended");
      return;
    }

    if (event.type === "warning" || event.type === "error") {
      setNotice(event.content.message ?? event.type);
    }
  }

  async function startStream() {
    resetRun();
    setStreamStatus("connecting");

    const activeAgent = AGENTS[agent];
    const url = joinUrl(settings.agentsBaseUrl, activeAgent.path);
    url.searchParams.set("request", requestText);
    url.searchParams.set("model", model);
    url.searchParams.set("temperature", temperature);
    if (scenarioId || activeAgent.needsScenario) url.searchParams.set("scenario_id", scenarioId);
    if (chatId) url.searchParams.set("chat_id", chatId);
    if (requestId) url.searchParams.set("request_id", requestId);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const activeToken = token || (hasKeycloakConfig() ? await refreshAccessToken(30) : "");
      const target = viaProxy(url, settings);
      const response = await safeFetch(target, {
        headers: {
          Accept: "text/event-stream",
          ...(activeToken ? { Authorization: `Bearer ${activeToken}` } : {}),
        },
        signal: controller.signal,
      }, settings.requestMode);

      if (!response.ok || !response.body) {
        throw new Error(`Stream failed: ${response.status} ${await response.text()}`);
      }

      setStreamStatus("running");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseMessages(buffer);
        buffer = parsed.rest;

        for (const event of parsed.events) {
          await handleStreamEvent(event);
        }
      }

      setStreamStatus((current) => (current === "running" ? "done" : current));
    } catch (error) {
      if (error.name === "AbortError") {
        setStreamStatus("aborted");
        setNotice("Поток остановлен пользователем.");
      } else {
        setStreamStatus("error");
        setNotice(error.message);
        appendEvent({ type: "client_error", content: { message: error.message } });
      }
    }
  }

  function stopStream() {
    abortRef.current?.abort();
  }

  async function mcpRequest(method, params = undefined, isNotification = false) {
    const url = new URL(settings.mcpUrl);
    const id = Date.now();
    const body = isNotification ? { jsonrpc: "2.0", method, params } : { jsonrpc: "2.0", id, method, params };
    const headers = {
      "Content-Type": "application/json",
      Accept: "application/json, text/event-stream",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(mcpSessionId ? { "Mcp-Session-Id": mcpSessionId } : {}),
    };

    const target = viaProxy(url, settings);
    const response = await safeFetch(target, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    }, settings.requestMode);

    const session = response.headers.get("mcp-session-id");
    if (session) setMcpSessionId(session);

    const contentType = response.headers.get("content-type") ?? "";
    const text = await response.text();
    if (!response.ok) {
      throw new Error(`${response.status}: ${text}`);
    }

    if (contentType.includes("text/event-stream")) {
      const parsed = parseSseMessages(text);
      return parsed.events.length ? parsed.events : text;
    }

    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }

  async function runMcpAction(action) {
    setMcpResult("Выполняю...");
    try {
      let result;
      if (action === "initialize") {
        result = await mcpRequest("initialize", {
          protocolVersion: "2024-11-05",
          capabilities: {},
          clientInfo: { name: "gmart-test-ui", version: "0.1.0" },
        });
        await mcpRequest("notifications/initialized", undefined, true);
      } else if (action === "tools/list") {
        result = await mcpRequest("tools/list", {});
      } else if (action === "prompts/list") {
        result = await mcpRequest("prompts/list", {});
      } else if (action === "tools/call") {
        result = await mcpRequest("tools/call", {
          name: mcpToolName,
          arguments: JSON.parse(mcpArgs || "{}"),
        });
      }
      setMcpResult(prettyJson(result));
    } catch (error) {
      setMcpResult(error.message);
    }
  }

  function toggleLayer(id) {
    setLayers((current) =>
      current.map((layer) => (layer.id === id ? { ...layer, visible: !layer.visible } : layer)),
    );
  }

  return (
    <main className={`app-shell theme-${settings.theme}`}>
      <section className="topbar">
        <div>
          <h1>gMART Test UI</h1>
          <p>Песочница для SSE-агентов, Keycloak-токенов, GeoJSON-слоёв и MCP-инструментов.</p>
        </div>
        <div className={`status-pill ${streamStatus}`}>{streamStatus}</div>
      </section>

      <section className="layout">
        <aside className="sidebar">
          <Panel title="Настройки">
            <Field label="Agents API URL">
              <input value={settings.agentsBaseUrl} onChange={(e) => patchSettings({ agentsBaseUrl: e.target.value })} />
            </Field>
            <Field label="MCP URL">
              <input value={settings.mcpUrl} onChange={(e) => patchSettings({ mcpUrl: e.target.value })} />
            </Field>
            <Field label="Режим запросов">
              <select value={settings.requestMode} onChange={(e) => patchSettings({ requestMode: e.target.value })}>
                <option value="proxy">Vite proxy</option>
                <option value="direct">Direct browser fetch</option>
              </select>
            </Field>
            <Field label="Тема">
              <select value={settings.theme} onChange={(e) => patchSettings({ theme: e.target.value })}>
                <option value="light">Светлая</option>
                <option value="dark">Тёмная</option>
              </select>
            </Field>
            <Field label="Auth helper URL">
              <input value={settings.authHelperUrl} onChange={(e) => patchSettings({ authHelperUrl: e.target.value })} />
            </Field>
            <Field label="Keycloak URL">
              <input placeholder="https://keycloak.example.com" value={settings.keycloakUrl} onChange={(e) => patchSettings({ keycloakUrl: e.target.value })} />
            </Field>
            <div className="two-columns">
              <Field label="Realm">
                <input value={settings.keycloakRealm} onChange={(e) => patchSettings({ keycloakRealm: e.target.value })} />
              </Field>
              <Field label="Client ID">
                <input value={settings.keycloakClientId} onChange={(e) => patchSettings({ keycloakClientId: e.target.value })} />
              </Field>
            </div>
            <Field label="Redirect URI">
              <input value={settings.redirectUri} onChange={(e) => patchSettings({ redirectUri: e.target.value })} />
            </Field>
            <Field label="Tile URL">
              <input value={settings.tileUrl} onChange={(e) => patchSettings({ tileUrl: e.target.value })} />
            </Field>
            <Field label="Tile attribution">
              <input value={settings.tileAttribution} onChange={(e) => patchSettings({ tileAttribution: e.target.value })} />
            </Field>
            <div className="button-row">
              <button onClick={resetSettings}>Сбросить настройки</button>
              <button onClick={checkConfiguredUrls} disabled={checkingUrls}>
                {checkingUrls ? "Проверяю..." : "Проверить URL"}
              </button>
            </div>
          </Panel>

          <Panel title="Авторизация">
            <div className="button-row">
              <button onClick={openAuthHelper}>Открыть auth helper</button>
              <button onClick={initKeycloak}>Инициализировать</button>
              <button onClick={login}>Войти</button>
              <button onClick={() => refreshAccessToken(-1)}>Refresh</button>
              <button onClick={logout}>Выйти</button>
            </div>
            <Field label="Manual Bearer token">
              <textarea
                rows={3}
                placeholder="Вставьте access token или Bearer ..."
                value={manualToken}
                onChange={(e) => setManualToken(e.target.value)}
              />
            </Field>
            <div className="button-row">
              <button onClick={applyManualToken}>Применить токен</button>
              <button
                onClick={() => {
                  setManualToken("");
                  setToken("");
                  setAuthState("anonymous");
                }}
              >
                Очистить токен
              </button>
            </div>
            <dl className="facts">
              <dt>Состояние</dt>
              <dd>{authState}</dd>
              <dt>Пользователь</dt>
              <dd>{tokenClaims?.preferred_username ?? tokenClaims?.email ?? "—"}</dd>
              <dt>Истекает</dt>
              <dd>{formatTimeFromSeconds(tokenClaims?.exp)}</dd>
            </dl>
          </Panel>

          <Panel title="Запрос агента">
            <Field label="Агент">
              <select value={agent} onChange={(e) => setAgent(e.target.value)}>
                {Object.entries(AGENTS).map(([key, value]) => (
                  <option key={key} value={key}>
                    {value.label}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Примеры">
              <select value="" onChange={(e) => e.target.value && setRequestText(e.target.value)}>
                <option value="">Выбрать пример...</option>
                {EXAMPLES[agent].map((example) => (
                  <option key={example} value={example}>
                    {example}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Промпт">
              <textarea rows={5} value={requestText} onChange={(e) => setRequestText(e.target.value)} />
            </Field>
            <div className="two-columns">
              <Field label="Scenario ID">
                <input value={scenarioId} onChange={(e) => setScenarioId(e.target.value)} />
              </Field>
              <Field label="Temperature">
                <input value={temperature} onChange={(e) => setTemperature(e.target.value)} />
              </Field>
            </div>
            <Field label="Model">
              <input value={model} onChange={(e) => setModel(e.target.value)} />
            </Field>
            <Field label="Chat ID">
              <input placeholder="заполнится после chat_created" value={chatId} onChange={(e) => setChatId(e.target.value)} />
            </Field>
            <Field label="Request ID">
              <input placeholder="для reconnect/resume" value={requestId} onChange={(e) => setRequestId(e.target.value)} />
            </Field>
            <div className="button-row">
              <button className="primary" onClick={startStream} disabled={streamStatus === "running" || streamStatus === "connecting"}>
                Запустить
              </button>
              <button onClick={stopStream}>Остановить</button>
              <button onClick={resetRun}>Очистить</button>
            </div>
          </Panel>
        </aside>

        <section className="workspace">
          <section className="map-pane">
            <div className="map-header">
              <h2>Карта слоёв</h2>
              <span>{layers.length} слоёв</span>
            </div>
            <div ref={mapNodeRef} className="map" />
            <div className="layers-list">
              {layers.length === 0 && <span className="muted">GeoJSON-слои появятся здесь после событий feature_collection.</span>}
              {layers.map((layer) => (
                <label key={layer.id} className="layer-item">
                  <input type="checkbox" checked={layer.visible} onChange={() => toggleLayer(layer.id)} />
                  <span className="swatch" style={{ background: layer.color }} />
                  <span>{layer.name}</span>
                  <small>{layer.count} features</small>
                </label>
              ))}
            </div>
          </section>

          <section className="result-grid">
            <Panel title="Ответ">
              {notice && <div className="notice">{notice}</div>}
              <pre className="answer">{answer || "Текстовые chunk-события будут собираться здесь."}</pre>
            </Panel>

            <Panel title="Таблицы">
              {tables.length === 0 && <p className="muted">Provision-таблицы появятся здесь.</p>}
              {tables.map((table, index) => (
                <div className="table-wrap" key={`${table.name}-${index}`}>
                  <h3>{table.title ?? table.name}</h3>
                  <table>
                    <thead>
                      <tr>
                        {table.columns.map((column) => (
                          <th key={column.key}>{column.label}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {table.rows.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {table.columns.map((column) => (
                            <td key={column.key}>{String(row[column.key] ?? "")}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </Panel>

            <Panel title="MCP инструменты">
              <div className="button-row">
                <button onClick={() => runMcpAction("initialize")}>Initialize</button>
                <button onClick={() => runMcpAction("tools/list")}>tools/list</button>
                <button onClick={() => runMcpAction("prompts/list")}>prompts/list</button>
              </div>
              <Field label="MCP session id">
                <input value={mcpSessionId} onChange={(e) => setMcpSessionId(e.target.value)} />
              </Field>
              <div className="two-columns">
                <Field label="Tool name">
                  <input value={mcpToolName} onChange={(e) => setMcpToolName(e.target.value)} />
                </Field>
                <div className="field button-field">
                  <button className="primary" onClick={() => runMcpAction("tools/call")}>
                    tools/call
                  </button>
                </div>
              </div>
              <Field label="Arguments JSON">
                <textarea rows={4} value={mcpArgs} onChange={(e) => setMcpArgs(e.target.value)} />
              </Field>
              <pre className="code-box">{mcpResult || "MCP-ответ появится здесь."}</pre>
            </Panel>

            <Panel title="Проверка URL">
              {urlChecks.length === 0 && (
                <p className="muted">Нажмите "Проверить URL" в настройках, чтобы проверить текущие Agents/MCP/Auth helper адреса.</p>
              )}
              <div className="checks">
                {urlChecks.map((check, index) => (
                  <details key={`${check.label}-${check.mode}-${index}`} open={!check.ok}>
                    <summary>
                      <span>
                        {check.ok ? "OK" : "FAIL"} · {check.label} · {check.mode}
                      </span>
                      <small>{check.status} · {check.ms} ms</small>
                    </summary>
                    <pre>{`URL: ${check.url}\n\n${check.detail}`}</pre>
                  </details>
                ))}
              </div>
            </Panel>

            <Panel title="SSE события">
              <div className="events">
                {events.length === 0 && <p className="muted">События потока появятся здесь.</p>}
                {events.map((item, index) => (
                  <details key={`${item.at}-${index}`}>
                    <summary>
                      <span>{item.event.type}</span>
                      <small>{new Date(item.at).toLocaleTimeString()}</small>
                    </summary>
                    <pre>{prettyJson(item.event.content)}</pre>
                  </details>
                ))}
              </div>
            </Panel>
          </section>
        </section>
      </section>
    </main>
  );
}

function Panel({ title, children }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

createRoot(document.getElementById("root")).render(<App />);
