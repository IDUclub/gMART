import { useEffect, useMemo, useRef, useState } from "react";
import Keycloak from "keycloak-js";
import ReactMarkdown from "react-markdown";
import MapPanel from "./MapPanel";
import {
  deleteChat,
  getChat,
  getChats,
  getModels,
  readSse,
  request,
} from "./api";
import type {
  Agent,
  AgentId,
  Chat,
  ChatSummary,
  LayerData,
  Message,
  Settings,
  StreamEvent,
  TableData,
} from "./types";
const AGENTS: Agent[] = [
  {
    id: "orchestrator",
    icon: "⌬",
    label: "Оркестратор",
    caption: "Единая точка входа",
    path: "/orchestrator/route/stream",
    needsScenario: false,
    examples: [
      "Построй зону ограничения вокруг школ 200 метров и найди требования к их размещению",
      "Оцени обеспеченность детскими садами и проверь нормативные ограничения",
    ],
  },
  {
    id: "restrictions",
    icon: "◈",
    label: "Ограничения",
    caption: "Геозоны и буферы",
    path: "/restrictions/generate_restrictions/stream",
    needsScenario: true,
    examples: [
      "Построй зону ограничения вокруг школ 200 метров",
      "Какие магазины попадают в радиус 200 метров от школ?",
    ],
  },
  {
    id: "provision",
    icon: "◎",
    label: "Обеспеченность",
    caption: "Сервисы и эффекты",
    path: "/provision/calculate_effects/stream",
    needsScenario: true,
    examples: [
      "Дай сводку по обеспеченности сервисами",
      "Как проект повлияет на обеспеченность детскими садами?",
    ],
  },
  {
    id: "documents",
    icon: "▤",
    label: "Документы",
    caption: "Поиск по IDU_DVD",
    path: "/documents/qa/stream",
    needsScenario: false,
    examples: [
      "Какие требования влияют на размещение школ?",
      "Найди требования к радиусам доступности детских садов",
    ],
  },
  {
    id: "norms",
    icon: "⌘",
    label: "Нормы",
    caption: "Граф NormGraph",
    path: "/norms/qa/stream",
    needsScenario: false,
    examples: [
      "Какие ограничения действуют для жилой застройки рядом со школой?",
      "Проверь противоречия в требованиях к санитарным зонам",
    ],
  },
  {
    id: "llm",
    icon: "✦",
    label: "Ассистент",
    caption: "Свободный диалог",
    path: "/llm/message/stream",
    needsScenario: false,
    examples: ["Помоги сформулировать запрос для анализа территории"],
  },
];
const defaults: Settings = {
  theme: "dark",
  basemap: "cartoDark",
  agentsUrl: location.pathname.startsWith("/ui")
    ? location.origin
    : "http://127.0.0.1:80",
  chatStorageUrl: "http://127.0.0.1:8010",
  authHelperUrl: "https://idu-auth-helper.idulab.ru/",
  keycloakUrl: "",
  keycloakRealm: "",
  keycloakClientId: "",
  model: "gpt-oss:20b",
  temperature: 1,
};
const colors = [
  "#39d98a",
  "#55a8ff",
  "#ffb84d",
  "#d77dff",
  "#ff6b7a",
  "#35c9ce",
];
function load() {
  try {
    return {
      ...defaults,
      ...JSON.parse(localStorage.getItem("gmart-ui") || "{}"),
    };
  } catch {
    return defaults;
  }
}
export default function App() {
  const [settings, setSettings] = useState<Settings>(load),
    [agentId, setAgentId] = useState<AgentId>("restrictions"),
    [mode, setMode] = useState<"workspace" | "admin">("workspace"),
    [scenario, setScenario] = useState("772"),
    [project, setProject] = useState(""),
    [token, setToken] = useState(""),
    [auth, setAuth] = useState("loading"),
    [chats, setChats] = useState<ChatSummary[]>([]),
    [chat, setChat] = useState<Chat | null>(null),
    [query, setQuery] = useState(""),
    [answer, setAnswer] = useState(""),
    [layers, setLayers] = useState<LayerData[]>([]),
    [tables, setTables] = useState<TableData[]>([]),
    [events, setEvents] = useState<Array<{ time: string; event: StreamEvent }>>(
      [],
    ),
    [status, setStatus] = useState("Готов к работе"),
    [busy, setBusy] = useState(false),
    [rightTab, setRightTab] = useState<"map" | "data" | "process">("map"),
    [models, setModels] = useState<string[]>([]),
    [settingsOpen, setSettingsOpen] = useState(false),
    [systemPassword, setSystemPassword] = useState(""),
    [systemConfig, setSystemConfig] = useState<Record<string, string> | null>(
      null,
    );
  const abort = useRef<AbortController | null>(null),
    kc = useRef<Keycloak | null>(null),
    stepBase = useRef("");
  const agent = AGENTS.find((a) => a.id === agentId)!;
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme;
    localStorage.setItem("gmart-ui", JSON.stringify(settings));
  }, [settings]);
  useEffect(() => {
    const urlToken =
      new URLSearchParams(location.search).get("access_token") ||
      new URLSearchParams(location.hash.slice(1)).get("access_token");
    if (urlToken) {
      setToken(urlToken);
      setAuth("ready");
      window.history.replaceState({}, "", location.pathname);
      return;
    }
    if (!settings.keycloakUrl) {
      setAuth("anonymous");
      return;
    }
    const client = new Keycloak({
      url: settings.keycloakUrl,
      realm: settings.keycloakRealm,
      clientId: settings.keycloakClientId,
    });
    kc.current = client;
    client
      .init({
        onLoad: "check-sso",
        pkceMethod: "S256",
        checkLoginIframe: false,
      })
      .then((ok) => {
        setToken(client.token || "");
        setAuth(ok ? "ready" : "anonymous");
      })
      .catch(() => setAuth("error"));
  }, []);
  useEffect(() => {
    if (!token) return;
    loadChats();
    getModels(settings, token)
      .then(setModels)
      .catch(() => {});
  }, [token, scenario]);
  useEffect(() => setQuery(agent.examples[0]), [agentId]);
  async function loadChats() {
    try {
      setChats((await getChats(settings, token, scenario)).items);
    } catch (e) {
      setStatus(err(e));
    }
  }
  async function openChat(id: string) {
    try {
      setChat(await getChat(settings, token, id));
      setAnswer("");
      setLayers([]);
      setTables([]);
    } catch (e) {
      setStatus(err(e));
    }
  }
  async function removeChat(id: string) {
    if (!confirm("Удалить этот диалог?")) return;
    await deleteChat(settings, token, id);
    if (chat?.chat_id === id) setChat(null);
    loadChats();
  }
  function login() {
    if (kc.current) kc.current.login();
    else {
      const url = new URL(settings.authHelperUrl);
      url.searchParams.set("returnUrl", location.origin + location.pathname);
      location.href = url.toString();
    }
  }
  async function freshToken() {
    if (kc.current) {
      await kc.current.updateToken(-1);
      setToken(kc.current.token || "");
      return kc.current.token || "";
    }
    return token;
  }
  function handle(event: StreamEvent) {
    setEvents((v) =>
      [{ time: new Date().toLocaleTimeString(), event }, ...v].slice(0, 100),
    );
    route(event);
  }
  function route(event: StreamEvent, nested = false) {
    if (event.type === "pipeline_started") setStatus("Агент начал работу");
    if (event.type === "status")
      setStatus(event.content?.text || labelStatus(event.content?.status));
    if (event.type === "chunk" || event.type === "Text") {
      const text =
        typeof event.content === "string"
          ? event.content
          : event.content?.text || "";
      setAnswer((v) =>
        event.content?.iteration && event.content.iteration > 1
          ? stepBase.current + text
          : v + text,
      );
      if (event.content?.done && !nested) {
        setBusy(false);
        setStatus("Ответ готов");
        loadChats();
      }
    }
    if (event.type === "plan") {
      const steps = event.content?.steps || [];
      setStatus(`План готов: шагов — ${steps.length}`);
      if (steps.length)
        setAnswer(
          (v) =>
            v +
            "**План работы**\n\n" +
            steps
              .map(
                (s: any) =>
                  `${s.step}. ${s.agent_title || labelAgent(s.agent)} — ${s.task}`,
              )
              .join("\n") +
            "\n\n",
        );
    }
    if (event.type === "step_started") {
      const step = event.content?.step,
        agent = labelAgent(event.content?.agent);
      setStatus(`Шаг ${step}: ${agent}`);
      setAnswer(
        (v) => (stepBase.current = `${v}---\n\n**Шаг ${step} · ${agent}**\n\n`),
      );
    }
    if (event.type === "step_event" && event.content?.event)
      route(event.content.event, true);
    if (event.type === "step_finished") {
      if (event.content?.status === "failed")
        setAnswer(
          (v) =>
            v +
            `\n\n> ⚠ Шаг ${event.content.step} не выполнен: ${event.content.summary || "ошибка агента"}\n\n`,
        );
      setStatus(`Шаг ${event.content?.step} завершён`);
    }
    if (event.type === "clarification") {
      setAnswer((v) => v + (event.content?.question || ""));
      setStatus("Нужно уточнение");
    }
    if (event.type === "orchestrator_final") {
      setStatus("Ответ готов");
      loadChats();
    }
    if (event.type === "feature_collection") {
      const fc =
        event.content?.feature_collection ||
        event.content?.data ||
        event.content;
      setLayers((v) => [
        ...v,
        {
          id: crypto.randomUUID(),
          name: event.content?.name || `Слой ${v.length + 1}`,
          color: colors[v.length % colors.length],
          visible: true,
          geojson: fc,
          count: fc?.features?.length || 0,
        },
      ]);
      setRightTab("map");
    }
    if (event.type === "table") {
      setTables((v) => [...v, event.content]);
      setRightTab("data");
    }
    if (event.type === "warning" || event.type === "error")
      setStatus(event.content?.message || "Ошибка выполнения");
    if (event.type === "service_event" && event.content?.event?.chat_id)
      setChat((v) => (v ? { ...v, chat_id: event.content.event.chat_id } : v));
    if (event.type === "token_expired")
      refreshPipeline(event.content?.request_id);
  }
  async function refreshPipeline(id: string) {
    const t = await freshToken();
    await request(settings.agentsUrl, `/pipelines/${id}/token`, t, {
      method: "POST",
      body: JSON.stringify({ token: t }),
    });
  }
  async function submit() {
    if (!token) {
      login();
      return;
    }
    if (!query.trim() || busy) return;
    setBusy(true);
    setAnswer("");
    setTables([]);
    setEvents([]);
    stepBase.current = "";
    setStatus("Подключение к агенту…");
    const url = new URL(agent.path, settings.agentsUrl);
    url.searchParams.set("request", query);
    url.searchParams.set("model", settings.model);
    url.searchParams.set("temperature", String(settings.temperature));
    if (scenario) url.searchParams.set("scenario_id", scenario);
    if (chat?.chat_id) url.searchParams.set("chat_id", chat.chat_id);
    abort.current = new AbortController();
    try {
      await readSse(url, await freshToken(), abort.current.signal, handle);
      setBusy(false);
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setStatus(err(e));
        setBusy(false);
      }
    }
  }
  async function loadSystem() {
    try {
      setSystemConfig(
        await request(settings.agentsUrl, "/system/config", token, {
          method: "POST",
          body: JSON.stringify({ password: systemPassword }),
        }),
      );
    } catch (e) {
      setStatus(err(e));
    }
  }
  const history = useMemo(() => chat?.messages || [], [chat]);
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">g</div>
          <div>
            gMART<small>GEOSPATIAL INTELLIGENCE</small>
          </div>
        </div>
        <nav>
          <button
            className={mode === "workspace" ? "active" : ""}
            onClick={() => setMode("workspace")}
          >
            <span>⌂</span>Рабочее пространство
          </button>
          {AGENTS.map((a) => (
            <button
              className={
                mode === "workspace" && agentId === a.id ? "active sub" : "sub"
              }
              onClick={() => {
                setMode("workspace");
                setAgentId(a.id);
              }}
              key={a.id}
            >
              <span>{a.icon}</span>
              <div>
                {a.label}
                <small>{a.caption}</small>
              </div>
            </button>
          ))}
          <button
            className={mode === "admin" ? "active" : ""}
            onClick={() => setMode("admin")}
          >
            <span>⚙</span>Система
          </button>
        </nav>
        <div className="side-bottom">
          <button onClick={() => setSettingsOpen(true)}>Настройки</button>
          <button
            onClick={() =>
              setSettings((s) => ({
                ...s,
                theme: s.theme === "dark" ? "light" : "dark",
                basemap: s.theme === "dark" ? "cartoLight" : "cartoDark",
              }))
            }
          >
            {settings.theme === "dark" ? "☀" : "◐"}
          </button>
        </div>
      </aside>
      <main>
        {mode === "workspace" ? (
          <>
            <header>
              <div>
                <span className="eyebrow">РАБОЧЕЕ ПРОСТРАНСТВО</span>
                <h1>{agent.label}</h1>
                <p>{agent.caption}</p>
              </div>
              <div className="context">
                <label>
                  Сценарий
                  <input
                    value={scenario}
                    onChange={(e) => setScenario(e.target.value)}
                    placeholder="ID"
                  />
                </label>
                <label>
                  Проект
                  <input
                    value={project}
                    onChange={(e) => setProject(e.target.value)}
                    placeholder="необязательно"
                  />
                </label>
                <span className={`connection ${busy ? "pulse" : ""}`}>
                  <i />
                  {status}
                </span>
                {auth !== "ready" && (
                  <button className="primary" onClick={login}>
                    Войти
                  </button>
                )}
              </div>
            </header>
            <div className="work-grid">
              <section className="history-panel">
                <div className="panel-head">
                  <strong>Диалоги</strong>
                  <button
                    onClick={() => {
                      setChat(null);
                      setAnswer("");
                    }}
                  >
                    ＋
                  </button>
                </div>
                <div className="search">
                  ⌕ <input placeholder="Поиск по диалогам" />
                </div>
                <div className="chat-list">
                  {chats.map((c) => (
                    <div
                      className={`chat-row ${chat?.chat_id === c.chat_id ? "active" : ""}`}
                      key={c.chat_id}
                    >
                      <button onClick={() => openChat(c.chat_id)}>
                        <strong>{c.title || "Новый диалог"}</strong>
                        <small>
                          {new Date(c.updated_at).toLocaleDateString("ru")}
                        </small>
                      </button>
                      <button
                        className="delete"
                        onClick={() => removeChat(c.chat_id)}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                  {token && !chats.length && (
                    <div className="empty">История пока пуста</div>
                  )}
                </div>
              </section>
              <section className="conversation">
                <div className="messages">
                  {!history.length && !answer ? (
                    <Welcome agent={agent} onExample={setQuery} />
                  ) : (
                    <>
                      {history.map((m) => (
                        <MessageView key={m.message_id} message={m} />
                      ))}
                      {answer && (
                        <div className="message assistant">
                          <div className="avatar">g</div>
                          <div>
                            <ReactMarkdown>{answer}</ReactMarkdown>
                          </div>
                        </div>
                      )}
                      {busy && (
                        <div className="thinking">
                          <i />
                          <i />
                          <i /> {status}
                        </div>
                      )}
                    </>
                  )}
                </div>
                <div className="composer">
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        submit();
                      }
                    }}
                    placeholder={`Задайте вопрос: ${agent.examples[0]}`}
                  />
                  <div>
                    <span>Enter — отправить · Shift+Enter — новая строка</span>
                    <button
                      onClick={() =>
                        busy
                          ? (abort.current?.abort(), setBusy(false))
                          : submit()
                      }
                      className="send"
                    >
                      {busy ? "■" : "↑"}
                    </button>
                  </div>
                </div>
              </section>
              <section className="inspector">
                <div className="tabs">
                  <button
                    className={rightTab === "map" ? "active" : ""}
                    onClick={() => setRightTab("map")}
                  >
                    Карта <b>{layers.length}</b>
                  </button>
                  <button
                    className={rightTab === "data" ? "active" : ""}
                    onClick={() => setRightTab("data")}
                  >
                    Данные <b>{tables.length}</b>
                  </button>
                  <button
                    className={rightTab === "process" ? "active" : ""}
                    onClick={() => setRightTab("process")}
                  >
                    Ход работы
                  </button>
                </div>
                {rightTab === "map" && (
                  <MapPanel
                    layers={layers}
                    basemap={settings.basemap}
                    onBasemap={(basemap) =>
                      setSettings((s) => ({ ...s, basemap }))
                    }
                    onToggle={(id) =>
                      setLayers((v) =>
                        v.map((l) =>
                          l.id === id ? { ...l, visible: !l.visible } : l,
                        ),
                      )
                    }
                  />
                )}{" "}
                {rightTab === "data" && <Tables tables={tables} />}{" "}
                {rightTab === "process" && <Process events={events} />}
              </section>
            </div>
          </>
        ) : (
          <Admin
            settings={settings}
            password={systemPassword}
            setPassword={setSystemPassword}
            config={systemConfig}
            load={loadSystem}
          />
        )}
      </main>
      {settingsOpen && (
        <SettingsModal
          settings={settings}
          setSettings={setSettings}
          close={() => setSettingsOpen(false)}
          models={models}
        />
      )}
    </div>
  );
}
function Welcome({
  agent,
  onExample,
}: {
  agent: Agent;
  onExample: (v: string) => void;
}) {
  return (
    <div className="welcome">
      <div className="hero-mark">{agent.icon}</div>
      <span className="eyebrow">АГЕНТ · {agent.label.toUpperCase()}</span>
      <h2>Что исследуем сегодня?</h2>
      <p>
        Опишите задачу естественным языком. Агент подберёт инструменты, покажет
        ход анализа и соберёт результат.
      </p>
      <div>
        {agent.examples.map((x) => (
          <button key={x} onClick={() => onExample(x)}>
            {x}
            <span>↗</span>
          </button>
        ))}
      </div>
    </div>
  );
}
function MessageView({ message }: { message: Message }) {
  return (
    <div className={`message ${message.role}`}>
      <div className="avatar">{message.role === "user" ? "В" : "g"}</div>
      <div>
        {message.parts.map((p) =>
          p.kind === "text" ? (
            <ReactMarkdown key={p.part_seq}>
              {String(p.payload.text || "")}
            </ReactMarkdown>
          ) : p.kind === "table" ? (
            <Tables key={p.part_seq} tables={[p.payload as TableData]} />
          ) : null,
        )}
      </div>
    </div>
  );
}
function Tables({ tables }: { tables: TableData[] }) {
  return (
    <div className="data-panel">
      {tables.length ? (
        tables.map((t, i) => (
          <div className="table-card" key={i}>
            <h3>{t.title || t.name || "Результаты"}</h3>
            <div>
              <table>
                <thead>
                  <tr>
                    {t.columns?.map((c) => (
                      <th key={c.key}>{c.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {t.rows?.map((r, n) => (
                    <tr key={n}>
                      {t.columns.map((c) => (
                        <td key={c.key}>{String(r[c.key] ?? "—")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))
      ) : (
        <div className="empty">Таблицы и показатели появятся здесь</div>
      )}
    </div>
  );
}
function Process({
  events,
}: {
  events: Array<{ time: string; event: StreamEvent }>;
}) {
  return (
    <div className="process">
      {events.length ? (
        events.map((x, i) => (
          <details key={i}>
            <summary>
              <span>{x.event.type}</span>
              <time>{x.time}</time>
            </summary>
            <pre>{JSON.stringify(x.event.content, null, 2)}</pre>
          </details>
        ))
      ) : (
        <div className="empty">
          Ход выполнения появится после запуска агента
        </div>
      )}
    </div>
  );
}
function Admin({
  settings,
  password,
  setPassword,
  config,
  load,
}: {
  settings: Settings;
  password: string;
  setPassword: (s: string) => void;
  config: Record<string, string> | null;
  load: () => void;
}) {
  const services = [
    "Agents API",
    "IDU MCP",
    "Redis",
    "Ollama",
    "IDU_DVD",
    "NormGraph",
    "ObjectEffectsAPI",
  ];
  return (
    <div className="admin">
      <header>
        <div>
          <span className="eyebrow">АДМИНИСТРИРОВАНИЕ</span>
          <h1>Состояние системы</h1>
          <p>Подключения, конфигурация и диагностика gMART</p>
        </div>
        <a
          className="button"
          href={new URL("/system/logs", settings.agentsUrl).toString()}
        >
          Скачать логи
        </a>
      </header>
      <div className="stat-grid">
        <article>
          <span>Компоненты</span>
          <strong>{services.length}</strong>
          <small>в контуре системы</small>
        </article>
        <article>
          <span>Agents API</span>
          <strong className="green">online</strong>
          <small>{settings.agentsUrl}</small>
        </article>
        <article>
          <span>Режим</span>
          <strong>Production</strong>
          <small>React UI + FastAPI</small>
        </article>
      </div>
      <section className="admin-panel">
        <div className="panel-head">
          <div>
            <h2>Подключения</h2>
            <p>Текущее состояние зависимых сервисов</p>
          </div>
        </div>
        <div className="service-grid">
          {services.map((x, i) => (
            <div key={x}>
              <i className={i ? "muted-dot" : ""} />
              <strong>{x}</strong>
              <small>
                {i ? "Статус доступен после проверки конфигурации" : "Доступен"}
              </small>
            </div>
          ))}
        </div>
      </section>
      <section className="admin-panel config">
        <h2>Конфигурация</h2>
        <p>Введите системный пароль для просмотра адресов подключений.</p>
        <div className="inline-form">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Системный пароль"
          />
          <button className="primary" onClick={load}>
            Загрузить
          </button>
        </div>
        {config && (
          <dl>
            {Object.entries(config).map(([k, v]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>
    </div>
  );
}
function SettingsModal({
  settings,
  setSettings,
  close,
  models,
}: {
  settings: Settings;
  setSettings: (s: Settings) => void;
  close: () => void;
  models: string[];
}) {
  const [s, setS] = useState(settings);
  return (
    <div
      className="modal"
      onMouseDown={(e) => e.target === e.currentTarget && close()}
    >
      <div className="modal-card">
        <div className="panel-head">
          <div>
            <span className="eyebrow">НАСТРОЙКИ</span>
            <h2>Среда и модель</h2>
          </div>
          <button onClick={close}>×</button>
        </div>
        <div className="form-grid">
          <label>
            Agents API
            <input
              value={s.agentsUrl}
              onChange={(e) => setS({ ...s, agentsUrl: e.target.value })}
            />
          </label>
          <label>
            ChatStorage
            <input
              value={s.chatStorageUrl}
              onChange={(e) => setS({ ...s, chatStorageUrl: e.target.value })}
            />
          </label>
          <label>
            Модель
            <select
              value={s.model}
              onChange={(e) => setS({ ...s, model: e.target.value })}
            >
              {[s.model, ...models.filter((x) => x !== s.model)].map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
          <label>
            Температура
            <input
              type="number"
              min="0"
              max="2"
              step=".1"
              value={s.temperature}
              onChange={(e) => setS({ ...s, temperature: +e.target.value })}
            />
          </label>
          <label>
            Keycloak URL
            <input
              value={s.keycloakUrl}
              onChange={(e) => setS({ ...s, keycloakUrl: e.target.value })}
            />
          </label>
          <label>
            Realm
            <input
              value={s.keycloakRealm}
              onChange={(e) => setS({ ...s, keycloakRealm: e.target.value })}
            />
          </label>
          <label>
            Client ID
            <input
              value={s.keycloakClientId}
              onChange={(e) => setS({ ...s, keycloakClientId: e.target.value })}
            />
          </label>
          <label>
            Auth helper
            <input
              value={s.authHelperUrl}
              onChange={(e) => setS({ ...s, authHelperUrl: e.target.value })}
            />
          </label>
        </div>
        <div className="modal-actions">
          <button onClick={close}>Отмена</button>
          <button
            className="primary"
            onClick={() => {
              setSettings(s);
              close();
            }}
          >
            Сохранить
          </button>
        </div>
      </div>
    </div>
  );
}
function err(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}
function labelAgent(key?: string) {
  return (
    (
      {
        restriction: "Ограничения",
        provision: "Обеспеченность",
        documents: "Документы",
        norms: "Нормы",
      } as Record<string, string>
    )[key || ""] ||
    key ||
    "агент"
  );
}
function labelStatus(s: string) {
  return (
    (
      {
        retrieval_planning: "Планирую поиск",
        searching: "Ищу источники",
        executing: "Выполняю инструменты",
        conflict_check: "Проверяю противоречия",
        answer_drafting: "Готовлю ответ",
        self_review: "Проверяю результат",
        finalizing: "Завершаю",
      } as Record<string, string>
    )[s] ||
    s ||
    "Выполняется"
  );
}
