import { useEffect, useMemo, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import {
  ArrowUp,
  ArrowCounterClockwise,
  Buildings,
  CaretDown,
  ChartDonut,
  CheckCircle,
  CirclesFour,
  ClockCounterClockwise,
  Command,
  Database,
  FileText,
  GearSix,
  List,
  MapTrifold,
  Moon,
  Plus,
  ShieldCheck,
  SlidersHorizontal,
  Sparkle,
  SquaresFour,
  Sun,
  TerminalWindow,
  Trash,
  X,
} from "@phosphor-icons/react";
import gsap from "gsap";
import Keycloak from "keycloak-js";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import MapPanel from "./MapPanel";
import McpConsole from "./McpConsole";
import {
  appendSseExchange,
  CHAT_PAGE_SIZE,
  mergeMessageWindow,
  oldestServerSequence,
} from "./chatWindow";
import { buildMessageBlocks, normalizeMessages } from "./messageHistory";
import {
  authAvailable,
  authLogin,
  deleteChat,
  getChat,
  getChats,
  getModels,
  readSse,
  replayToolCall,
  request,
} from "./api";
import type {
  Agent,
  AgentId,
  Chat,
  ChatSummary,
  LayerData,
  Message,
  MessagePart,
  Settings,
  StatusEntry,
  StreamEvent,
  TableData,
} from "./types";
gsap.registerPlugin(useGSAP);
const AGENTS: Agent[] = [
  {
    id: "orchestrator",
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
    id: "compliance",
    label: "Соответствие",
    caption: "Проверка по нормам",
    path: "/compliance/check/stream",
    needsScenario: true,
    examples: [
      "Проверь соответствие проекта нормативным ограничениям",
      "Какие объекты проекта нарушают противопожарные расстояния?",
    ],
  },
  {
    id: "provision",
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
    id: "scenario_data",
    label: "Городские данные",
    caption: "Справочники, объекты и слои",
    path: "/scenario-data/qa/stream",
    needsScenario: false,
    examples: [
      "Какие типы городских сервисов доступны?",
      "Какие объекты есть в сценарии и сколько их по типам?",
      "Покажи на карте физические объекты сценария",
    ],
  },
  {
    id: "documents",
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
  // Empty on purpose: the agents resolve the model from the provider's own list, so the
  // UI must not ship a backend-specific id (an Ollama-style "gpt-oss:20b" 404s on vLLM).
  model: "",
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
type ActiveExchange = {
  question: string;
  answer: string;
  tables: TableData[];
  finalized: boolean;
};
type HistoryWindow = {
  hasMore: boolean;
  nextBeforeSeq: number | null;
  loading: boolean;
};
type CachedChatWindow = { chat: Chat; history: HistoryWindow };
const emptyHistoryWindow: HistoryWindow = {
  hasMore: false,
  nextBeforeSeq: null,
  loading: false,
};
const MAX_CACHED_CHAT_WINDOWS = 6;
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
  const appRoot = useRef<HTMLDivElement>(null);
  const [settings, setSettings] = useState<Settings>(load),
    [agentId, setAgentId] = useState<AgentId>("restrictions"),
    [mode, setMode] = useState<"workspace" | "mcp" | "admin">("workspace"),
    [scenario, setScenario] = useState("772"),
    [project, setProject] = useState(""),
    [token, setToken] = useState(""),
    [auth, setAuth] = useState("loading"),
    [chats, setChats] = useState<ChatSummary[]>([]),
    [chat, setChat] = useState<Chat | null>(null),
    [historyWindow, setHistoryWindow] =
      useState<HistoryWindow>(emptyHistoryWindow),
    [query, setQuery] = useState(""),
    [answer, setAnswer] = useState(""),
    [layers, setLayers] = useState<LayerData[]>([]),
    [tables, setTables] = useState<TableData[]>([]),
    [events, setEvents] = useState<Array<{ time: string; event: StreamEvent }>>(
      [],
    ),
    [status, setStatus] = useState("Готов к работе"),
    [statusEntries, setStatusEntries] = useState<StatusEntry[]>([]),
    [pendingQuestion, setPendingQuestion] = useState(""),
    [restoreState, setRestoreState] = useState<Record<string, string>>({}),
    [undoLayers, setUndoLayers] = useState<LayerData[] | null>(null),
    [busy, setBusy] = useState(false),
    [rightTab, setRightTab] = useState<"map" | "data" | "process">("map"),
    [historyOpen, setHistoryOpen] = useState(false),
    [agentMenuOpen, setAgentMenuOpen] = useState(false),
    [resultOpen, setResultOpen] = useState(false),
    [models, setModels] = useState<string[]>([]),
    [settingsOpen, setSettingsOpen] = useState(false),
    [loginOpen, setLoginOpen] = useState(false),
    [authApi, setAuthApi] = useState(false),
    [systemPassword, setSystemPassword] = useState(""),
    [systemConfig, setSystemConfig] = useState<Record<string, string> | null>(
      null,
    );
  const abort = useRef<AbortController | null>(null),
    kc = useRef<Keycloak | null>(null),
    // Credentials for the /auth/token proxy login: kept in memory only (never
    // persisted) so the short-lived token can be re-requested before expiry.
    helperCreds = useRef<{ username: string; password: string } | null>(null),
    reloginTimer = useRef<number | null>(null),
    resultAutoOpened = useRef(false),
    stepBase = useRef(""),
    chatIdRef = useRef<string | undefined>(undefined),
    activeRequestIdRef = useRef<string | undefined>(undefined),
    activeExchangeRef = useRef<ActiveExchange | null>(null),
    chatWindowsRef = useRef<Map<string, CachedChatWindow>>(new Map()),
    messagesScroller = useRef<HTMLDivElement>(null),
    messagesEnd = useRef<HTMLDivElement>(null),
    undoTimer = useRef<number | null>(null);
  const agent = AGENTS.find((a) => a.id === agentId)!;
  useGSAP(
    () => {
      gsap.fromTo(
        ".sidebar",
        { y: -24, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.7, ease: "power3.out" },
      );
      gsap.fromTo(
        ".conversation, .inspector",
        { y: 22, opacity: 0, scale: 0.985 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.7,
          stagger: 0.08,
          ease: "power3.out",
        },
      );
      gsap.fromTo(
        ".terrain-orbit",
        { scale: 0.8, opacity: 0.2 },
        { scale: 1, opacity: 1, duration: 1.1, ease: "power2.out" },
      );
      gsap.fromTo(
        ".prompt-grid button",
        { y: 38, opacity: 0, scale: 0.96 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.7,
          stagger: 0.09,
          ease: "power3.out",
        },
      );
    },
    { scope: appRoot },
  );
  useEffect(() => {
    document.documentElement.dataset.theme = settings.theme;
    localStorage.setItem("gmart-ui", JSON.stringify(settings));
  }, [settings]);
  useEffect(() => {
    authAvailable(settings).then(setAuthApi);
    return () => {
      if (reloginTimer.current) window.clearTimeout(reloginTimer.current);
    };
  }, []);
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
  }, [token]);
  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: busy ? "smooth" : "auto" });
  }, [answer, busy, pendingQuestion, statusEntries]);
  useEffect(() => {
    if (!chat) return;
    rememberChatWindow(chat, historyWindow);
  }, [chat, historyWindow]);
  // Unauthenticated on purpose: /llm/available_models carries no token requirement, and the
  // list must be known before the first request so a stale saved model cannot be sent.
  useEffect(() => {
    getModels(settings, token)
      .then((list) => {
        setModels(list);
        setSettings((prev) =>
          list.length && !list.includes(prev.model)
            ? { ...prev, model: list[0] }
            : prev,
        );
      })
      .catch(() => {});
  }, [token, settings.agentsUrl]);
  useEffect(() => setQuery(agent.examples[0]), [agentId]);
  async function loadChats() {
    try {
      setChats((await getChats(settings, token)).items);
    } catch (e) {
      setStatus(err(e));
    }
  }
  function rememberChatWindow(value: Chat, history: HistoryWindow) {
    const windows = chatWindowsRef.current;
    windows.delete(value.chat_id);
    windows.set(value.chat_id, {
      chat: value,
      history: { ...history, loading: false },
    });
    while (windows.size > MAX_CACHED_CHAT_WINDOWS) {
      const oldestId = windows.keys().next().value;
      if (!oldestId) break;
      windows.delete(oldestId);
    }
  }
  function scrollChatToBottom() {
    window.requestAnimationFrame(() => {
      messagesEnd.current?.scrollIntoView({ behavior: "auto" });
    });
  }
  async function openChat(id: string) {
    if (busy || chat?.chat_id === id) return;
    if (chat) rememberChatWindow(chat, historyWindow);
    const cached = chatWindowsRef.current.get(id);
    if (cached) {
      setChat(cached.chat);
      setHistoryWindow(cached.history);
      chatIdRef.current = cached.chat.chat_id;
      activeExchangeRef.current = null;
      if (cached.chat.scenario_id != null)
        setScenario(String(cached.chat.scenario_id));
      if (cached.chat.project_id != null)
        setProject(String(cached.chat.project_id));
      const cachedAgent = String(cached.chat.metadata?.agent_id || "");
      if (AGENTS.some((item) => item.id === cachedAgent))
        setAgentId(cachedAgent as AgentId);
      setAnswer("");
      setPendingQuestion("");
      setStatusEntries([]);
      setLayers([]);
      setTables(extractStoredTables(cached.chat.messages));
      scrollChatToBottom();
      return;
    }
    try {
      const stored = await getChat(settings, token, id, {
        limit: CHAT_PAGE_SIZE,
      });
      setChat(stored);
      setHistoryWindow({
        hasMore: Boolean(stored.has_more),
        nextBeforeSeq: stored.next_before_seq ?? null,
        loading: false,
      });
      chatIdRef.current = stored.chat_id;
      activeExchangeRef.current = null;
      if (stored.scenario_id != null) setScenario(String(stored.scenario_id));
      if (stored.project_id != null) setProject(String(stored.project_id));
      const storedAgent = String(stored.metadata?.agent_id || "");
      if (AGENTS.some((item) => item.id === storedAgent))
        setAgentId(storedAgent as AgentId);
      setAnswer("");
      setPendingQuestion("");
      setStatusEntries([]);
      setLayers([]);
      setTables(extractStoredTables(stored.messages));
      scrollChatToBottom();
    } catch (e) {
      setStatus(err(e));
    }
  }
  async function loadOlderMessages() {
    const current = chat;
    const beforeSeq = historyWindow.nextBeforeSeq;
    if (
      !current ||
      busy ||
      !historyWindow.hasMore ||
      !beforeSeq ||
      historyWindow.loading
    )
      return;

    const scroller = messagesScroller.current;
    const previousHeight = scroller?.scrollHeight || 0;
    const previousTop = scroller?.scrollTop || 0;
    setHistoryWindow((value) => ({ ...value, loading: true }));
    try {
      const page = await getChat(settings, token, current.chat_id, {
        limit: CHAT_PAGE_SIZE,
        beforeSeq,
      });
      const merged = mergeMessageWindow(page.messages, current.messages);
      setChat((value) => {
        if (!value || value.chat_id !== current.chat_id) return value;
        return {
          ...value,
          messages: mergeMessageWindow(page.messages, value.messages).messages,
        };
      });
      setTables(extractStoredTables(merged.messages));
      setHistoryWindow({
        hasMore: Boolean(page.has_more),
        nextBeforeSeq: page.next_before_seq ?? null,
        loading: false,
      });
      window.requestAnimationFrame(() => {
        if (!scroller) return;
        scroller.scrollTop =
          scroller.scrollHeight - previousHeight + previousTop;
      });
    } catch (error) {
      setHistoryWindow((value) => ({ ...value, loading: false }));
      setStatus(err(error));
    }
  }
  async function removeChat(id: string) {
    if (!confirm("Удалить этот диалог?")) return;
    await deleteChat(settings, token, id);
    if (chat?.chat_id === id) {
      setChat(null);
      chatIdRef.current = undefined;
      setHistoryWindow(emptyHistoryWindow);
    }
    chatWindowsRef.current.delete(id);
    loadChats();
  }
  function login() {
    if (authApi) setLoginOpen(true);
    else if (kc.current) kc.current.login();
    else {
      const url = new URL(settings.authHelperUrl);
      url.searchParams.set("returnUrl", location.origin + location.pathname);
      location.href = url.toString();
    }
  }
  async function helperLogin(username: string, password: string) {
    const data = await authLogin(settings, username, password);
    helperCreds.current = { username, password };
    setToken(data.access_token);
    setAuth("ready");
    scheduleRelogin(data.expires_in);
    return data.access_token;
  }
  function scheduleRelogin(expiresIn?: number) {
    if (reloginTimer.current) window.clearTimeout(reloginTimer.current);
    if (!expiresIn || expiresIn <= 60) return;
    reloginTimer.current = window.setTimeout(
      async () => {
        const creds = helperCreds.current;
        if (!creds) return;
        try {
          await helperLogin(creds.username, creds.password);
        } catch {
          helperCreds.current = null;
          setToken("");
          setAuth("anonymous");
        }
      },
      (expiresIn - 30) * 1000,
    );
  }
  async function freshToken() {
    if (helperCreds.current) {
      const { username, password } = helperCreds.current;
      return helperLogin(username, password);
    }
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
  function updateStatus(text: string, state: StatusEntry["state"] = "active") {
    if (!text) return;
    setStatus(text);
    setStatusEntries((previous) => {
      const completed = previous.map((entry) =>
        entry.state === "active" ? { ...entry, state: "done" as const } : entry,
      );
      const last = completed.at(-1);
      if (last?.text === text)
        return [...completed.slice(0, -1), { ...last, state }];
      return [
        ...completed,
        {
          id: crypto.randomUUID(),
          text,
          time: new Date().toLocaleTimeString("ru", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          }),
          state,
        },
      ].slice(-30);
    });
  }
  function updateSseAnswer(
    update: string | ((current: string) => string),
  ): string {
    const exchange = activeExchangeRef.current;
    const current = exchange?.answer || "";
    const next = typeof update === "function" ? update(current) : update;
    if (exchange) exchange.answer = next;
    setAnswer(next);
    return next;
  }
  function finalizeActiveExchange(fallbackAnswer?: string) {
    const exchange = activeExchangeRef.current;
    if (!exchange || exchange.finalized) return;
    if (!exchange.answer.trim() && fallbackAnswer)
      exchange.answer = fallbackAnswer;
    exchange.finalized = true;

    const id = chatIdRef.current;
    if (!id) return;
    setChat((current) => {
      const base: Chat = current || {
        chat_id: id,
        title: exchange.question || null,
        scenario_id: scenario || null,
        project_id: project || null,
        updated_at: new Date().toISOString(),
        metadata: { agent_id: agentId },
        messages: [],
      };
      const result = appendSseExchange(id, base.messages, exchange);
      if (
        result.removed.some(
          (message) => !message.message_id.startsWith("browser:"),
        )
      ) {
        const nextBeforeSeq = oldestServerSequence(result.messages);
        if (nextBeforeSeq != null)
          setHistoryWindow((value) => ({
            ...value,
            hasMore: true,
            nextBeforeSeq,
          }));
      }
      return {
        ...base,
        updated_at: new Date().toISOString(),
        messages: result.messages,
      };
    });
    setPendingQuestion("");
    setAnswer("");
    void loadChats();
  }
  function route(event: StreamEvent, nested = false) {
    if (event.type === "pipeline_started") {
      activeRequestIdRef.current = event.content?.request_id;
      updateStatus("Агент начал работу");
    }
    if (event.type === "status")
      updateStatus(event.content?.text || labelStatus(event.content?.status));
    if (event.type === "chunk" || event.type === "Text") {
      const text =
        typeof event.content === "string"
          ? event.content
          : event.content?.text || "";
      updateSseAnswer((current) =>
        event.content?.iteration && event.content.iteration > 1
          ? stepBase.current + text
          : current + text,
      );
      if (event.content?.done && !nested) {
        activeRequestIdRef.current = undefined;
        setBusy(false);
        updateStatus("Ответ готов", "done");
        finalizeActiveExchange();
      }
    }
    if (event.type === "plan_created") {
      const steps = event.content?.steps || [];
      updateStatus(`План составлен: ${steps.length} ${pluralize(steps.length, "шаг", "шага", "шагов")}`);
    }
    if (event.type === "plan_revision_created") {
      const revision = event.content?.revision || 1;
      const steps = event.content?.steps || [];
      updateStatus(`План №${revision}: ${steps.length} ${pluralize(steps.length, "шаг", "шага", "шагов")}`);
    }
    if (event.type === "plan") {
      const steps = event.content?.steps || [];
      updateStatus(`План готов: шагов — ${steps.length}`);
      if (steps.length)
        updateSseAnswer(
          (current) =>
            current +
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
      if (event.content?.step_id) {
        updateStatus(event.content?.purpose || `Выполняю ${event.content.step_id}`);
        return;
      }
      const step = event.content?.step,
        agent = labelAgent(event.content?.agent);
      updateStatus(`Шаг ${step}: ${agent}`);
      updateSseAnswer(
        (current) =>
          (stepBase.current = `${current}---\n\n**Шаг ${step} · ${agent}**\n\n`),
      );
    }
    if (event.type === "step_completed")
      updateStatus(`Шаг ${event.content?.step_id || "плана"} завершён`);
    if (event.type === "mapping_started")
      updateStatus(event.content?.text || "Получаю актуальные справочники…");
    if (event.type === "mapping_completed")
      updateStatus(event.content?.text || "Актуальные справочники получены");
    if (event.type === "artifact_created")
      updateStatus(`Набор данных подготовлен: ${event.content?.rows ?? 0} строк`);
    if (event.type === "validation_started")
      updateStatus(event.content?.text || "Проверяю полноту результата…");
    if (event.type === "validation_completed")
      updateStatus("Результат проверен");
    if (event.type === "replanning")
      updateStatus(`Уточняю план: редакция ${event.content?.revision || ""}`);
    if (event.type === "clarification_required") {
      updateStatus("Нужно уточнение", "warning");
    }
    if (event.type === "pipeline_failed")
      updateStatus(
        Array.isArray(event.content?.reasons)
          ? event.content.reasons.join("; ")
          : "Получен частичный результат",
        "warning",
      );
    if (event.type === "step_event" && event.content?.event)
      route(event.content.event, true);
    if (event.type === "step_finished") {
      if (event.content?.status === "failed")
        updateSseAnswer(
          (current) =>
            current +
            `\n\n> Шаг ${event.content.step} не выполнен: ${event.content.summary || "ошибка агента"}\n\n`,
        );
      updateStatus(`Шаг ${event.content?.step} завершён`);
    }
    if (event.type === "clarification") {
      updateSseAnswer((current) => current + (event.content?.question || ""));
      updateStatus("Нужно уточнение", "warning");
    }
    if (event.type === "orchestrator_final") {
      updateStatus("Ответ готов", "done");
      finalizeActiveExchange();
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
      if (!resultAutoOpened.current) {
        resultAutoOpened.current = true;
        setResultOpen(true);
      }
    }
    if (event.type === "table") {
      if (activeExchangeRef.current)
        activeExchangeRef.current.tables.push(event.content);
      setTables((v) => [...v, event.content]);
      setRightTab("data");
      if (!resultAutoOpened.current) {
        resultAutoOpened.current = true;
        setResultOpen(true);
      }
    }
    if (event.type === "warning" || event.type === "error")
      updateStatus(event.content?.message || "Ошибка выполнения", "warning");
    if (event.type === "service_event" && event.content?.event?.chat_id) {
      const id = String(event.content.event.chat_id);
      chatIdRef.current = id;
      setChat((value) =>
        value
          ? { ...value, chat_id: id }
          : {
              chat_id: id,
              title: activeExchangeRef.current?.question || null,
              scenario_id: scenario || null,
              project_id: project || null,
              updated_at: new Date().toISOString(),
              metadata: { agent_id: agentId },
              messages: [],
            },
      );
    }
    if (event.type === "token_expired")
      void refreshPipeline(event.content?.request_id);
  }
  async function restoreLayers(message: Message, part: MessagePart) {
    const key = `${message.message_id}:${part.part_seq}`;
    setRestoreState((value) => ({ ...value, [key]: "Восстанавливаю…" }));
    try {
      const calls = Array.isArray(part.payload?.tool_calls)
        ? part.payload.tool_calls
        : Array.isArray(part.payload?.calls)
          ? part.payload.calls
          : [];
      const targetCall = calls.at(-1);
      const response = await replayToolCall(
        settings,
        token,
        message.message_id,
        part.part_seq,
        Number(targetCall?.step || calls.length || 1),
        scenario,
        project,
      );
      const collections = findFeatureCollections(response);
      if (!collections.length)
        throw new Error("В сохранённом результате нет геометрий");
      setLayers((current) => [
        ...current,
        ...collections.map((geojson, index) => ({
          id: crypto.randomUUID(),
          name: `Восстановленный слой ${current.length + index + 1}`,
          color: colors[(current.length + index) % colors.length],
          visible: true,
          geojson,
          count: geojson.features.length,
        })),
      ]);
      setRightTab("map");
      setResultOpen(true);
      setRestoreState((value) => ({
        ...value,
        [key]: `Восстановлено: ${collections.length}`,
      }));
    } catch (error) {
      setRestoreState((value) => ({ ...value, [key]: err(error) }));
    }
  }
  function clearLayers() {
    if (!layers.length) return;
    setUndoLayers(layers);
    setLayers([]);
    if (undoTimer.current) window.clearTimeout(undoTimer.current);
    undoTimer.current = window.setTimeout(() => setUndoLayers(null), 8000);
  }
  function restoreClearedLayers() {
    if (!undoLayers) return;
    setLayers(undoLayers);
    setUndoLayers(null);
  }
  async function refreshPipeline(id: string) {
    updateStatus("Обновляю доступ к данным…");
    const t = await freshToken();
    let lastError: unknown;
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        await request(settings.agentsUrl, `/pipelines/${id}/token`, t, {
          method: "POST",
          body: JSON.stringify({ token: t }),
        });
        updateStatus("Доступ обновлён, продолжаю запрос…");
        return;
      } catch (error) {
        lastError = error;
        await new Promise((resolve) => window.setTimeout(resolve, 300));
      }
    }
    updateStatus(`Не удалось обновить доступ: ${err(lastError)}`, "warning");
  }
  async function cancelActivePipeline() {
    const requestId = activeRequestIdRef.current;
    try {
      if (requestId) {
        const currentToken = await freshToken();
        await request(
          settings.agentsUrl,
          `/pipelines/${requestId}/cancel`,
          currentToken,
          { method: "POST" },
        );
      }
      updateStatus("Запрос остановлен", "warning");
    } catch (error) {
      updateStatus(`Не удалось остановить запрос: ${err(error)}`, "warning");
    } finally {
      activeRequestIdRef.current = undefined;
      abort.current?.abort();
      setBusy(false);
    }
  }
  async function submit() {
    if (!token) {
      login();
      return;
    }
    if (!query.trim() || busy) return;
    const submittedQuery = query.trim();
    activeExchangeRef.current = {
      question: submittedQuery,
      answer: "",
      tables: [],
      finalized: false,
    };
    setBusy(true);
    setPendingQuestion(submittedQuery);
    setQuery("");
    setAnswer("");
    setTables([]);
    setEvents([]);
    setStatusEntries([]);
    stepBase.current = "";
    updateStatus("Подключение к агенту…");
    const url = new URL(agent.path, settings.agentsUrl);
    url.searchParams.set("request", submittedQuery);
    // Omitted when unknown: the agents then use the provider's default.
    if (settings.model) url.searchParams.set("model", settings.model);
    url.searchParams.set("temperature", String(settings.temperature));
    if (scenario) url.searchParams.set("scenario_id", scenario);
    if (chatIdRef.current) url.searchParams.set("chat_id", chatIdRef.current);
    abort.current = new AbortController();
    try {
      await readSse(url, await freshToken(), abort.current.signal, handle);
      setBusy(false);
      finalizeActiveExchange(
        "Поток завершился без отдельного финального сообщения.",
      );
    } catch (e) {
      const aborted = (e as Error).name === "AbortError";
      const message = aborted ? "Запрос остановлен пользователем." : err(e);
      updateStatus(message, "warning");
      finalizeActiveExchange(message);
      setBusy(false);
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
  const history = useMemo(
    () => normalizeMessages(chat?.messages || []),
    [chat?.messages],
  );
  return (
    <div className="app-shell" ref={appRoot}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Command weight="bold" />
          </div>
          <div>
            gMART<small>пространственный интеллект</small>
          </div>
        </div>
        <nav>
          <p className="nav-label">Навигация</p>
          <button
            className={mode === "workspace" ? "active" : ""}
            onClick={() => setMode("workspace")}
          >
            <span>
              <Sparkle weight="duotone" />
            </span>
            Работа
          </button>
          <p className="nav-label">Команда агентов</p>
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
              <span>
                <AgentGlyph id={a.id} />
              </span>
              <div>
                {a.label}
                <small>{a.caption}</small>
              </div>
            </button>
          ))}
          <button
            className={mode === "mcp" ? "active" : ""}
            onClick={() => setMode("mcp")}
          >
            <span>
              <TerminalWindow weight="duotone" />
            </span>
            MCP-консоль
          </button>
          <button
            className={mode === "admin" ? "active" : ""}
            onClick={() => setMode("admin")}
          >
            <span>
              <GearSix weight="duotone" />
            </span>
            Система
          </button>
        </nav>
        <div className="side-bottom">
          <button
            className="history-trigger"
            onClick={() => setHistoryOpen(true)}
            aria-label="История сообщений"
          >
            <ClockCounterClockwise /> <span>История сообщений</span>
          </button>
          <button onClick={() => setSettingsOpen(true)}>
            <SlidersHorizontal /> <span>Настройки</span>
          </button>
          <button
            onClick={() =>
              setSettings((s) => ({
                ...s,
                theme: s.theme === "dark" ? "light" : "dark",
                basemap: s.theme === "dark" ? "cartoLight" : "cartoDark",
              }))
            }
          >
            {settings.theme === "dark" ? <Sun /> : <Moon />}
          </button>
        </div>
      </aside>
      <main className="main-stage">
        {mode === "workspace" ? (
          <>
            <header className="workspace-header">
              <div className="agent-picker-wrap">
                <button
                  className="agent-picker"
                  onClick={() => setAgentMenuOpen((value) => !value)}
                >
                  <span className="agent-picker-icon">
                    <AgentGlyph id={agent.id} />
                  </span>
                  <span>
                    <small>Активный агент</small>
                    <strong>{agent.label}</strong>
                  </span>
                  <CaretDown />
                </button>
                {agentMenuOpen && (
                  <div className="agent-menu">
                    {AGENTS.map((item) => (
                      <button
                        className={item.id === agent.id ? "active" : ""}
                        key={item.id}
                        onClick={() => {
                          setAgentId(item.id);
                          setAgentMenuOpen(false);
                          resultAutoOpened.current = false;
                        }}
                      >
                        <AgentGlyph id={item.id} />
                        <span>
                          <strong>{item.label}</strong>
                          <small>{item.caption}</small>
                        </span>
                      </button>
                    ))}
                  </div>
                )}
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
                <button
                  className="result-toggle"
                  onClick={() => setResultOpen((value) => !value)}
                >
                  <MapTrifold /> Результат
                  {(layers.length || tables.length) > 0 && (
                    <b>{layers.length + tables.length}</b>
                  )}
                </button>
                {auth !== "ready" && (
                  <button className="primary login-button" onClick={login}>
                    Войти
                  </button>
                )}
              </div>
            </header>
            <div className={`work-grid ${resultOpen ? "result-open" : ""}`}>
              <section className="conversation">
                <div
                  className="messages"
                  ref={messagesScroller}
                >
                  {!history.length && !answer && !pendingQuestion ? (
                    <Welcome agent={agent} onExample={setQuery} />
                  ) : (
                    <>
                      {historyWindow.hasMore && (
                        <button
                          className="history-load-more"
                          onClick={() => void loadOlderMessages()}
                          disabled={historyWindow.loading}
                        >
                          <ClockCounterClockwise />
                          {historyWindow.loading
                            ? "Загружаю историю…"
                            : "Показать предыдущие сообщения"}
                        </button>
                      )}
                      {history.map((m) => (
                        <MessageView
                          key={m.message_id}
                          message={m}
                          restore={restoreLayers}
                          restoreState={restoreState}
                          openTables={() => {
                            setRightTab("data");
                            setResultOpen(true);
                          }}
                        />
                      ))}
                      {pendingQuestion && (
                        <TransientMessage
                          role="user"
                          text={pendingQuestion}
                          detail="Запрос отправлен"
                          pending
                        />
                      )}
                      {!!statusEntries.length && (
                        <LiveStatus
                          entries={statusEntries}
                          current={status}
                          busy={busy}
                        />
                      )}
                      {answer && (
                        <TransientMessage
                          role="assistant"
                          text={answer}
                          detail={busy ? "Ответ формируется" : "Ответ получен"}
                        />
                      )}
                      {(layers.length > 0 ||
                        tables.length > 0 ||
                        events.length > 0) && (
                        <ArtifactSummary
                          layers={layers}
                          tables={tables}
                          events={events}
                          open={(tab) => {
                            setRightTab(tab);
                            setResultOpen(true);
                          }}
                        />
                      )}
                      <div ref={messagesEnd} />
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
                        busy ? void cancelActivePipeline() : void submit()
                      }
                      className="send"
                    >
                      {busy ? <X weight="bold" /> : <ArrowUp weight="bold" />}
                    </button>
                  </div>
                </div>
              </section>
              <section className={`inspector ${resultOpen ? "open" : ""}`}>
                <div className="tabs">
                  <button
                    className={rightTab === "map" ? "active" : ""}
                    onClick={() => setRightTab("map")}
                  >
                    <MapTrifold /> Карта <b>{layers.length}</b>
                  </button>
                  <button
                    className={rightTab === "data" ? "active" : ""}
                    onClick={() => setRightTab("data")}
                  >
                    <Database /> Данные <b>{tables.length}</b>
                  </button>
                  <button
                    className={rightTab === "process" ? "active" : ""}
                    onClick={() => setRightTab("process")}
                  >
                    <ChartDonut /> Процесс
                  </button>
                  <button
                    className="close-result"
                    onClick={() => setResultOpen(false)}
                    aria-label="Закрыть результат"
                  >
                    <X />
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
                    onRemove={(id) =>
                      setLayers((value) =>
                        value.filter((layer) => layer.id !== id),
                      )
                    }
                    onClear={clearLayers}
                  />
                )}{" "}
                {rightTab === "data" && <Tables tables={tables} />}{" "}
                {rightTab === "process" && <Process events={events} />}
              </section>
            </div>
          </>
        ) : mode === "mcp" ? (
          <McpConsole settings={settings} token={token} setToken={setToken} />
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
      <ChatHistoryDrawer
        open={historyOpen}
        chats={chats}
        activeId={chat?.chat_id}
        close={() => setHistoryOpen(false)}
        create={() => {
          if (chat) rememberChatWindow(chat, historyWindow);
          setChat(null);
          chatIdRef.current = undefined;
          activeExchangeRef.current = null;
          setHistoryWindow(emptyHistoryWindow);
          setAnswer("");
          setPendingQuestion("");
          setStatusEntries([]);
          setLayers([]);
          setTables([]);
          setEvents([]);
          setHistoryOpen(false);
        }}
        openChat={(id) => {
          openChat(id);
          setHistoryOpen(false);
        }}
        removeChat={removeChat}
      />
      {settingsOpen && (
        <SettingsModal
          settings={settings}
          setSettings={setSettings}
          close={() => setSettingsOpen(false)}
          models={models}
        />
      )}
      {loginOpen && (
        <LoginModal login={helperLogin} close={() => setLoginOpen(false)} />
      )}
      {undoLayers && (
        <div className="undo-toast" role="status">
          <span>Слои удалены с карты</span>
          <button onClick={restoreClearedLayers}>
            <ArrowCounterClockwise /> Отменить
          </button>
        </div>
      )}
    </div>
  );
}
function ArtifactSummary({
  layers,
  tables,
  events,
  open,
}: {
  layers: LayerData[];
  tables: TableData[];
  events: Array<{ time: string; event: StreamEvent }>;
  open: (tab: "map" | "data" | "process") => void;
}) {
  return (
    <section className="artifact-summary">
      <div>
        <span className="artifact-icon">
          <MapTrifold weight="duotone" />
        </span>
        <div>
          <strong>Результаты собраны</strong>
          <p>Карта, данные и ход анализа доступны в контексте ответа.</p>
        </div>
      </div>
      <div className="artifact-actions">
        <button onClick={() => open("map")}>
          <MapTrifold /> Слои <b>{layers.length}</b>
        </button>
        <button onClick={() => open("data")}>
          <Database /> Таблицы <b>{tables.length}</b>
        </button>
        <button onClick={() => open("process")}>
          <ChartDonut /> Процесс <b>{events.length}</b>
        </button>
      </div>
    </section>
  );
}

function LiveStatus({
  entries,
  current,
  busy,
}: {
  entries: StatusEntry[];
  current: string;
  busy: boolean;
}) {
  const root = useRef<HTMLElement>(null);
  useGSAP(
    () => {
      gsap.fromTo(
        ".status-step",
        { y: 12, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.35, stagger: 0.04, ease: "power2.out" },
      );
    },
    { scope: root, dependencies: [busy, entries.length] },
  );
  return (
    <section
      className={`live-status ${busy ? "active" : "complete"}`}
      ref={root}
    >
      <div className="live-status-head">
        <span className="status-orb" />
        <div>
          <small>{busy ? "Агент работает" : "Выполнение завершено"}</small>
          <strong>{current}</strong>
        </div>
        <time>{entries.at(-1)?.time}</time>
      </div>
      <details open={busy}>
        <summary>Ход выполнения · {entries.length}</summary>
        <ol>
          {entries.map((entry) => (
            <li className={`status-step ${entry.state}`} key={entry.id}>
              <i />
              <span>{entry.text}</span>
              <time>{entry.time}</time>
            </li>
          ))}
        </ol>
      </details>
    </section>
  );
}

function ChatHistoryDrawer({
  open,
  chats,
  activeId,
  close,
  create,
  openChat,
  removeChat,
}: {
  open: boolean;
  chats: ChatSummary[];
  activeId?: string;
  close: () => void;
  create: () => void;
  openChat: (id: string) => void;
  removeChat: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [scenarioFilter, setScenarioFilter] = useState("");
  const [projectFilter, setProjectFilter] = useState("");
  const [agentFilter, setAgentFilter] = useState("");
  const filtered = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase("ru");
    return chats.filter((item) => {
      const agent = String(item.metadata?.agent_id || "");
      return (
        (!needle ||
          (item.title || "Новый диалог")
            .toLocaleLowerCase("ru")
            .includes(needle)) &&
        (!scenarioFilter ||
          String(item.scenario_id || "").includes(scenarioFilter)) &&
        (!projectFilter ||
          String(item.project_id || "").includes(projectFilter)) &&
        (!agentFilter || agent === agentFilter)
      );
    });
  }, [agentFilter, chats, projectFilter, scenarioFilter, search]);
  if (!open) return null;

  return (
    <>
      <button
        className={`drawer-backdrop ${open ? "open" : ""}`}
        onClick={close}
        aria-label="Закрыть историю"
      />
      <aside
        className={`history-drawer ${open ? "open" : ""}`}
        aria-hidden={!open}
      >
        <div className="drawer-head">
          <div>
            <ClockCounterClockwise />
            <h2>Диалоги</h2>
          </div>
          <button onClick={close} aria-label="Закрыть">
            <X />
          </button>
        </div>
        <button className="new-chat" onClick={create}>
          <Plus /> Новый диалог
        </button>
        <div className="drawer-search">
          <List />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Найти диалог"
          />
        </div>
        <div className="history-filters">
          <input
            value={scenarioFilter}
            onChange={(event) => setScenarioFilter(event.target.value)}
            placeholder="Сценарий"
          />
          <input
            value={projectFilter}
            onChange={(event) => setProjectFilter(event.target.value)}
            placeholder="Проект"
          />
          <select
            value={agentFilter}
            onChange={(event) => setAgentFilter(event.target.value)}
          >
            <option value="">Все агенты</option>
            {AGENTS.map((item) => (
              <option value={item.id} key={item.id}>
                {item.label}
              </option>
            ))}
          </select>
        </div>
        {!!chats.length && (
          <div className="recent-chats" aria-label="Недавние диалоги">
            {chats.slice(0, 6).map((item) => (
              <button key={item.chat_id} onClick={() => openChat(item.chat_id)}>
                {item.title || "Новый диалог"}
              </button>
            ))}
          </div>
        )}
        <div className="history-count">Найдено: {filtered.length}</div>
        <div className="drawer-chat-list">
          {filtered.map((item) => (
            <div
              className={`drawer-chat ${activeId === item.chat_id ? "active" : ""}`}
              key={item.chat_id}
            >
              <button onClick={() => openChat(item.chat_id)}>
                <strong>{item.title || "Новый диалог"}</strong>
                <small>
                  {labelAgent(String(item.metadata?.agent_id || ""))} · сценарий{" "}
                  {item.scenario_id || "—"} ·{" "}
                  {new Date(item.updated_at).toLocaleDateString("ru")}
                </small>
              </button>
              <button
                className="delete"
                onClick={() => removeChat(item.chat_id)}
                aria-label="Удалить диалог"
              >
                <Trash />
              </button>
            </div>
          ))}
          {!filtered.length && (
            <div className="empty">Подходящих диалогов нет</div>
          )}
        </div>
      </aside>
    </>
  );
}

function AgentGlyph({ id }: { id: AgentId }) {
  const props = { weight: "duotone" as const };
  if (id === "orchestrator") return <CirclesFour {...props} />;
  if (id === "restrictions") return <ShieldCheck {...props} />;
  if (id === "compliance") return <CheckCircle {...props} />;
  if (id === "provision") return <ChartDonut {...props} />;
  if (id === "scenario_data") return <Buildings {...props} />;
  if (id === "documents") return <FileText {...props} />;
  if (id === "norms") return <SquaresFour {...props} />;
  return <Sparkle {...props} />;
}
function LoginModal({
  login,
  close,
}: {
  login: (username: string, password: string) => Promise<string>;
  close: () => void;
}) {
  const [username, setUsername] = useState(""),
    [password, setPassword] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  async function submit() {
    setBusy(true);
    setError("");
    try {
      await login(username, password);
      close();
    } catch (e) {
      setError(err(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div
      className="modal"
      onMouseDown={(e) => e.target === e.currentTarget && close()}
    >
      <div className="modal-card">
        <div className="panel-head">
          <div>
            <span className="context-title">Безопасный вход</span>
            <h2>Вход в IDU</h2>
          </div>
          <button onClick={close}>×</button>
        </div>
        <div className="form-grid">
          <label>
            Логин
            <input
              value={username}
              autoFocus
              autoComplete="username"
              onChange={(e) => setUsername(e.target.value)}
            />
          </label>
          <label>
            Пароль
            <input
              type="password"
              value={password}
              autoComplete="current-password"
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !busy && submit()}
            />
          </label>
        </div>
        {error && <small className="login-error">{error}</small>}
        <div className="modal-actions">
          <button onClick={close}>Отмена</button>
          <button
            className="primary"
            disabled={busy || !username || !password}
            onClick={submit}
          >
            {busy ? "Вход…" : "Войти"}
          </button>
        </div>
      </div>
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
      <div className="welcome-copy">
        <div className="agent-sign">
          <AgentGlyph id={agent.id} />
        </div>
        <p className="welcome-kicker">{agent.label} готов к работе</p>
        <h2>
          Исследуйте город <span className="title-map" aria-hidden="true" />
          через данные
        </h2>
        <p className="welcome-lead">
          Опишите задачу своими словами. Агент соберёт инструменты, покажет ход
          анализа и представит результат на карте.
        </p>
      </div>
      <div className="terrain-orbit" aria-hidden="true">
        <div className="terrain-ring ring-one" />
        <div className="terrain-ring ring-two" />
        <MapTrifold weight="thin" />
      </div>
      <div className="prompt-grid">
        {agent.examples.map((x, index) => (
          <button key={x} onClick={() => onExample(x)}>
            <span>{x}</span>
            <ArrowUp className="prompt-arrow" weight="bold" />
            <small>
              {index === 0 ? "Начать с примера" : "Попробовать запрос"}
            </small>
          </button>
        ))}
      </div>
    </div>
  );
}
function MessageView({
  message,
  restore,
  restoreState,
  openTables,
}: {
  message: Message;
  restore: (message: Message, part: MessagePart) => void;
  restoreState: Record<string, string>;
  openTables: () => void;
}) {
  const isUser = message.role.toLowerCase() === "user";
  const blocks = buildMessageBlocks(message.parts);
  const formattedTime = formatMessageTime(message.created_at);
  return (
    <article className={`message ${isUser ? "user" : "assistant"}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? "В" : "g"}
      </div>
      <div className="message-card">
        <header className="message-meta">
          <strong>{isUser ? "Вы" : "gMART"}</strong>
          {formattedTime && (
            <time dateTime={message.created_at}>{formattedTime}</time>
          )}
        </header>
        <div className="message-content">
          {blocks.map((block) => {
            if (block.kind === "markdown")
              return (
                <MarkdownContent key={block.key}>{block.text}</MarkdownContent>
              );

            const p = block.part;
            return p.kind === "table" ? (
              <StoredTablePart
                key={block.key}
                table={p.payload as TableData}
                open={openTables}
              />
            ) : p.kind === "tool_call" ? (
              <div className="stored-tool-call" key={block.key}>
                <div>
                  <MapTrifold />
                  <span>
                    <strong>Сохранённый результат инструментов</strong>
                    <small>{p.mcp_source || "MCP-источник"}</small>
                  </span>
                </div>
                <button onClick={() => restore(message, p)}>
                  <ArrowCounterClockwise /> Восстановить слои
                </button>
                {restoreState[`${message.message_id}:${p.part_seq}`] && (
                  <small>
                    {restoreState[`${message.message_id}:${p.part_seq}`]}
                  </small>
                )}
              </div>
            ) : p.kind === "status" ? (
              <div className="stored-status" key={block.key}>
                {String(p.payload.text || p.payload.status || "Этап выполнен")}
              </div>
            ) : [
                "plan",
                "plan_revision",
                "artifact_ref",
                "validation",
                "failure",
              ].includes(p.kind) ? (
              <details className="stored-status" key={block.key}>
                <summary>{storedPartTitle(p.kind, p.payload)}</summary>
                <pre>{JSON.stringify(p.payload, null, 2)}</pre>
              </details>
            ) : (
              <div className="stored-status" key={block.key}>
                Сохранённая часть: {p.kind}
              </div>
            );
          })}
        </div>
      </div>
    </article>
  );
}

function storedPartTitle(kind: string, payload: Record<string, any>) {
  if (kind === "plan") return "План получения данных";
  if (kind === "plan_revision")
    return `Редакция плана №${payload.revision || 1}`;
  if (kind === "artifact_ref")
    return `Набор данных: ${payload.rows ?? 0} строк`;
  if (kind === "validation") return "Проверка полноты ответа";
  if (kind === "failure") return "Ограничения выполнения";
  return "Служебная часть";
}

function StoredTablePart({
  table,
  open,
}: {
  table: TableData;
  open: () => void;
}) {
  const rows = Array.isArray(table.rows) ? table.rows.length : 0;
  const columns = Array.isArray(table.columns) ? table.columns.length : 0;
  return (
    <div className="stored-table-part">
      <Database weight="duotone" />
      <div>
        <strong>{table.title || table.name || "Сохранённая таблица"}</strong>
        <small>
          {rows} {pluralize(rows, "строка", "строки", "строк")}
          {columns > 0 && ` · ${columns} столбцов`}
        </small>
      </div>
      <button onClick={open}>Открыть в данных</button>
    </div>
  );
}

function TransientMessage({
  role,
  text,
  detail,
  pending = false,
}: {
  role: "user" | "assistant";
  text: string;
  detail: string;
  pending?: boolean;
}) {
  const isUser = role === "user";
  return (
    <article className={`message ${role} ${pending ? "pending-message" : ""}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? "В" : "g"}
      </div>
      <div className="message-card">
        <header className="message-meta">
          <strong>{isUser ? "Вы" : "gMART"}</strong>
          <span>{detail}</span>
        </header>
        <div className="message-content">
          <MarkdownContent>{text}</MarkdownContent>
        </div>
      </div>
    </article>
  );
}

function MarkdownContent({ children }: { children: string }) {
  return (
    <div className="message-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ node: _node, ...props }) => (
            <a {...props} target="_blank" rel="noreferrer" />
          ),
          table: ({ node: _node, ...props }) => (
            <div className="markdown-table-wrap">
              <table {...props} />
            </div>
          ),
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

function formatMessageTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("ru-RU", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function pluralize(
  value: number,
  one: string,
  few: string,
  many: string,
): string {
  const tens = value % 100;
  const units = value % 10;
  if (tens >= 11 && tens <= 19) return many;
  if (units === 1) return one;
  if (units >= 2 && units <= 4) return few;
  return many;
}

function extractStoredTables(messages: Message[]): TableData[] {
  return messages.flatMap((message) =>
    message.parts
      .filter((part) => part.kind === "table")
      .map((part) => part.payload as TableData),
  );
}

function findFeatureCollections(value: unknown): GeoJSON.FeatureCollection[] {
  const found: GeoJSON.FeatureCollection[] = [];
  const seen = new Set<unknown>();
  function walkValue(current: unknown) {
    if (!current || typeof current !== "object" || seen.has(current)) return;
    seen.add(current);
    if (
      (current as { type?: string }).type === "FeatureCollection" &&
      Array.isArray((current as { features?: unknown[] }).features)
    ) {
      found.push(current as GeoJSON.FeatureCollection);
      return;
    }
    if (Array.isArray(current)) current.forEach(walkValue);
    else Object.values(current as Record<string, unknown>).forEach(walkValue);
  }
  walkValue(value);
  return found;
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
          <span className="context-title">Управление контуром</span>
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
            <span className="context-title">Персонализация пространства</span>
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
              {(models.length ? models : [s.model].filter(Boolean)).map((x) => (
                <option key={x}>{x}</option>
              ))}
              {!models.length && !s.model && (
                <option value="">по умолчанию у провайдера</option>
              )}
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
        restrictions: "Ограничения",
        compliance: "Соответствие",
        provision: "Обеспеченность",
        scenario_data: "Данные сценария",
        documents: "Документы",
        norms: "Нормы",
        orchestrator: "Оркестратор",
        llm: "Ассистент",
      } as Record<string, string>
    )[key || ""] ||
    key ||
    "Не определён"
  );
}
function labelStatus(s: string) {
  return (
    (
      {
        tool_discovery: "Загружаю источники данных",
        planning: "Выбираю источник данных",
        tool_execution: "Получаю данные",
        response_analysis: "Считаю результат",
        answer_review: "Проверяю результат",
        answer_retry: "Дополняю данные",
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
