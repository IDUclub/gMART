import { useMemo, useState } from "react";
import { request } from "./api";
import type { Settings } from "./types";

type JsonSchema = {
  type?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  default?: unknown;
  examples?: unknown[];
  example?: unknown;
  enum?: unknown[];
};

type McpTool = {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: JsonSchema;
  };
};

type McpPrompt = {
  name?: string;
  description?: string;
  arguments?: Array<{ name?: string; required?: boolean }>;
};

function exampleValue(schema: JsonSchema): unknown {
  if (schema.default !== undefined) return schema.default;
  if (schema.example !== undefined) return schema.example;
  if (schema.examples?.length) return schema.examples[0];
  if (schema.enum?.length) return schema.enum[0];
  if (schema.type === "object" || schema.properties) {
    return Object.fromEntries(
      Object.entries(schema.properties || {})
        .filter(([key]) => schema.required?.includes(key))
        .map(([key, value]) => [key, exampleValue(value)]),
    );
  }
  if (schema.type === "array") return [];
  if (schema.type === "integer" || schema.type === "number") return 0;
  if (schema.type === "boolean") return false;
  return "";
}

function pretty(value: unknown) {
  return JSON.stringify(value, null, 2);
}

export default function McpConsole({
  settings,
  token,
  setToken,
}: {
  settings: Settings;
  token: string;
  setToken: (token: string) => void;
}) {
  const [tools, setTools] = useState<McpTool[]>([]);
  const [prompts, setPrompts] = useState<McpPrompt[]>([]);
  const [selectedName, setSelectedName] = useState("");
  const [search, setSearch] = useState("");
  const [argumentsJson, setArgumentsJson] = useState("{}");
  const [metaJson, setMetaJson] = useState("{}");
  const [result, setResult] = useState("");
  const [status, setStatus] = useState("Каталог ещё не загружен");
  const [busy, setBusy] = useState(false);
  const [manualToken, setManualToken] = useState(token);

  const selected = tools.find((tool) => tool.function.name === selectedName);
  const visibleTools = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase("ru");
    if (!needle) return tools;
    return tools.filter((tool) =>
      `${tool.function.name} ${tool.function.description || ""}`
        .toLocaleLowerCase("ru")
        .includes(needle),
    );
  }, [search, tools]);

  function selectTool(tool: McpTool) {
    setSelectedName(tool.function.name);
    setArgumentsJson(pretty(exampleValue(tool.function.parameters || {})));
    setResult("");
  }

  async function loadCatalog() {
    setBusy(true);
    setStatus("Подключаюсь к IDU MCP…");
    try {
      const [nextTools, nextPrompts] = await Promise.all([
        request<McpTool[]>(settings.agentsUrl, "/mcp-diagnostics/tools", token),
        request<McpPrompt[]>(
          settings.agentsUrl,
          "/mcp-diagnostics/prompts",
          token,
        ),
      ]);
      setTools(nextTools);
      setPrompts(nextPrompts);
      setStatus(
        `Подключено: ${nextTools.length} инструментов, ${nextPrompts.length} промптов`,
      );
      const current =
        nextTools.find((tool) => tool.function.name === selectedName) ||
        nextTools[0];
      if (current) selectTool(current);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function callTool() {
    if (!selected) return;
    setBusy(true);
    setStatus(`Выполняю ${selected.function.name}…`);
    try {
      const argumentsValue = JSON.parse(argumentsJson) as unknown;
      const metaValue = JSON.parse(metaJson) as unknown;
      if (
        !argumentsValue ||
        Array.isArray(argumentsValue) ||
        typeof argumentsValue !== "object"
      ) {
        throw new Error("Arguments должны быть JSON-объектом");
      }
      if (
        !metaValue ||
        Array.isArray(metaValue) ||
        typeof metaValue !== "object"
      ) {
        throw new Error("Meta должна быть JSON-объектом");
      }
      const response = await request<{ result: unknown }>(
        settings.agentsUrl,
        "/mcp-diagnostics/tools/call",
        token,
        {
          method: "POST",
          body: JSON.stringify({
            name: selected.function.name,
            arguments: argumentsValue,
            meta: metaValue,
          }),
        },
      );
      setResult(pretty(response.result));
      setStatus(`${selected.function.name}: успешно`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setResult(message);
      setStatus(`${selected.function.name}: ошибка`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mcp-console">
      <header>
        <div>
          <span className="eyebrow">ДИАГНОСТИКА · IDU MCP</span>
          <h1>MCP-консоль</h1>
          <p>Каталог и прямой запуск инструментов текущего MCP-контура</p>
        </div>
        <div className="mcp-header-actions">
          <label className="mcp-token">
            Bearer-токен
            <span>
              <input
                type="password"
                value={manualToken}
                onChange={(event) => setManualToken(event.target.value)}
                placeholder={token ? "Токен установлен" : "Вставьте токен"}
              />
              <button
                onClick={() =>
                  setToken(manualToken.trim().replace(/^Bearer\s+/i, ""))
                }
              >
                Применить
              </button>
            </span>
          </label>
          <span className={`connection ${busy ? "pulse" : ""}`}>
            <i /> {status}
          </span>
          <button className="primary" disabled={busy} onClick={loadCatalog}>
            {busy ? "Подождите…" : "Обновить каталог"}
          </button>
        </div>
      </header>

      <div className="mcp-layout">
        <section className="mcp-catalog">
          <div className="panel-head">
            <strong>Инструменты</strong>
            <b>{tools.length}</b>
          </div>
          <div className="search">
            ⌕
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Имя или описание"
            />
          </div>
          <div className="mcp-tool-list">
            {visibleTools.map((tool) => (
              <button
                key={tool.function.name}
                className={selectedName === tool.function.name ? "active" : ""}
                onClick={() => selectTool(tool)}
              >
                <strong>{tool.function.name}</strong>
                <small>{tool.function.description || "Без описания"}</small>
              </button>
            ))}
            {!visibleTools.length && (
              <div className="empty-inline">
                {tools.length ? "Ничего не найдено" : "Загрузите каталог MCP"}
              </div>
            )}
          </div>
        </section>

        <section className="mcp-editor">
          {selected ? (
            <>
              <div className="mcp-tool-heading">
                <div>
                  <span className="eyebrow">ИНСТРУМЕНТ</span>
                  <h2>{selected.function.name}</h2>
                  <p>
                    {selected.function.description || "Описание отсутствует"}
                  </p>
                </div>
                <button className="primary" disabled={busy} onClick={callTool}>
                  Запустить
                </button>
              </div>
              <div className="mcp-editor-grid">
                <label>
                  Arguments JSON
                  <textarea
                    value={argumentsJson}
                    onChange={(event) => setArgumentsJson(event.target.value)}
                    spellCheck={false}
                  />
                </label>
                <label>
                  Input schema
                  <pre>{pretty(selected.function.parameters || {})}</pre>
                </label>
              </div>
              <details className="mcp-meta">
                <summary>Meta JSON</summary>
                <textarea
                  value={metaJson}
                  onChange={(event) => setMetaJson(event.target.value)}
                  spellCheck={false}
                />
              </details>
              <div className="mcp-result">
                <div className="panel-head">
                  <strong>Результат</strong>
                  {result && (
                    <button
                      onClick={() => navigator.clipboard.writeText(result)}
                    >
                      Копировать
                    </button>
                  )}
                </div>
                <pre>{result || "Ответ инструмента появится здесь"}</pre>
              </div>
            </>
          ) : (
            <div className="empty">Выберите инструмент из каталога</div>
          )}
        </section>

        <aside className="mcp-prompts">
          <div className="panel-head">
            <strong>Промпты MCP</strong>
            <b>{prompts.length}</b>
          </div>
          {prompts.map((prompt, index) => (
            <details key={prompt.name || index}>
              <summary>{prompt.name || `Промпт ${index + 1}`}</summary>
              <p>{prompt.description || "Без описания"}</p>
              {!!prompt.arguments?.length && (
                <ul>
                  {prompt.arguments.map((argument, argumentIndex) => (
                    <li key={argument.name || argumentIndex}>
                      <code>{argument.name}</code>
                      {argument.required ? " · обязательно" : ""}
                    </li>
                  ))}
                </ul>
              )}
            </details>
          ))}
          {!prompts.length && (
            <div className="empty-inline">Промпты ещё не загружены</div>
          )}
        </aside>
      </div>
    </div>
  );
}
