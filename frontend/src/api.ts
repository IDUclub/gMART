import type { Chat, ChatSummary, Settings, StreamEvent } from "./types";

export async function request<T>(
  base: string,
  path: string,
  token: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(new URL(path, base.replace(/\/+$/, "") + "/"), {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...init.headers,
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail?.message ?? body?.detail ?? body?.message ?? `Ошибка API: ${response.status}`,
    );
  }
  return response.json();
}

export async function readSse(
  url: URL,
  token: string,
  signal: AbortSignal,
  onEvent: (event: StreamEvent) => void,
) {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}`, Accept: "text/event-stream" },
    signal,
  });
  if (!response.ok)
    throw new Error(`Не удалось открыть поток: ${response.status} ${await response.text()}`);
  const reader = response.body?.getReader();
  if (!reader) throw new Error("Пустой SSE-поток");
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) onEvent(JSON.parse(data));
    }
  }
}

export async function getChats(settings: Settings, token: string) {
  const query = new URLSearchParams({ limit: "100", offset: "0" });
  return request<{ items: ChatSummary[] }>(
    settings.chatStorageUrl,
    `/api/v1/chat_history/chats?${query}`,
    token,
  );
}

export const getChat = (settings: Settings, token: string, id: string) =>
  request<Chat>(settings.chatStorageUrl, `/api/v1/chat_history/${id}`, token);

export const deleteChat = (settings: Settings, token: string, id: string) =>
  request(settings.chatStorageUrl, `/api/v1/chat_history/${id}`, token, { method: "DELETE" });

export const replayToolCall = (
  settings: Settings,
  token: string,
  messageId: string,
  partSeq: number,
  toolCall: number,
  scenario?: string,
  project?: string,
) => {
  const query = new URLSearchParams();
  if (scenario) query.set("scenario_id", scenario);
  if (project) query.set("project_id", project);
  return request<unknown>(
    settings.chatStorageUrl,
    `/api/v1/chat_history/messages/${messageId}/parts/${partSeq}/tool_calls/${toolCall}/execute?${query}`,
    token,
    { method: "GET" },
  );
};

export const getModels = (settings: Settings, token: string) =>
  request<string[]>(settings.agentsUrl, "/llm/available_models", token);

export async function authAvailable(settings: Settings): Promise<boolean> {
  try {
    const response = await fetch(
      new URL("auth/available", settings.agentsUrl.replace(/\/+$/, "") + "/"),
    );
    if (!response.ok) return false;
    return Boolean((await response.json())?.enabled);
  } catch {
    return false;
  }
}

export async function authLogin(
  settings: Settings,
  username: string,
  password: string,
): Promise<{ access_token: string; expires_in?: number }> {
  const response = await fetch(
    new URL("auth/token", settings.agentsUrl.replace(/\/+$/, "") + "/"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      body?.detail?.message ?? body?.detail ?? body?.message ?? `Ошибка входа: ${response.status}`,
    );
  }
  return response.json();
}
