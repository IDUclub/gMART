import type {
  Chat,
  ChatSummary,
  Settings,
  StreamEvent,
  UserDocumentDeleteResult,
  UserDocumentJobStatus,
  UserDocumentList,
  UserDocumentUpload,
} from "./types";

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
      body?.detail?.message ??
        body?.detail ??
        body?.message ??
        `Ошибка API: ${response.status}`,
    );
  }
  return response.json();
}

export async function readSse<T = StreamEvent>(
  url: URL,
  token: string,
  signal: AbortSignal,
  onEvent: (event: T) => void,
) {
  const response = await fetch(url, {
    headers: { Authorization: `Bearer ${token}`, Accept: "text/event-stream" },
    signal,
  });
  if (!response.ok)
    throw new Error(
      `Не удалось открыть поток: ${response.status} ${await response.text()}`,
    );
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

export const getChat = (
  settings: Settings,
  token: string,
  id: string,
  page?: { limit: number; beforeSeq?: number | null },
) => {
  const query = new URLSearchParams();
  if (page) query.set("message_limit", String(page.limit));
  if (page?.beforeSeq != null) query.set("before_seq", String(page.beforeSeq));
  const suffix = query.size ? `?${query}` : "";
  return request<Chat>(
    settings.chatStorageUrl,
    `/api/v1/chat_history/${id}${suffix}`,
    token,
  );
};

export const deleteChat = (settings: Settings, token: string, id: string) =>
  request(settings.chatStorageUrl, `/api/v1/chat_history/${id}`, token, {
    method: "DELETE",
  });

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
      body?.detail?.message ??
        body?.detail ??
        body?.message ??
        `Ошибка входа: ${response.status}`,
    );
  }
  return response.json();
}

export function userDocumentScopeParams(
  scenario?: string,
  project?: string,
): URLSearchParams {
  const query = new URLSearchParams();
  if (scenario?.trim()) query.set("scenario_id", scenario.trim());
  if (project?.trim()) query.set("project_id", project.trim());
  return query;
}

export function userDocumentPath(name: string) {
  return `/documents/user-documents/${encodeURIComponent(name)}`;
}

export function listUserDocuments(
  settings: Settings,
  token: string,
  scenario?: string,
  project?: string,
) {
  const query = userDocumentScopeParams(scenario, project);
  return request<UserDocumentList>(
    settings.agentsUrl,
    `/documents/user-documents?${query}`,
    token,
  );
}

export async function uploadUserDocument(
  settings: Settings,
  token: string,
  payload: {
    file: File;
    scenario?: string;
    project?: string;
    name?: string;
    version?: string;
  },
  onProgress?: (progress: number) => void,
): Promise<UserDocumentUpload> {
  const form = new FormData();
  form.set("file", payload.file);
  if (payload.scenario?.trim())
    form.set("scenario_id", payload.scenario.trim());
  if (payload.project?.trim()) form.set("project_id", payload.project.trim());
  if (payload.name?.trim()) form.set("name", payload.name.trim());
  if (payload.version?.trim()) form.set("version", payload.version.trim());

  const url = new URL(
    "documents/user-documents",
    settings.agentsUrl.replace(/\/+$/, "") + "/",
  );
  return sendUserDocumentFile(url, "POST", token, form, onProgress);
}

export async function updateUserDocument(
  settings: Settings,
  token: string,
  name: string,
  payload: {
    file: File;
    scenario?: string;
    project?: string;
    version?: string;
  },
  onProgress?: (progress: number) => void,
): Promise<UserDocumentUpload> {
  const form = new FormData();
  form.set("file", payload.file);
  if (payload.scenario?.trim())
    form.set("scenario_id", payload.scenario.trim());
  if (payload.project?.trim()) form.set("project_id", payload.project.trim());
  if (payload.version?.trim()) form.set("version", payload.version.trim());

  const url = new URL(
    userDocumentPath(name).replace(/^\//, ""),
    settings.agentsUrl.replace(/\/+$/, "") + "/",
  );
  return sendUserDocumentFile(url, "PATCH", token, form, onProgress);
}

function sendUserDocumentFile(
  url: URL,
  method: "POST" | "PATCH",
  token: string,
  form: FormData,
  onProgress?: (progress: number) => void,
): Promise<UserDocumentUpload> {
  return new Promise<UserDocumentUpload>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) {
        onProgress?.(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onerror = () => reject(new Error("Сеть недоступна во время загрузки"));
    xhr.onabort = () =>
      reject(new DOMException("Загрузка отменена", "AbortError"));
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress?.(100);
        resolve(body as UserDocumentUpload);
        return;
      }
      reject(
        new Error(
          body?.detail?.message ??
            body?.detail ??
            body?.message ??
            `Ошибка обработки документа: ${xhr.status}`,
        ),
      );
    };
    xhr.send(form);
  });
}

export function deleteUserDocument(
  settings: Settings,
  token: string,
  name: string,
  scenario?: string,
  project?: string,
  version?: string,
) {
  const query = userDocumentScopeParams(scenario, project);
  if (version?.trim()) query.set("version", version.trim());
  const suffix = query.size ? `?${query}` : "";
  return request<UserDocumentDeleteResult>(
    settings.agentsUrl,
    `${userDocumentPath(name)}${suffix}`,
    token,
    { method: "DELETE" },
  );
}

export function readUserDocumentJob(
  settings: Settings,
  token: string,
  jobId: string,
  signal: AbortSignal,
  onStatus: (status: UserDocumentJobStatus) => void,
) {
  const url = new URL(
    `documents/user-documents/jobs/${encodeURIComponent(jobId)}/stream`,
    settings.agentsUrl.replace(/\/+$/, "") + "/",
  );
  return readSse<UserDocumentJobStatus>(url, token, signal, onStatus);
}
