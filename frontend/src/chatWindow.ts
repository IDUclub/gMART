import type { Message, TableData } from "./types";

export const CHAT_PAGE_SIZE = 40;
export const MAX_CHAT_MESSAGES = 200;
export const MAX_CHAT_BYTES = 6 * 1024 * 1024;

export type MessageWindowLimits = {
  maxMessages: number;
  maxBytes: number;
};

export type SseExchange = {
  question: string;
  answer: string;
  tables: TableData[];
};

const defaultLimits: MessageWindowLimits = {
  maxMessages: MAX_CHAT_MESSAGES,
  maxBytes: MAX_CHAT_BYTES,
};

export function estimateMessageBytes(message: Message): number {
  try {
    return new TextEncoder().encode(JSON.stringify(message)).byteLength;
  } catch {
    return 0;
  }
}

export function trimMessageWindow(
  messages: Message[],
  limits: MessageWindowLimits = defaultLimits,
): { messages: Message[]; removed: Message[] } {
  const kept = [...messages];
  const removed: Message[] = [];
  let bytes = kept.reduce(
    (total, message) => total + estimateMessageBytes(message),
    0,
  );

  while (
    kept.length > 1 &&
    (kept.length > limits.maxMessages || bytes > limits.maxBytes)
  ) {
    const oldest = kept.shift();
    if (!oldest) break;
    removed.push(oldest);
    bytes -= estimateMessageBytes(oldest);
  }
  return { messages: kept, removed };
}

export function mergeMessageWindow(
  older: Message[],
  current: Message[],
  limits: MessageWindowLimits = defaultLimits,
) {
  const merged = new Map<string, Message>();
  for (const message of older) merged.set(message.message_id, message);
  // The current browser window is authoritative: SSE-backed messages win on collision.
  for (const message of current) merged.set(message.message_id, message);
  return trimMessageWindow([...merged.values()], limits);
}

export function appendSseExchange(
  chatId: string,
  current: Message[],
  exchange: SseExchange,
  makeId: () => string = () => crypto.randomUUID(),
  now: () => string = () => new Date().toISOString(),
  limits: MessageWindowLimits = defaultLimits,
) {
  const highestSeq = current.reduce(
    (highest, message) =>
      Number.isFinite(Number(message.seq))
        ? Math.max(highest, Number(message.seq))
        : highest,
    0,
  );
  const createdAt = now();
  const userMessage: Message = {
    message_id: `browser:${makeId()}`,
    chat_id: chatId,
    seq: highestSeq + 1,
    role: "user",
    parts: [
      { part_seq: 1, kind: "text", payload: { text: exchange.question } },
    ],
    metadata: { source: "sse" },
    created_at: createdAt,
    updated_at: createdAt,
  };
  const assistantMessage: Message = {
    message_id: `browser:${makeId()}`,
    chat_id: chatId,
    seq: highestSeq + 2,
    role: "assistant",
    parts: [
      {
        part_seq: 1,
        kind: "text",
        payload: {
          text:
            exchange.answer.trim() ||
            "Запрос завершён без текстового ответа. Результаты доступны в данных и на карте.",
        },
      },
      ...exchange.tables.map((table, index) => ({
        part_seq: index + 2,
        kind: "table",
        payload: table,
      })),
    ],
    metadata: { source: "sse" },
    created_at: createdAt,
    updated_at: createdAt,
  };

  return trimMessageWindow([...current, userMessage, assistantMessage], limits);
}

export function oldestServerSequence(messages: Message[]): number | null {
  for (const message of messages) {
    if (message.message_id.startsWith("browser:")) continue;
    const seq = Number(message.seq);
    if (Number.isFinite(seq)) return seq;
  }
  return null;
}
