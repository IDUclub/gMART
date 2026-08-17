import assert from "node:assert/strict";
import test from "node:test";

import {
  appendSseExchange,
  mergeMessageWindow,
  oldestServerSequence,
  trimMessageWindow,
} from "./chatWindow.ts";
import type { Message } from "./types.ts";

function message(id: string, seq: number, text = id): Message {
  return {
    message_id: id,
    chat_id: "chat",
    seq,
    role: seq % 2 ? "user" : "assistant",
    parts: [{ part_seq: 1, kind: "text", payload: { text } }],
    created_at: `2026-08-17T00:00:0${seq}Z`,
  };
}

test("SSE exchange is appended as the authoritative user/assistant pair", () => {
  let id = 0;
  const result = appendSseExchange(
    "chat",
    [message("stored", 1)],
    { question: "Вопрос", answer: "**Ответ**", tables: [] },
    () => String(++id),
    () => "2026-08-17T12:00:00Z",
  );

  assert.deepEqual(
    result.messages.map((item) => [item.seq, item.role]),
    [
      [1, "user"],
      [2, "user"],
      [3, "assistant"],
    ],
  );
  assert.equal(result.messages[2].parts[0].payload.text, "**Ответ**");
  assert.equal(result.messages[2].metadata?.source, "sse");
});

test("oldest messages are evicted when the browser window reaches its limit", () => {
  const result = trimMessageWindow(
    [message("one", 1), message("two", 2), message("three", 3)],
    { maxMessages: 2, maxBytes: Number.MAX_SAFE_INTEGER },
  );

  assert.deepEqual(
    result.messages.map((item) => item.message_id),
    ["two", "three"],
  );
  assert.deepEqual(
    result.removed.map((item) => item.message_id),
    ["one"],
  );
});

test("current browser messages win when an older page is merged", () => {
  const current = message("same", 2, "SSE");
  current.metadata = { source: "sse" };
  const result = mergeMessageWindow(
    [message("old", 1), message("same", 2, "storage")],
    [current],
  );

  assert.equal(result.messages[1].parts[0].payload.text, "SSE");
  assert.equal(oldestServerSequence(result.messages), 1);
});
