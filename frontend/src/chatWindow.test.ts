import assert from "node:assert/strict";
import test from "node:test";

import {
  appendFilePart,
  appendIterationChunk,
  appendSseExchange,
  finalizeSseExchange,
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

test("a revised SSE answer is reset once and then accumulated", () => {
  const base = "**План работы**\n\n";
  const first = appendIterationChunk(
    "Черновой ответ",
    base,
    "Уточнённый ",
    2,
    1,
  );
  const second = appendIterationChunk(
    first.answer,
    base,
    "ответ",
    2,
    first.iteration,
  );
  const terminal = appendIterationChunk(
    second.answer,
    base,
    "",
    2,
    second.iteration,
  );

  assert.equal(terminal.answer, "**План работы**\n\nУточнённый ответ");
  assert.equal(terminal.iteration, 2);
});

test("ordinary SSE chunks keep accumulating without an iteration marker", () => {
  const result = appendIterationChunk("Первый ", "", "второй", undefined, 1);

  assert.equal(result.answer, "Первый второй");
  assert.equal(result.iteration, 1);
});

test("terminal server error survives stream EOF and cannot be replaced by fallback", () => {
  const exchange = { answer: "", finalized: false };
  const error = "Не удалось проверить ответ по источникам";
  assert.equal(finalizeSseExchange(exchange, error), true);
  assert.equal(
    finalizeSseExchange(
      exchange,
      "Поток завершился без отдельного финального сообщения.",
    ),
    false,
  );
  assert.equal(exchange.answer, error);
});

test("finalization preserves the completed answer", () => {
  const exchange = { answer: "Текст пункта [1]", finalized: false };
  finalizeSseExchange(exchange, "fallback");
  assert.equal(exchange.answer, "Текст пункта [1]");
});

const report = {
  name: "compliance_report",
  title: "Отчёт о проверке соответствия нормам",
  url: "http://gmart/files/compliance_report/abc",
  filename: "compliance_report_772.md",
};

test("file arriving with the exchange becomes a part of the assistant message", () => {
  const result = appendSseExchange("chat", [], {
    question: "Проверь нормы",
    answer: "Готово",
    tables: [],
    files: [report],
  });
  const assistant = result.messages.at(-1)!;
  assert.deepEqual(
    assistant.parts.map((part) => [part.part_seq, part.kind]),
    [
      [1, "text"],
      [2, "file"],
    ],
  );
});

test("late file is attached once to the latest assistant message", () => {
  const messages = [message("q", 1), message("a", 2), message("q2", 3)];
  const next = appendFilePart(messages, report);
  assert.deepEqual(
    next[1].parts.map((part) => [part.part_seq, part.kind]),
    [
      [1, "text"],
      [2, "file"],
    ],
  );
  assert.equal(messages[1].parts.length, 1, "input is not mutated");
  assert.equal(appendFilePart(next, report), next);
  assert.equal(appendFilePart([message("q", 1)], report).length, 1);
});
