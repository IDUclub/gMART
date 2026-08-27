import assert from "node:assert/strict";
import test from "node:test";
import { parseSseBuffer } from "./api.ts";

test("Synapse SSE parser preserves cursor, event name, and multiline data", () => {
  const parsed = parseSseBuffer<{ text: string }>(
    'id: 42-0\r\nevent: synapse_event\r\ndata: {"text":\r\ndata: "ready"}\r\n\r\n',
  );

  assert.deepEqual(parsed.frames, [
    {
      id: "42-0",
      event: "synapse_event",
      data: { text: "ready" },
    },
  ]);
  assert.equal(parsed.rest, "");
});

test("Synapse SSE parser keeps a partial frame for the next network chunk", () => {
  const first = parseSseBuffer<{ ok: boolean }>(
    'id: 43-0\r\nevent: synapse_event\r\ndata: {"ok":',
  );
  assert.equal(first.frames.length, 0);

  const second = parseSseBuffer<{ ok: boolean }>(first.rest + "true}\r\n\r\n");
  assert.deepEqual(second.frames[0], {
    id: "43-0",
    event: "synapse_event",
    data: { ok: true },
  });
});
