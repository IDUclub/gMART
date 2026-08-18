import assert from "node:assert/strict";
import test from "node:test";

import { reusableChatId } from "./agentSession.ts";
import type { Chat } from "./types.ts";

function chat(agentId?: string): Chat {
  return {
    chat_id: "existing-chat",
    title: null,
    scenario_id: null,
    project_id: null,
    updated_at: "2026-08-18T17:25:00Z",
    metadata: agentId ? { agent_id: agentId } : {},
    messages: [],
  };
}

test("a chat id is reused only by the agent that created the chat", () => {
  assert.equal(
    reusableChatId(chat("restrictions"), "restrictions"),
    "existing-chat",
  );
  assert.equal(reusableChatId(chat("restrictions"), "norms"), undefined);
  assert.equal(reusableChatId(chat(), "restrictions"), undefined);
});
