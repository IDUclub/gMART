import assert from "node:assert/strict";
import test from "node:test";
import { storedSource } from "./analysisArtifacts.ts";

test("a confirmed source remains readable from chat history after runtime cache expiry", () => {
  const artifact = {id: "source:a1", kind: "source_evidence", confirmed: true,
    content: {system: "documents", sources: [{id: "clause", version: "2026", text: "50 m"}]}};
  const message = {message_id: "message", role: "assistant", created_at: "", parts: [
    {part_seq: 0, kind: "data", payload: {event_type: "analysis_context",
      content: {continue_from: "run", artifacts: [artifact]}}}
  ]};
  assert.deepEqual(storedSource([message], "run", "source:a1"), artifact);
  assert.equal(storedSource([message], "other-run", "source:a1"), null);
  assert.equal(storedSource([message], "run", "other-artifact"), null);
  const draft = {...message, parts: [{part_seq: 0, kind: "data", payload: {event_type: "analysis_context",
    content: {continue_from: "run", artifacts: [{...artifact, confirmed: false}]}}}]};
  assert.equal(storedSource([draft], "run", "source:a1"), null);
});
