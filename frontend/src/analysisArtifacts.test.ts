import assert from "node:assert/strict";
import test from "node:test";
import { extractStoredLayers, analysisComplete, storedContinuation, storedTables } from "./analysisArtifacts.ts";

test("history restores continuation and original table identity without duplicating replay", () => {
  const table = {name: "schools", title: "Schools", columns: [], rows: [{service_id: 1}]};
  const message = {message_id: "m", role: "assistant", created_at: "", parts: [
    {part_seq: 0, kind: "table", payload: table},
    {part_seq: 1, kind: "data", payload: {event_type: "analysis_context", content: {status: "blocked", continue_from: "run-1", scenario_id: 17,
      artifacts: [{id: "stable", kind: "table", confirmed: true, content: table}]}}}
  ]};
  assert.deepEqual(storedContinuation([message], "chat-a"), {id: "run-1", chatId: "chat-a", scenario: "17"});
  assert.equal(storedTables([message, {...message, message_id: "m2"}]).length, 1);
  assert.equal(storedTables([message])[0].artifact_id, "stable");
  const completed = {...message, parts: [{part_seq: 0, kind: "data", payload: {event_type: "analysis_context", content: {status: "completed", continue_from: "run-2"}}}]};
  assert.equal(storedContinuation([message, completed], "chat-a"), null);
});

test("history restores full confirmed layers and deduplicates reused artifacts", () => {
  const layer = { part_seq: 0, kind: "data", payload: { event_type: "feature_collection", artifact_id: "a1", confirmed: true,
    content: { name: "Школы", feature_collection: { type: "FeatureCollection", features: [{ type: "Feature", geometry: null, properties: { id: 1 } }] } } } };
  const message = {message_id: "m", role: "assistant", created_at: "", parts: [layer]};
  const layers = extractStoredLayers([message, {...message, message_id: "m2"}, {...message, parts: [{...layer, payload: {...layer.payload, artifact_id: "draft", confirmed: false}}]}], ["#aaa"]);
  assert.equal(layers.length, 1);
  assert.equal(layers[0].geojson.features[0].properties?.id, 1);
  assert.equal(layers[0].name, "Школы");
});

test("a blocked analysis is not success even when individual steps completed", () => {
  assert.equal(analysisComplete({status: "blocked", steps: [{status: "completed"}]}), false);
  assert.equal(analysisComplete({status: "completed", steps: []}), true);
  assert.equal(analysisComplete({steps: [{status: "completed"}]}), true);
  assert.equal(analysisComplete({steps: []}), false);
});
