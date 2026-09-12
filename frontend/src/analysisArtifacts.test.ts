import assert from "node:assert/strict";
import test from "node:test";
import { extractStoredLayers, analysisComplete } from "./analysisArtifacts.ts";

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
