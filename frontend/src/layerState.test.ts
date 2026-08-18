import assert from "node:assert/strict";
import test from "node:test";
import { appendLatestVisibleLayer } from "./layerState.ts";
import type { LayerData } from "./types.ts";

function layer(id: string, visible: boolean): LayerData {
  return {
    id,
    name: id,
    color: "#000",
    visible,
    geojson: { type: "FeatureCollection", features: [] },
    count: 0,
  };
}

test("a newly received layer is the only visible layer", () => {
  const result = appendLatestVisibleLayer(
    [layer("first", true), layer("second", false)],
    layer("latest", false),
  );

  assert.deepEqual(
    result.map(({ id, visible }) => ({ id, visible })),
    [
      { id: "first", visible: false },
      { id: "second", visible: false },
      { id: "latest", visible: true },
    ],
  );
});
