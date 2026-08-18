import type { LayerData } from "./types";

export function appendLatestVisibleLayer(
  layers: LayerData[],
  layer: LayerData,
): LayerData[] {
  return [
    ...layers.map((existing) => ({ ...existing, visible: false })),
    { ...layer, visible: true },
  ];
}
