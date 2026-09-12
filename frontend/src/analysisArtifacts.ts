import type { LayerData, Message, ComplianceSummary } from "./types";

export function extractStoredComplianceSummary(messages: Message[]): ComplianceSummary | null {
  for (const message of [...messages].reverse()) {
    const part = [...message.parts].reverse().find(part => part.kind === "compliance_summary");
    if (part) return part.payload as ComplianceSummary;
  }
  return null;
}

export function extractStoredLayers(messages: Message[], colors: string[]): LayerData[] {
  const layers = new Map<string, LayerData>();
  for (const message of messages) {
    for (const part of message.parts) {
      const payload = part.payload;
      if (part.kind !== "data" || payload.event_type !== "feature_collection" || payload.confirmed !== true) continue;
      const content = payload.content;
      const geojson = content?.feature_collection || content?.data || content;
      if (geojson?.type !== "FeatureCollection" || !Array.isArray(geojson.features)) continue;
      const id = payload.artifact_id || `${message.message_id}:${part.part_seq}`;
      layers.set(id, { id, name: content.name || "Сохранённый слой", color: colors[layers.size % colors.length],
        visible: true, geojson, count: geojson.features.length });
    }
  }
  return [...layers.values()];
}

export function analysisComplete(content: { status?: string; steps?: Array<{status?: string}> }): boolean {
  if (content.status) return content.status === "completed";
  return Boolean(content.steps?.length && content.steps.every(step => step.status === "completed"));
}
