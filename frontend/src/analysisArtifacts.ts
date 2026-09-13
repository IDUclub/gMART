import type { LayerData, Message, ComplianceSummary, TableData } from "./types";

export function storedContinuation(messages: Message[], chatId: string) {
  for (const message of [...messages].reverse()) {
    if (message.role !== "assistant") continue;
    const snapshot = message.parts.find(p => p.kind === "data" && p.payload.event_type === "analysis_context")?.payload.content;
    if (!snapshot) continue;
    return snapshot.status === "blocked" && typeof snapshot.continue_from === "string"
      ? {id: snapshot.continue_from, chatId, scenario: String(snapshot.scenario_id ?? "")} : null;
  }
  return null;
}

export function storedTables(messages: Message[]): TableData[] {
  const tables = new Map<string, TableData>();
  for (const message of messages) {
    const snapshot = message.parts.find(p => p.kind === "data" && p.payload.event_type === "analysis_context")?.payload.content;
    if (snapshot) {
      for (const artifact of snapshot.artifacts || []) {
        if (artifact.kind === "table" && artifact.confirmed) tables.set(artifact.id, {...artifact.content, artifact_id: artifact.id});
      }
    } else for (const part of message.parts) {
      if (part.kind === "table") tables.set(part.payload.artifact_id || `${message.message_id}:${part.part_seq}`, part.payload as TableData);
    }
  }
  return [...tables.values()];
}

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
