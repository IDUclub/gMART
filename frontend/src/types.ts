export type AgentId =
  | "orchestrator"
  | "restrictions"
  | "compliance"
  | "provision"
  | "scenario_data"
  | "documents"
  | "norms"
  | "llm";
export type Agent = {
  id: AgentId;
  label: string;
  caption: string;
  path: string;
  needsScenario: boolean;
  examples: string[];
};
export type TableData = {
  name?: string;
  title?: string;
  columns: Array<{ key: string; label: string }>;
  rows: Array<Record<string, unknown>>;
};
export type LayerData = {
  id: string;
  name: string;
  color: string;
  visible: boolean;
  geojson: GeoJSON.FeatureCollection;
  count: number;
};
export type StreamEvent = { type: string; content: any };
export type ChatSummary = {
  chat_id: string;
  title: string | null;
  scenario_id: string | number | null;
  project_id: string | number | null;
  updated_at: string;
  metadata?: Record<string, unknown>;
};
export type MessagePart = {
  part_seq: number;
  kind: string;
  payload: Record<string, any>;
  mcp_source?: string | null;
};
export type Message = {
  message_id: string;
  chat_id?: string;
  seq?: number;
  role: string;
  parts: MessagePart[];
  created_at: string;
  updated_at?: string;
  metadata?: Record<string, unknown>;
};
export type Chat = ChatSummary & {
  messages: Message[];
  has_more?: boolean;
  next_before_seq?: number | null;
};
export type StatusEntry = {
  id: string;
  text: string;
  time: string;
  state: "active" | "done" | "warning";
};
export type UserDocument = {
  doc_id: string;
  name: string;
  version: string;
  other_versions?: string[];
  blocks?: string[];
  tags?: string[];
  node_count?: number;
  uploaded_at?: string | null;
  source?: string | null;
};
export type UserDocumentList = {
  count: number;
  documents: UserDocument[];
};
export type UserDocumentUpload = {
  job_id: string;
  status: string;
};
export type UserDocumentDeleteResult = {
  name: string;
  versions_removed: string[];
  points_deleted: number;
  points_updated: number;
};
export type UserDocumentJobStatus = {
  job_id: string;
  status: "queued" | "processing" | "done" | "error";
  filename?: string | null;
  stage?: string | null;
  stage_index?: number | null;
  stage_total?: number | null;
  phase?: string | null;
  progress?: number | null;
  progress_total?: number | null;
  task_progress?: number | null;
  overall_progress?: number | null;
  doc_id?: string | null;
  name?: string | null;
  version?: string | null;
  nodes?: number | null;
  error?: string | null;
};
export type Settings = {
  theme: "light" | "dark";
  basemap: string;
  agentsUrl: string;
  chatStorageUrl: string;
  authHelperUrl: string;
  keycloakUrl: string;
  keycloakRealm: string;
  keycloakClientId: string;
  model: string;
  temperature: number;
};
