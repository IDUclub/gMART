import type { UserDocumentJobStatus } from "./types";

export const DOCUMENT_PIPELINE_STAGES = [
  { key: "upload", label: "Передача файла" },
  { key: "structure-markup", label: "Разметка структуры" },
  { key: "type-tagging", label: "Типы и теги" },
  { key: "hierarchy", label: "Иерархия" },
  { key: "identity", label: "Название и версия" },
  { key: "references", label: "Ссылки" },
  { key: "embeddings", label: "Векторизация" },
  { key: "indexing", label: "Запись в индекс" },
] as const;

export type DocumentPipelineStageKey =
  (typeof DOCUMENT_PIPELINE_STAGES)[number]["key"];
export type DocumentPipelineProgress = Record<DocumentPipelineStageKey, number>;

const clamp = (value: number) => Math.max(0, Math.min(100, Math.round(value)));

export function emptyDocumentPipelineProgress(): DocumentPipelineProgress {
  return Object.fromEntries(
    DOCUMENT_PIPELINE_STAGES.map(({ key }) => [key, 0]),
  ) as DocumentPipelineProgress;
}

export function withUploadProgress(
  current: DocumentPipelineProgress,
  progress: number,
): DocumentPipelineProgress {
  return { ...current, upload: clamp(progress) };
}

export function withJobProgress(
  current: DocumentPipelineProgress,
  job: UserDocumentJobStatus,
): DocumentPipelineProgress {
  const next = { ...current, upload: 100 };
  if (job.status === "done") {
    for (const { key } of DOCUMENT_PIPELINE_STAGES) next[key] = 100;
    return next;
  }

  const currentIndex = job.stage_index ?? 0;
  for (let index = 1; index < DOCUMENT_PIPELINE_STAGES.length; index += 1) {
    const { key } = DOCUMENT_PIPELINE_STAGES[index];
    if (index < currentIndex) next[key] = 100;
    if (index === currentIndex) {
      next[key] = Math.max(next[key], clamp(job.task_progress ?? 0));
    }
  }
  return next;
}

export function documentStageLabel(stage?: string | null): string {
  return (
    DOCUMENT_PIPELINE_STAGES.find((item) => item.key === stage)?.label ??
    stage ??
    "Ожидание в очереди"
  );
}
