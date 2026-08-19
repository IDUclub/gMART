import { useEffect, useMemo, useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import {
  CaretDown,
  CheckCircle,
  FileArrowUp,
  FileText,
  FolderOpen,
  PencilSimple,
  SpinnerGap,
  Trash,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import gsap from "gsap";
import {
  deleteUserDocument,
  listUserDocuments,
  readUserDocumentJob,
  updateUserDocument,
  uploadUserDocument,
} from "./api";
import {
  DOCUMENT_PIPELINE_STAGES,
  documentStageLabel,
  emptyDocumentPipelineProgress,
  withJobProgress,
  withUploadProgress,
} from "./documentProgress";
import type { DocumentPipelineProgress } from "./documentProgress";
import type { Settings, UserDocument, UserDocumentJobStatus } from "./types";

type Props = {
  settings: Settings;
  token: string;
  scenario: string;
  project: string;
  getToken: () => Promise<string>;
};

const SUPPORTED_FORMATS = ["DOCX", "TXT", "MD", "HTML", "HTM"];

function fileStem(filename: string) {
  return filename.replace(/\.[^.]+$/, "").trim();
}

function readableDate(value?: string | null) {
  if (!value) return "дата не указана";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString("ru", { dateStyle: "medium", timeStyle: "short" });
}

export default function DocumentLibrary({
  settings,
  token,
  scenario,
  project,
  getToken,
}: Props) {
  const root = useRef<HTMLElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const streamAbort = useRef<AbortController | null>(null);
  const [open, setOpen] = useState(true);
  const [documents, setDocuments] = useState<UserDocument[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [version, setVersion] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [busyDocumentId, setBusyDocumentId] = useState<string | null>(null);
  const [deleteCandidateId, setDeleteCandidateId] = useState<string | null>(
    null,
  );
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(
    null,
  );
  const [jobStatus, setJobStatus] = useState<UserDocumentJobStatus | null>(
    null,
  );
  const [overallProgress, setOverallProgress] = useState(0);
  const [stageProgress, setStageProgress] = useState<DocumentPipelineProgress>(
    emptyDocumentPipelineProgress,
  );
  const [feedback, setFeedback] = useState<{
    state: "idle" | "working" | "done" | "error";
    text: string;
  }>({ state: "idle", text: "Файл попадёт в индекс выбранного проекта" });

  const hasScope = Boolean(scenario.trim() || project.trim());
  const accept = useMemo(
    () =>
      SUPPORTED_FORMATS.map((format) => `.${format.toLowerCase()}`).join(","),
    [],
  );

  useGSAP(
    () => {
      if (!open) return;
      gsap.fromTo(
        ".document-panel-body > *",
        { opacity: 0, scale: 0.96, y: 12 },
        { opacity: 1, scale: 1, y: 0, duration: 0.46, stagger: 0.07 },
      );
      gsap.fromTo(
        ".document-heading-word",
        { opacity: 0.12 },
        { opacity: 1, duration: 0.38, stagger: 0.06 },
      );
    },
    { scope: root, dependencies: [open] },
  );

  async function refresh(silent = false, accessToken = token) {
    if (!accessToken || !hasScope) {
      setDocuments([]);
      return [];
    }
    if (!silent) setLoading(true);
    try {
      const result = await listUserDocuments(
        settings,
        accessToken,
        scenario,
        project,
      );
      setDocuments(result.documents || []);
      return result.documents || [];
    } catch (error) {
      setFeedback({ state: "error", text: String((error as Error).message) });
      return [];
    } finally {
      if (!silent) setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
    return () => {
      streamAbort.current?.abort();
    };
    // Refresh is intentionally keyed to the selected user scope.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, scenario, project, settings.agentsUrl]);

  function chooseFile(next: File | null) {
    setFile(next);
    if (next) {
      setName((current) => current || fileStem(next.name));
      setFeedback({ state: "idle", text: `${next.name} готов к загрузке` });
    }
  }

  function applyJobStatus(status: UserDocumentJobStatus) {
    setJobStatus(status);
    setStageProgress((current) => withJobProgress(current, status));
    setOverallProgress((current) =>
      Math.max(current, status.overall_progress ?? 10),
    );
    if (status.status === "error") {
      setFeedback({
        state: "error",
        text: status.error || "IDU_DVD не смог проиндексировать документ",
      });
      return;
    }
    if (status.status === "done") {
      setFeedback({
        state: "done",
        text: "Документ проиндексирован и доступен агенту",
      });
      return;
    }
    const phase = status.phase ? ` · ${status.phase}` : "";
    setFeedback({
      state: "working",
      text: `${documentStageLabel(status.stage)}${phase}`,
    });
  }

  function beginDocumentProgress(message: string) {
    streamAbort.current?.abort();
    setUploading(true);
    setJobStatus(null);
    setOverallProgress(0);
    setStageProgress(emptyDocumentPipelineProgress());
    setFeedback({ state: "working", text: message });
  }

  function applyTransferProgress(progress: number) {
    setStageProgress((current) => withUploadProgress(current, progress));
    setOverallProgress(Math.round(progress / 10));
  }

  async function followDocumentJob(jobId: string, initialToken: string) {
    setFeedback({ state: "working", text: "Ожидание очереди IDU_DVD" });
    let accessToken = initialToken;
    let terminal: UserDocumentJobStatus | null = null;
    for (let attempt = 0; attempt < 3 && !terminal; attempt += 1) {
      const controller = new AbortController();
      streamAbort.current = controller;
      if (attempt > 0) accessToken = await getToken();
      try {
        await readUserDocumentJob(
          settings,
          accessToken,
          jobId,
          controller.signal,
          (status) => {
            applyJobStatus(status);
            if (status.status === "done" || status.status === "error") {
              terminal = status;
            }
          },
        );
      } catch (error) {
        if ((error as Error).name === "AbortError" || attempt === 2) {
          throw error;
        }
      }
      if (!terminal && attempt < 2) {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));
      }
    }
    if (!terminal) {
      throw new Error("Поток прогресса завершился до окончания индексации");
    }
    if ((terminal as UserDocumentJobStatus).status === "done") {
      await refresh(true, accessToken);
    }
    return terminal as UserDocumentJobStatus;
  }

  async function upload() {
    if (!file || !hasScope || uploading) return;
    setBusyDocumentId(null);
    beginDocumentProgress("Передаю файл в IDU_DVD");
    try {
      const documentName = name.trim() || fileStem(file.name);
      const accessToken = await getToken();
      const uploadResult = await uploadUserDocument(
        settings,
        accessToken,
        {
          file,
          scenario,
          project,
          name: documentName,
          version,
        },
        applyTransferProgress,
      );
      setFile(null);
      setVersion("");
      if (fileInput.current) fileInput.current.value = "";
      await followDocumentJob(uploadResult.job_id, accessToken);
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setFeedback({ state: "error", text: String((error as Error).message) });
      }
    } finally {
      setUploading(false);
    }
  }

  async function updateDocument(document: UserDocument, nextFile: File) {
    if (!hasScope || uploading || deletingDocumentId) return;
    setBusyDocumentId(document.doc_id);
    setDeleteCandidateId(null);
    beginDocumentProgress(`Передаю новую версию «${document.name}»`);
    try {
      const accessToken = await getToken();
      const result = await updateUserDocument(
        settings,
        accessToken,
        document.name,
        { file: nextFile, scenario, project },
        applyTransferProgress,
      );
      const terminal = await followDocumentJob(result.job_id, accessToken);
      if (terminal.status === "done") {
        setFeedback({
          state: "done",
          text: `Документ «${document.name}» обновлён`,
        });
      }
    } catch (error) {
      if ((error as Error).name !== "AbortError") {
        setFeedback({ state: "error", text: String((error as Error).message) });
      }
    } finally {
      setUploading(false);
      setBusyDocumentId(null);
    }
  }

  async function removeDocument(document: UserDocument, rowId: string) {
    if (uploading || deletingDocumentId) return;
    if (deleteCandidateId !== rowId) {
      setDeleteCandidateId(rowId);
      setFeedback({
        state: "idle",
        text: `Подтвердите удаление «${document.name}»`,
      });
      return;
    }
    setDeletingDocumentId(rowId);
    try {
      const accessToken = await getToken();
      await deleteUserDocument(
        settings,
        accessToken,
        document.name,
        scenario,
        project,
      );
      setDeleteCandidateId(null);
      setExpandedId(null);
      await refresh(true, accessToken);
      setFeedback({
        state: "done",
        text: `Документ «${document.name}» удалён`,
      });
    } catch (error) {
      setFeedback({ state: "error", text: String((error as Error).message) });
    } finally {
      setDeletingDocumentId(null);
    }
  }

  const FeedbackIcon =
    feedback.state === "done"
      ? CheckCircle
      : feedback.state === "error"
        ? WarningCircle
        : feedback.state === "working"
          ? SpinnerGap
          : FileText;

  return (
    <section className={`document-library ${open ? "open" : ""}`} ref={root}>
      <button
        className="document-panel-toggle"
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
      >
        <span className="document-panel-icon">
          <FolderOpen weight="duotone" />
        </span>
        <span className="document-panel-title">
          <strong>
            {"Свои документы".split(" ").map((word) => (
              <span className="document-heading-word" key={word}>
                {word}{" "}
              </span>
            ))}
          </strong>
          <small>
            {documents.length
              ? `${documents.length} в текущем проекте`
              : "Персональный индекс IDU_DVD"}
          </small>
        </span>
        <span className={`document-feedback ${feedback.state}`}>
          <FeedbackIcon className={uploading ? "spin" : ""} />
          {feedback.text}
        </span>
        <CaretDown className="document-panel-caret" />
      </button>

      {open && (
        <div className="document-panel-body">
          <div className="document-upload-card">
            <label className="document-file-drop">
              <input
                ref={fileInput}
                type="file"
                accept={accept}
                onChange={(event) =>
                  chooseFile(event.target.files?.[0] || null)
                }
              />
              <FileArrowUp weight="duotone" />
              <span>
                <strong>{file ? file.name : "Выберите документ"}</strong>
                <small>
                  {file
                    ? `${Math.max(1, Math.ceil(file.size / 1024))} КБ`
                    : "Файл останется в индексе вашего проекта"}
                </small>
              </span>
            </label>
            <div className="document-upload-fields">
              <label>
                Название
                <input
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="По имени файла"
                />
              </label>
              <label>
                Версия
                <input
                  value={version}
                  onChange={(event) => setVersion(event.target.value)}
                  placeholder="необязательно"
                />
              </label>
              <button
                className="primary document-upload-action"
                type="button"
                disabled={!file || !hasScope || uploading}
                onClick={() => void upload()}
              >
                {uploading ? <SpinnerGap className="spin" /> : <FileArrowUp />}
                {uploading ? "Индексирую" : "Загрузить"}
              </button>
            </div>
            {!hasScope && (
              <p className="document-scope-warning">
                Укажите сценарий или проект в верхней панели.
              </p>
            )}
            <div className="format-marquee" aria-label="Поддерживаемые форматы">
              <div>
                {[...SUPPORTED_FORMATS, ...SUPPORTED_FORMATS].map(
                  (format, index) => (
                    <span key={`${format}-${index}`}>{format}</span>
                  ),
                )}
              </div>
            </div>
          </div>

          <div className="document-accordion">
            <header>
              <div>
                <strong>Доступно агенту</strong>
                <small>Общая база и документы текущего проекта</small>
              </div>
              <button
                type="button"
                onClick={() => void refresh()}
                disabled={loading || !hasScope}
              >
                {loading ? <SpinnerGap className="spin" /> : "Обновить"}
              </button>
            </header>
            <div className="document-accordion-list">
              {!documents.length ? (
                <div className="document-empty">
                  <FileText weight="thin" />
                  <span>
                    <strong>Пока нет своих документов</strong>
                    <small>Загрузите файл, затем задайте вопрос агенту</small>
                  </span>
                </div>
              ) : (
                documents.map((document) => {
                  const rowId = `${document.doc_id}:${document.version}`;
                  const expanded = expandedId === rowId;
                  const confirmingDelete = deleteCandidateId === rowId;
                  const deleting = deletingDocumentId === rowId;
                  const updating = busyDocumentId === document.doc_id;
                  return (
                    <div
                      className={`document-row-shell ${expanded ? "expanded" : ""}`}
                      key={rowId}
                    >
                      <button
                        type="button"
                        className="document-row"
                        onClick={() => {
                          setExpandedId(expanded ? null : rowId);
                          setDeleteCandidateId(null);
                        }}
                      >
                        <FileText weight="duotone" />
                        <span>
                          <strong>{document.name}</strong>
                          <small>
                            {document.version || "без версии"} ·{" "}
                            {document.node_count || 0} фрагментов
                          </small>
                        </span>
                        <CaretDown />
                      </button>
                      {expanded && (
                        <div className="document-row-detail">
                          <p>
                            Загружен {readableDate(document.uploaded_at)}. Агент
                            ищет по этому документу при выбранном сценарии.
                          </p>
                          <div className="document-row-actions">
                            <label
                              className={`document-update-action ${uploading || deletingDocumentId ? "disabled" : ""}`}
                            >
                              <input
                                type="file"
                                accept={accept}
                                disabled={
                                  uploading || Boolean(deletingDocumentId)
                                }
                                onChange={(event) => {
                                  const nextFile = event.target.files?.[0];
                                  event.currentTarget.value = "";
                                  if (nextFile)
                                    void updateDocument(document, nextFile);
                                }}
                              />
                              {updating ? (
                                <SpinnerGap className="spin" />
                              ) : (
                                <PencilSimple />
                              )}
                              {updating ? "Обновляю" : "Обновить файлом"}
                            </label>
                            <button
                              className={`document-delete-action ${confirmingDelete ? "confirm" : ""}`}
                              type="button"
                              disabled={
                                uploading || Boolean(deletingDocumentId)
                              }
                              onClick={() =>
                                void removeDocument(document, rowId)
                              }
                            >
                              {deleting ? (
                                <SpinnerGap className="spin" />
                              ) : (
                                <Trash />
                              )}
                              {deleting
                                ? "Удаляю"
                                : confirmingDelete
                                  ? "Удалить безвозвратно"
                                  : "Удалить"}
                            </button>
                            {confirmingDelete && !deleting && (
                              <button
                                className="document-cancel-action"
                                type="button"
                                onClick={() => setDeleteCandidateId(null)}
                              >
                                <X />
                                Отмена
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {(uploading || jobStatus || overallProgress > 0) && (
            <div
              className={`document-progress-card ${jobStatus?.status || "uploading"}`}
            >
              <header>
                <div>
                  <strong>Весь пайплайн</strong>
                  <small>
                    {jobStatus?.status === "queued"
                      ? "Задача ожидает вычислительный слот"
                      : jobStatus?.status === "done"
                        ? `${jobStatus.nodes || 0} фрагментов готовы к поиску`
                        : jobStatus?.phase ||
                          documentStageLabel(jobStatus?.stage)}
                  </small>
                </div>
                <b>{Math.round(overallProgress)}%</b>
              </header>
              <div
                className="document-overall-track"
                role="progressbar"
                aria-label="Общий прогресс индексации"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(overallProgress)}
              >
                <span style={{ width: `${overallProgress}%` }} />
              </div>
              <div className="document-stage-grid">
                {DOCUMENT_PIPELINE_STAGES.map(({ key, label }) => {
                  const progress = stageProgress[key];
                  const active =
                    (key === "upload" && !jobStatus) ||
                    jobStatus?.stage === key;
                  return (
                    <div
                      className={`document-stage ${active ? "active" : ""} ${progress === 100 ? "complete" : ""}`}
                      key={key}
                    >
                      <span>
                        <strong>{label}</strong>
                        <small>{progress}%</small>
                      </span>
                      <div
                        role="progressbar"
                        aria-label={label}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={progress}
                      >
                        <i style={{ width: `${progress}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
