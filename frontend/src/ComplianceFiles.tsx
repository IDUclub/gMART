import { useRef, useState } from "react";
import {
  FileArrowUp,
  SpinnerGap,
  Trash,
  WarningCircle,
} from "@phosphor-icons/react";
import {
  deleteComplianceUpload,
  setComplianceUploadRoles,
  uploadComplianceFiles,
} from "./api";
import type { ComplianceUpload, Settings, UploadLayerRole } from "./types";

type Props = {
  settings: Settings;
  getToken: () => Promise<string>;
  uploads: ComplianceUpload[];
  onChange: (uploads: ComplianceUpload[]) => void;
};

const ROLE_LABELS: Record<UploadLayerRole, string> = {
  project: "Проект — проверяется",
  context: "Окружение — контекст",
  border: "Граница проекта",
};

/**
 * Files of the user's design for the compliance agent: upload, review the detected
 * layer roles and send the upload ids with the next question.
 */
export default function ComplianceFiles({
  settings,
  getToken,
  uploads,
  onChange,
}: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setError("");
    try {
      const created = await uploadComplianceFiles(
        settings,
        await getToken(),
        Array.from(files),
      );
      onChange([...uploads, created]);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  }

  async function setRole(uploadId: string, layer: string, role: UploadLayerRole) {
    setError("");
    try {
      const updated = await setComplianceUploadRoles(
        settings,
        await getToken(),
        uploadId,
        { [layer]: role },
      );
      onChange(uploads.map((item) => (item.upload_id === uploadId ? updated : item)));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function remove(uploadId: string) {
    setError("");
    try {
      await deleteComplianceUpload(settings, await getToken(), uploadId);
    } catch (e) {
      // An expired upload is gone already; drop it from the list anyway.
      setError((e as Error).message);
    }
    onChange(uploads.filter((item) => item.upload_id !== uploadId));
  }

  const layers = uploads.reduce((total, item) => total + item.layers.length, 0);
  return (
    <section className="document-library compliance-files open">
      <div className="document-panel-toggle">
        <span className="document-panel-icon">
          <FileArrowUp />
        </span>
        <span className="document-panel-title">
          <strong>Файлы проекта для проверки</strong>
          <small>
            GeoJSON, ZIP (GeoJSON или Shapefile), GeoPackage · до 50 МиБ · хранятся
            24 часа
          </small>
        </span>
        <span className={`document-feedback ${error ? "error" : ""}`}>
          {busy ? (
            <>
              <SpinnerGap className="spin" /> Загружаю…
            </>
          ) : error ? (
            <>
              <WarningCircle /> {error}
            </>
          ) : uploads.length ? (
            `${uploads.length} загрузок · ${layers} слоёв — будут проверены`
          ) : (
            "Без файлов проверяется сценарий"
          )}
        </span>
        <button
          type="button"
          className="compliance-files-add"
          disabled={busy}
          onClick={() => input.current?.click()}
        >
          Добавить файлы
        </button>
        <input
          ref={input}
          type="file"
          multiple
          hidden
          accept=".geojson,.json,.zip,.gpkg"
          onChange={(event) => void upload(event.target.files)}
        />
      </div>
      {!!uploads.length && (
        <div className="compliance-files-list">
          {uploads.map((item) => (
            <div className="compliance-upload" key={item.upload_id}>
              <header>
                <strong>{item.files.join(", ")}</strong>
                <button
                  type="button"
                  title="Убрать загрузку"
                  onClick={() => void remove(item.upload_id)}
                >
                  <Trash />
                </button>
              </header>
              {item.layers.map((layer) => (
                <label className="compliance-layer" key={layer.name}>
                  <span>
                    {layer.name}
                    <small>
                      {layer.feature_count} об. · {layer.geometry_types.join(", ")}
                      {layer.warnings.length ? ` · ${layer.warnings.join(", ")}` : ""}
                    </small>
                  </span>
                  <select
                    value={layer.role}
                    onChange={(event) =>
                      void setRole(
                        item.upload_id,
                        layer.name,
                        event.target.value as UploadLayerRole,
                      )
                    }
                  >
                    {(Object.keys(ROLE_LABELS) as UploadLayerRole[]).map((role) => (
                      <option key={role} value={role}>
                        {ROLE_LABELS[role]}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
