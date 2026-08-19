import assert from "node:assert/strict";
import test from "node:test";
import {
  emptyDocumentPipelineProgress,
  withJobProgress,
  withUploadProgress,
} from "./documentProgress.ts";

test("tracks browser upload independently from server stages", () => {
  const progress = withUploadProgress(emptyDocumentPipelineProgress(), 64);
  assert.equal(progress.upload, 64);
  assert.equal(progress["structure-markup"], 0);
});

test("completes earlier stages and advances the current stage", () => {
  const progress = withJobProgress(emptyDocumentPipelineProgress(), {
    job_id: "job-1",
    status: "processing",
    stage: "hierarchy",
    stage_index: 3,
    stage_total: 7,
    task_progress: 42,
    overall_progress: 48,
  });

  assert.equal(progress.upload, 100);
  assert.equal(progress["structure-markup"], 100);
  assert.equal(progress["type-tagging"], 100);
  assert.equal(progress.hierarchy, 42);
  assert.equal(progress.identity, 0);
});

test("marks every stage complete on terminal success", () => {
  const progress = withJobProgress(emptyDocumentPipelineProgress(), {
    job_id: "job-1",
    status: "done",
    overall_progress: 100,
  });
  assert.ok(Object.values(progress).every((value) => value === 100));
});
