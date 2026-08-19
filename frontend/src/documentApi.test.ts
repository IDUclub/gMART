import assert from "node:assert/strict";
import test from "node:test";
import { userDocumentPath, userDocumentScopeParams } from "./api.ts";

test("user document scope includes trimmed scenario and project", () => {
  const query = userDocumentScopeParams(" 772 ", " 4242 ");
  assert.equal(query.get("scenario_id"), "772");
  assert.equal(query.get("project_id"), "4242");
});

test("user document scope omits empty values", () => {
  assert.equal(userDocumentScopeParams(" ", "").toString(), "");
});

test("user document mutation path safely encodes its name", () => {
  assert.equal(
    userDocumentPath("Регламент / версия 2"),
    "/documents/user-documents/%D0%A0%D0%B5%D0%B3%D0%BB%D0%B0%D0%BC%D0%B5%D0%BD%D1%82%20%2F%20%D0%B2%D0%B5%D1%80%D1%81%D0%B8%D1%8F%202",
  );
});
