import assert from "node:assert/strict";
import test from "node:test";
import { uid } from "./uid.ts";

const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

function withCrypto<T>(replacement: Partial<Crypto>, body: () => T): T {
  const original = globalThis.crypto;
  // getRandomValues lives on the Crypto prototype, so it has to be carried over explicitly.
  const stub = {
    getRandomValues: original.getRandomValues.bind(original),
    randomUUID: original.randomUUID?.bind(original),
    ...replacement,
  };
  Object.defineProperty(globalThis, "crypto", {
    value: stub,
    configurable: true,
  });
  try {
    return body();
  } finally {
    Object.defineProperty(globalThis, "crypto", {
      value: original,
      configurable: true,
    });
  }
}

test("a secure context delegates to crypto.randomUUID", () => {
  const value = withCrypto(
    { randomUUID: () => "11111111-2222-4333-8444-555555555555" },
    uid,
  );
  assert.equal(value, "11111111-2222-4333-8444-555555555555");
});

test("an insecure context still produces a distinct UUIDv4", () => {
  const values = withCrypto({ randomUUID: undefined }, () => [uid(), uid()]);
  for (const value of values) assert.match(value, UUID_V4);
  assert.notEqual(values[0], values[1]);
});
