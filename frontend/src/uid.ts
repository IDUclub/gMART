/**
 * Random identifier for client-side objects (layers, status entries, browser-only messages).
 *
 * `crypto.randomUUID` is exposed only in a secure context (https, or localhost). The stand
 * is served over plain http, where the method is missing and the call throws — taking the
 * whole workspace down through the error boundary. `crypto.getRandomValues` has no such
 * restriction, so an insecure context falls back to assembling a UUIDv4 by hand.
 */
export function uid(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-");
}
