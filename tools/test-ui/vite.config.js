import { Readable } from "node:stream";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

async function readRequestBody(req, hasBody) {
  if (!hasBody) {
    return undefined;
  }

  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

function buildRequestOptions(method, headers, body) {
  return {
    method,
    headers,
    body,
    duplex: body ? "half" : undefined,
  };
}

async function fetchWithLocalhostFallback(target, method, headers, body) {
  const requestOptions = buildRequestOptions(method, headers, body);

  try {
    return await fetch(target, requestOptions);
  } catch (error) {
    const targetUrl = new URL(target);
    if (targetUrl.hostname !== "localhost") {
      throw error;
    }

    targetUrl.hostname = "127.0.0.1";
    try {
      return await fetch(targetUrl.toString(), buildRequestOptions(method, headers, body));
    } catch (fallbackError) {
      fallbackError.message = [
        fallbackError.message,
        `Original target: ${target}`,
        `IPv4 fallback target: ${targetUrl.toString()}`,
        "Hint: check that the service is running and try 127.0.0.1 instead of localhost in UI settings.",
      ].join("\n");
      throw fallbackError;
    }
  }
}

function formatProxyError(error, target) {
  if (!(error instanceof Error)) {
    return String(error);
  }

  const cause = error.cause instanceof Error ? `\nCause: ${error.cause.message}` : "";
  return `Proxy fetch failed for ${target}\n${error.message}${cause}`;
}

function dynamicProxyPlugin() {
  return {
    name: "gmart-dynamic-proxy",
    configureServer(server) {
      server.middlewares.use("/__gmart_proxy", async (req, res) => {
        try {
          const incoming = new URL(req.url ?? "", "http://localhost");
          const target = incoming.searchParams.get("url");

          if (!target) {
            res.statusCode = 400;
            res.end("Missing url query parameter");
            return;
          }

          const headers = new Headers();
          for (const [name, value] of Object.entries(req.headers)) {
            if (!value) continue;
            const lower = name.toLowerCase();
            if (
              lower === "host" ||
              lower === "connection" ||
              lower === "content-length" ||
              lower === "origin" ||
              lower === "referer"
            ) {
              continue;
            }
            headers.set(name, Array.isArray(value) ? value.join(", ") : value);
          }

          const method = req.method ?? "GET";
          const hasBody = !["GET", "HEAD"].includes(method);
          const body = await readRequestBody(req, hasBody);
          const upstream = await fetchWithLocalhostFallback(target, method, headers, body);

          res.statusCode = upstream.status;
          upstream.headers.forEach((value, name) => {
            if (name.toLowerCase() !== "content-encoding") {
              res.setHeader(name, value);
            }
          });

          if (!upstream.body) {
            res.end();
            return;
          }

          Readable.fromWeb(upstream.body).pipe(res);
        } catch (error) {
          server.config.logger.error(error);
          res.statusCode = 502;
          res.setHeader("content-type", "text/plain; charset=utf-8");
          res.end(formatProxyError(error, target ?? "unknown target"));
        }
      });
    },
  };
}

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/test-ui/" : "/",
  plugins: [react(), dynamicProxyPlugin()],
}));
