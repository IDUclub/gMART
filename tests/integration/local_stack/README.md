# Isolated SDK integration stack

Builds gMART Agents/IDU MCP, ObjectEffectsAPI, IDU_DVD, NormGraph and ChatStorage from
explicit checkouts. Redis, MongoDB, Neo4j, Qdrant and MinIO run in a separate Compose
project/network. Published ports bind only to localhost; existing containers are not stopped.
Kafka is unconfigured, so the stand does not publish into or consume from shared topics.

Copy `stack.env.example` to `stack.local.env`, set checkout paths and credentials, and supply
an OpenAI-compatible embedding endpoint/model/dimension. All generative clients use the same
`LLM_BASE_URL` and `LLM_MODEL`; DVD and NormGraph share one embedding space. The generative
`local-gpu` endpoint is not an embedding server. Do not substitute random/fake embeddings for
an end-to-end document test.

The requested embedding origin is `http://a.dgx:8010`; clients append `/v1/embeddings`.
The hostname must resolve both on the host and inside Docker (corporate DNS/VPN may be needed).
Verified model: `ai-sage/Giga-Embeddings-instruct`, 2048 dimensions. The server returns 404 for
`/v1/models`, but `/v1/embeddings` returns the model ID and vectors successfully.

From this directory:

```powershell
docker compose --env-file stack.local.env build
docker compose --env-file stack.local.env up -d
python smoke.py --env-file stack.local.env --output results --mode health
python smoke.py --env-file stack.local.env --output results --mode llm
python smoke.py --env-file stack.local.env --output results --mode artifacts
python smoke.py --env-file stack.local.env --output results --mode effects --scenario 772
python smoke.py --env-file stack.local.env --output results --mode seed-documents
python smoke.py --env-file stack.local.env --output results --mode documents
```

Run `smoke.py` with the gMART development Python environment. `llm` sends only a synthetic
string. `artifacts` serializes a synthetic table and nonempty GeoJSON using gMART, stores them
through ChatStorage HTTP/MongoDB, and verifies the reopened history payloads. It creates a
test chat in the isolated database. `effects` calls the real local MCP and upstream Urban API;
it expects the known missing-normative case in scenario 772 and does not call the LLM.

`--mode data` and `--mode analysis` send real scenario tool results to the configured LLM.
Use only with authorization for that data transfer. They save SSE traces and verify exact
terminal replay. `analysis` specifically exercises the known missing-school-normative case:
separate school/kindergarten service counts, tables and layers, followed by provision. It requires
an attempted provision calculation, a concrete normative blocker and both complete artifact sets
in reopened history, then exports the verified tables/GeoJSON. It fails if the model omits a task
or returns incomplete output; see `docs/diagnostics/2026-09-13-scenario-772.md` for current results.
`--mode seed-documents` uploads a synthetic document named `LOCAL SDK TEST`
through the direct fragment ingestion API, waits for the durable queue, verifies DVD vector
search, runs NormGraph extraction/search and checks idempotent repeated sync. It leaves the
fixture in the isolated databases. This tests direct ingestion, not file parsing/OCR.
`--mode documents` compares the fixture's source clause and extracted restriction through the
real analytical orchestrator, requiring both specialists to complete, the correct numeric answer,
both evidence references, full artifacts in reopened chat history and exact terminal replay.
It sends only the synthetic fixture to the configured inference services.

Ports: Agents/UI 18000, IDU MCP 18002, Effects 18080, ChatStorage 18010, DVD 18100,
NormGraph 18020, Redis 16389, Neo4j Bolt 17687. Urban API/MCP and inference remain external.
This is a hybrid integration stand, not an offline deployment. Stop only this project with
`docker compose --env-file stack.local.env down`.
