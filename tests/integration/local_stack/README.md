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
Confirm the model ID and vector dimension from the server before starting DVD/NormGraph.

From this directory:

```powershell
docker compose --env-file stack.local.env build
docker compose --env-file stack.local.env up -d
python smoke.py --env-file stack.local.env --output results --mode health
python smoke.py --env-file stack.local.env --output results --mode llm
python smoke.py --env-file stack.local.env --output results --mode artifacts
python smoke.py --env-file stack.local.env --output results --mode effects --scenario 772
```

Run `smoke.py` with the gMART development Python environment. `llm` sends only a synthetic
string. `artifacts` serializes a synthetic table and nonempty GeoJSON using gMART, stores them
through ChatStorage HTTP/MongoDB, and verifies the reopened history payloads. It creates a
test chat in the isolated database. `effects` calls the real local MCP and upstream Urban API;
it expects the known missing-normative case in scenario 772 and does not call the LLM.

`--mode data` and `--mode analysis` send real scenario tool results to the configured LLM.
Use only with authorization for that data transfer. They save SSE traces and verify exact
terminal replay. `--mode documents` expects a previously ingested synthetic document named
`LOCAL SDK TEST`; it does not seed the document or certify full document ingestion by itself.

Ports: Agents/UI 18000, IDU MCP 18002, Effects 18080, ChatStorage 18010, DVD 18100,
NormGraph 18020, Redis 16389, Neo4j Bolt 17687. Urban API/MCP and inference remain external.
This is a hybrid integration stand, not an offline deployment. Stop only this project with
`docker compose --env-file stack.local.env down`.
