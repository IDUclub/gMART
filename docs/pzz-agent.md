# PZZ agent

The PZZ agent follows the auto-check workflow from
[PzzCompareAPI](https://github.com/IDUclub/PzzCompareAPI/blob/main/service/api/classifier.py):
identify input columns, submit classification, monitor the task, retrieve the report,
and stream a grounded answer. gMART owns the LLM answer, pipeline state, chat history,
SSE and A2A lifecycle. Classification remains in PZZ.

## Configuration

```dotenv
PZZ_MCP_SERVER=http://10.32.11.90:31053/mcp
PZZ_API_URL=http://10.32.11.90:31009
```

Both settings are optional for application startup. The orchestrator includes `pzz`
only when `PZZ_MCP_SERVER` is configured. REST and MCP are separate deployments;
**do not derive the REST address by removing `/mcp`**. They must point to the same
PZZ task/upload storage. REST provides uploads and the classify-only summary, which
the MCP catalogue does not expose. Calls use the existing service-token transport
and `X-User-Id` convention. GitHub deployment variables are forwarded by compose.

## REST / SSE

- `GET /pzz/check/stream`: the standard gMART query parameters (`request`, `model`,
  `temperature`, `scenario_id`, `chat_id`, `request_id`). Suitable for scenario queries.
- `POST /pzz/check/stream`: the same fields in JSON plus structured `inputs`.
- `POST /pzz/uploads`: multipart `file`; returns the upstream `upload_id`. Uploads
  are associated with the authenticated caller in gMART; use IDs from this endpoint.
  The upload limit is 50 MiB. GeoJSON is accepted directly; GeoPackage, GML, KML and GeoParquet are converted to WGS84 GeoJSON before upload.
  The building tool performs its own file parsing and auto-detection in PZZ.

Example scenario request:

```json
{
  "request": "Проверь размещение зданий по ПЗЗ",
  "scenario_id": 772,
  "inputs": {"mode": "scenario", "year": 2026, "source": "PZZ"}
}
```

Example file request after uploading cadastral and zone GeoJSON files:

```json
{
  "request": "Какие участки не соответствуют ПЗЗ?",
  "inputs": {
    "mode": "pzz_check",
    "cadastral_upload_id": "<id returned by /pzz/uploads>",
    "pzz_zones_upload_id": "<id returned by /pzz/uploads>"
  }
}
```

The three original file modes are supported:

| Mode | Inputs | Report |
| --- | --- | --- |
| `pzz_check` | cadastral and zone GeoJSON, inline or upload IDs | `object_zone_fit` |
| `classify_only` | cadastral GeoJSON, inline or upload ID | `classify_summary` |
| `building_pzz_check` | `buildings_upload_id`, `pzz_zones_upload_id`, optional `descriptions_upload_id` | `object_zone_fit` |

Inline layers use `cadastral_geojson` and `pzz_zones_geojson`. Columns are detected
from exact known aliases, then with an LLM restricted to actual property names.
Explicit overrides are `cadastral_vri_col`, `pzz_zone_code_col`, `pzz_zone_name_col`.
If a required column is unresolved, no task is submitted.

Building checks may return `confirm`, `suggest_upload` or `detection_failed`.
These become `clarification` events, including the upstream suggestions. Send a new
request with `confirmed_zone_map` only after the user approves that mapping; the
LLM never creates this field. A new clarification answer starts a new pipeline;
`request_id` reconnects the previous pipeline without changing its inputs.

Additional fields: `group_by` (`zone` or `object`), `priority` (1–10),
`force_recompute`, and scenario `physical_object_type_id` (default 4). For scenarios,
`year` and `source` must be supplied or extracted from the request/history; neither
is silently defaulted. Custom cadastral references use `labels_upload_id` and `classifier_upload_id`.
For CSV/XLSX zone descriptions, upload with `kind=zone_descriptions`: the PZZ
converter produces the same label JSON used by its auto flow. Custom references
are submitted through the equivalent REST task endpoint (MCP does not expose
these optional fields); monitoring and results still use MCP.

SSE uses the existing `{ "type": ..., "content": ... }` envelope:
`pipeline_started`, `service_event`, `status`, `tool_call`, `clarification`,
`object_zone_fit` / `classify_summary`, `feature_collection` (file result),
`chunk`, `warning`, `error`. Chunks carry `iteration`; replace an earlier answer
when a resumed answer has a higher iteration. A final `chunk.content.done=true`
marks a successful answer. Errors never become a successful classification.

The pipeline stores its external task ID and reconnect events in Redis. Reconnect
with `request_id` resumes polling that task; finished runs only replay their events.
Concurrent execution of one request is serialized. Polling waits up to ten minutes
per connection. An interrupted submission with no confirmed task ID is not retried
blindly: the caller receives a clarification to check PZZ task state first.

## A2A

- `GET /pzz/.well-known/agent-card.json` (legacy alias `/pzz/agent.json`).
- `POST /pzz/a2a`: the same JSON-RPC methods and streaming aliases as other agents.

```json
{
  "jsonrpc": "2.0",
  "id": "pzz-1",
  "method": "message/stream",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {"kind": "text", "text": "Проверь здания по ПЗЗ"},
        {"kind": "data", "data": {
          "scenario_id": 772,
          "inputs": {"mode": "scenario", "year": 2026, "source": "PZZ"}
        }}
      ]
    }
  }
}
```

Responses include a Task, status updates, text/data artifacts, and a terminal status.
Clarification yields `input-required`, including structured zone suggestions.
A2A and orchestrator sub-agent calls pass `persist_history=false`; only the outer
orchestrator persists its combined response.

## Orchestrator

`pzz` is part of the planner catalogue and dispatches the same `PzzService` in-process.
Existing `GET /orchestrator/route/stream` works for scenario questions. For file
references or explicit PZZ parameters, use `POST /orchestrator/route/stream` with
standard request fields and `pzz_inputs` (the same schema as `inputs` above).
These fields go directly to PZZ and are not generated by the planner. The planner
receives a compact manifest (mode, layer presence, year and source), without file
contents or upload IDs. File modes do not require a year/source even when a scenario
is selected. PZZ validates required inputs before submitting a task and returns
`clarification` for missing layers. A file request without attachments remains a
file request; the selected scenario does not silently switch it to Urban API data.
PZZ events are wrapped in `step_event`; a clarification stops dependent steps.
