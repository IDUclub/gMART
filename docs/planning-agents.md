# Planning specialists

GenPlanner, GenBuilder and PZZ are independent agents available to the analytical
orchestrator. They run through the same SDK and request budget as the existing
specialists. Each has an optional MCP URL, a separate A2A card and caller-scoped
A2A task store. No planning operation writes a scenario to Urban API.

| Agent | Configuration | REST stream | A2A |
| --- | --- | --- | --- |
| GenPlanner | `GENPLANNER_MCP_SERVER` | `POST /genplanner/run/stream` | `POST /genplanner/a2a` |
| GenBuilder | `GENBUILDER_MCP_SERVER` | `POST /genbuilder/run/stream` | `POST /genbuilder/a2a` |
| PZZ | `PZZ_MCP_SERVER`, `PZZ_API_URL` | `POST /pzz/run/stream` | `POST /pzz/a2a` |

Each card is at `/<agent>/.well-known/agent-card.json` and `/<agent>/agent.json`.
An unset MCP URL disables that specialist in the orchestrator catalogue and gives
HTTP 503 on its execution endpoints. MCP requests use the service credential and
verified `X-User-Id`; scenario context remains a program-controlled argument.

A stream body contains `request`, optional `scenario_id`, `model`, `temperature`
and `input_artifacts`. A2A uses the existing gMART scenario extension; explicit
input artifacts can be passed in request metadata. A2A clarification ends in
`input-required`, downstream errors in `failed`. Stores are currently in memory,
so tasks do not survive worker restarts.

```json
{
  "request": "Generate buildings for 4500 residents using the supplied zones",
  "scenario_id": 772,
  "input_artifacts": {
    "zones": {"type": "FeatureCollection", "features": []}
  }
}
```

The example shows the envelope only: real nonempty polygon geometry is required.
The model references complete data using `{"$artifact":"zones","path":[]}`.
It sees explicit previews rather than coordinates, and cannot submit model-written
geometry. The executor resolves references, validates the actual MCP schema and
retains full results as `source_evidence`, `feature_collection` and `table` events.
`select_layer`, `layer_values` and `summarize_layer` operate over all features,
including rows omitted from a preview. `compare_layer_coverage` measures polygon
area retained in a local metric CRS; compare recreation layers specifically to
verify park preservation. It does not prove attribute or legal-status preservation.

`prepare_building_blocks` maps actual GenPlanner zone labels to GenBuilder's
`properties.zone`. GenBuilder's excluded features carry zeroed numerical fields;
these are not evidence of the existing residents or floor areas. Requested and
achieved population must be distinguished.

For an unsaved variant, the orchestrator passes selected confirmed artifacts to
provision and compliance as well. ObjectEffectsAPI must expose
`CalculateVariantServicesProvision` on its effects MCP endpoint. That operation
adds generated residential buildings and proposed services to the existing
scenario without persisting them. Its population is the **total** scenario
population, including existing residents; demand follows the existing restored
floor-area and gravity model. A baseline calculation is a separate operation.
A missing variant tool is an integration error, never a fallback to baseline.

Compliance retains its authoritative NormGraph CheckPlan and deterministic
geometry templates. Request-local variant layers augment residential buildings
and proposed services and replace supplied functional zones. Missing attributes
remain missing. Residential additions currently use Urban physical type 4;
other generated physical types are not automatically mapped. Compare results and
coverage rather than treating unverified criteria as passed.

PZZ uploads complete files to its REST `/uploads` endpoint and starts/polls the
corresponding MCP task. Upload IDs expire according to the service response.
`confirm`, `suggest_upload`, `detection_failed`, pending jobs and failed jobs are
not successful checks. The agent does not invent a confirmed zone map. The
scenario classifier's approximate functional-type mapping is distinct from a
legal check against actual territorial PZZ codes.

The frontend owns approval and saving scenarios. Results are preliminary design
artifacts, and a proposed service point is a candidate location, not a building
footprint. Integration routes and tests do not add a scenario-saving operation.
