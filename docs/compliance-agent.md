# Compliance agent: phase-one contract

The phase-one compliance agent reuses the existing MCP tools and executes this chain:

1. `search_restrictions` or `restrictions_applicable` in NormGraph;
2. `GetServices` / `GetPhysicalObjects` in IDU MCP;
3. `CreateBuffers`;
4. `CreateRestrictions` with the inclusive `intersects` predicate.

Only canonical NormGraph restrictions with a positive distance in metres and a
supported spatial kind are executable. A distance explicitly provided by the user
is represented as a request-scoped rule with `origin=user`; it is not written back
to NormGraph.

## HTTP API

`GET /compliance/check/stream` accepts the same query parameters and authentication
as `GET /restrictions/generate_restrictions/stream`. When `NORM_GRAPH_MCP_URL` is
configured, applicable restrictions are retrieved before the GIS plan is built.
Without it, the endpoint can still execute an explicit request-scoped distance rule.

The SSE stream returns the affected-object and generator layers as GeoJSON. Every
affected feature keeps its original layer properties and adds:

- `source_layer` — target layer name;
- `object_ref` — composite identifier, namespace, source IDs, layer and display name;
- `restriction_name` and `restriction_description` — compatibility fields;
- `restriction_evidence` — one record per matching generator/rule, including the
  intersection operation, inclusive-boundary policy, buffer distance, generator
  `object_ref`, NormGraph restriction ID and provenance.

The current replay model stores MCP tool calls and reruns them against current
scenario data. Exact historical result snapshots are intentionally deferred; see
the TODO in `RestrictionParserService` before implementing snapshot persistence.

## Deferred script-generation phase

Ollama-generated GeoPandas scripts are outside phase one. Their future executor
must accept immutable layer metadata plus request-scoped rules, validate generated
Python without showing it to non-programmer users, run it in an isolated process or
container with resource/time/network limits, and return only validated GeoJSON plus
structured evidence.
