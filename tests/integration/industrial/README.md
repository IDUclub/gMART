# Industrial orchestrator acceptance

The suite adapts the five partner DOCX scenarios to existing agents. Prepared
project versions replace the unavailable GenPlanner, GenBuilder and PzzCompare
functions. The orchestrator and specialists use the real configured vLLM; real
MCP/calculation services execute their decisions. No test prescribes agent order
or provides model decisions or canned calculation results.

## Control data

`control.py` serves immutable synthetic Urban inputs through `sources.py` (REST)
and `server.py` (MCP). Both read the same records. Unknown routes/IDs fail and
writes are unsupported. There is no passthrough to production Urban API.

The control project is 910, context territory 911, base scenario 91000. Every
version has one residential building, school, kindergarten, parking site and
protected park. The context has 1000 residents and school/kindergarten capacities
100/50. Demand norms are 100/50 places per 1000 residents, with 1000 m availability.
The context is spatially separate and balanced. Calculation summaries include
both project and context; the expected totals explicitly account for this.

| Scenario | Version IDs | Population | School deficit | Kindergarten deficit |
|---|---|---:|---:|---:|
| Redevelopment | 91001 | 12000 | 400 | 200 |
| Preinvestment | 91002 | 4500 | 150 | 75 |
| Social infrastructure before/after | 91003 / 91004 | 25000 | 700 / 0 | 350 / 0 |
| Master plans A/B/C | 91007 / 91008 / 91009 | 8000 | 200 / 0 / 300 | 100 / 0 / 150 |
| Revision before/after | 91005 / 91006 | 12000 | 400 / 0 | 200 / 0 |

`preflight.py` contains independently worked literal capacity/demand/deficit
expectations. It calls the real ObjectEffectsAPI and checks returned layers.
Expectations are not generated from actual service responses.
Before calculations it verifies the control MCP through the application's actual
UrbanMcpClient, including structured catalogue, table and geometry responses.

The spatial check uses synthetic document LOCAL SDK TEST, version 2026, clause
1.1, school-to-open-parking distance >= 50 m. Seed it with the existing
`tests/integration/local_stack/smoke.py --mode seed-documents` setup command.
DVD and NormGraph remain real services. This is an artificial test norm, not a
statement about applicable planning law.

## Execution

Use the repository Python environment. Run from the checkout root. The existing
local stack env file contains auth secrets and is never copied into reports.

```powershell
docker compose --env-file <local-stack.env> -f tests/integration/local_stack/compose.yaml -f tests/integration/industrial/compose.yaml up -d --build
python -m tests.integration.industrial.preflight --env-file <local-stack.env> --output <new-preflight-directory>
python tests/integration/local_stack/harness.py --suite industrial --mode full --env-file <local-stack.env> --output <new-series-directory>
```

The overlay changes all Urban source URLs, including DVD's scenario-to-project
lookup. Preflight searches the seeded clause through DVD with scenario scope,
exercising both its Urban dependency and the remote embedding endpoint.
It also exercises the actual compliance dispatcher with a quoted canonical
distance, persisted NormGraph plans, Redis and IDU geometry for before/after
scenarios. This probe must use no model and must check exactly the scoped clause.
Goal formation uses medium reasoning and allows up to 16384 output tokens.
Ready analytical synthesis keeps high reasoning; the separate judge uses medium.
The test vLLM advertises a 65536-token model window; this overlay uses that bound,
800000 total tokens, 120 model calls and 160 tool calls with the existing 600-second
deadline. Bounds are admission ceilings, not mandatory expenditure.
Model/embedding endpoints,
specialists, calculators, storage and authorization remain on the configured
local stack. `--episodes 1` runs a diagnostic episode and **cannot** report overall
acceptance. Restore real Urban integration by running compose without the overlay
and recreating agents, idu_mcp and effects. Do not use the old 772 acceptance with
the control-source overlay: it deliberately has no scenario 772.

## Acceptance and audit

Five scenarios × three independent prompt formulations = fifteen episodes.
Each episode contains two user turns in the same chat through the public API.
The `continue_from` API remains separate regression coverage for resuming the same
interrupted goal; a new user request is submitted with `chat_id`.
The second turn adds a prepared version, a spatial check, or a comparison goal;
the test does not specify which specialist to call first.

Each series records source/harness hashes, Git HEAD, all stack image IDs and
environment hashes, model endpoint/name, requests, timings, SSE events, complete
persisted contexts, artifact checks, exact request replay and an independent
text evaluation. An existing output directory is rejected. Changed builds abort
the series. Failed attempts remain in their original reports.

Code validates numbers in scenario scope, complete tables, valid WGS84 layers,
source geometry/identity, compliance coverage, normative source identity/version,
and evidence references. The separate LLM evaluation checks relevance, grounding,
limitations, conversation continuity and scenario-specific conclusions. Passing
criteria require an exact answer quote and existing evidence IDs; uncertainty or
malformed judgments produce `needs_review`, never success.
One bounded repair of invalid JSON/references is allowed; substantive negative
verdicts survive a repair. All raw attempts and reference maps are saved.

Successful analysis can conclude that a project is unsuitable. Missing-data and
service blockers do not satisfy a positive scenario. The existing deterministic
unit suite, contract probes and 772 live series remain separate failure/regression
coverage. A passing calculation preflight is not orchestrator acceptance.

Source failures can be injected with `INDUSTRIAL_SOURCE_FAULTS` on the control
source container: a JSON list of exact `path`, `kind` and optional `times` fields.
Kinds: `unavailable` (503), `unauthorized` (401), `missing_normative` (requires
`service_type_id`), `truncated` (an explicitly incomplete FeatureCollection).
Both HTTP and MCP use the same fault boundary. No writes to real Urban API are
needed. Fault profiles belong to separate runs; they must not be enabled for the
positive fifteen-episode acceptance.

The first full measurement and subsequent fixes are documented in
`docs/diagnostics/2026-09-14-industrial-harness.md`. The measured orchestrator did
not meet 15/15; passing infrastructure checks are not a claim of analytical
stability. Source layers require their matching complete tables. Compliance
requires complete passed/violated geometry tied to the correct source version,
not just the expected violation count. Both the outer harness and the live
runner reject existing output directories.
