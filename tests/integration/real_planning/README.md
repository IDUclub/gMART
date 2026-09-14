# Real planning acceptance through local services

The target is five partner scenarios, three independent two-turn conversations
per scenario (15 episodes). The minimum requested milestone is 8 passed episodes;
full acceptance requires 15. Runs use the local updated services and actual remote models and MCP services. Precomputed replacement plans and direct tool probes do not count. The user explicitly
authorized mocks for missing normative inputs. `mock_normatives.json` supplies school
(100 places/1000 residents, 15 minutes) and kindergarten (60 places/1000 residents,
10 minutes) test values only where Urban API has no normative for that service.
These illustrative values have no legal authority. Responses retain fixture identity
and SHA256. Geometry, generation and ObjectNat calculations remain real.

The cases cover industrial redevelopment, preinvestment capacity, education
infrastructure, comparison of three masterplans and revision after user feedback.
Their prompts and criteria are in `scenarios.py`. The default scenario is 772;
use a different scenario only after verifying it belongs to the same project.

Build and start the updated gMART Agents/IDU MCP, ObjectEffectsAPI and NormGraph
locally. From the ICII workspace root:

```bash
docker compose --env-file integration/service-auth.env -f gMART/tests/integration/real_planning/compose.yaml up -d --build
```

This isolated Compose project exposes localhost ports 18000 (Agents/UI), 18002
(IDU MCP), 18080 (Effects) and 18020 (NormGraph). It creates its own Redis and
Neo4j data volume. It connects to the existing local ChatStorage on port 8010;
`CHAT_STORAGE_ORIGIN` can override that address. DVD, Urban, the three planning
services and inference remain external. Kafka is unconfigured. NormGraph starts
with an empty graph; ingest/sync real source documents before claiming graph-based
normative coverage. Existing containers and data volumes are preserved. `PLANNING_SUBNET` overrides the
default 172.31.3.0/24 subnet when it overlaps another local route.

The multi-specialist stand allows 1800 seconds, 120 model calls, 160 tool calls,
24 orchestration steps and one million total tokens per turn. All specialists
share these limits. Planning action selection defaults to low reasoning;
`PLANNING_REASONING_EFFORT` overrides it. Timings and usage remain assessment data;
these execution limits do not change the output acceptance criteria.

All generative model calls use `http://10.32.11.27:8001/v1`. This stand enables
`OPENAI_STRUCTURED_TRANSPORT=responses_function`: schema-based model stages return
their decision through a single output-envelope function on `/v1/responses`.
The server is given no domain tools to execute. The SDK validates the returned
arguments before the existing application executor can call an MCP tool. Raw
text and streaming model calls continue to use `/v1/chat/completions` on the same
server. Other deployments default to `OPENAI_STRUCTURED_TRANSPORT=chat`.
The runner allows 2100 seconds per HTTP call so its timeout exceeds the configured
1800-second turn budget.

Run the acceptance runner from gMART:

```bash
uv run python -m tests.integration.real_planning.runner \
  --auth-env ../integration/auth.env \
  --output dev/real-planning/acceptance-unique-name
```

The auth file contains the existing helper URL/key and test-account credentials.
Credentials and bearer tokens are not copied into the result directory. Keep
that directory private and out of source control. A new output directory is
required for every series so failed attempts cannot be silently overwritten.

The runner verifies the deployed application content hash against the checkout
before executing and checks the application/configuration identity between
turns. `/system/build-info` is authenticated and returns hashes only. This is
not a fingerprint of every downstream service; record the local Docker image IDs
with an assessment and avoid changing downstream versions during a series.

Every episode saves input requests, raw SSE, parsed events, full confirmed
artifacts, timings, exact-replay results and a separate model review. Source
snapshots include scenario metadata, zoning versions, zone geometry, physical
objects and indicators. Authentication is refreshed between long-running turns.
`report.json` retains failures, aggregate counts and the minimum/full milestones.

Automatic checks cover terminal status, answer/evidence references, full tables,
GeoJSON validity and coordinate range, required specialists, replay and stable
build identity. The separate review requires exact answer quotes and existing
source artifacts; specialist prose alone is not independent proof. Truncated or
oversized evidence produces `needs_review`, never automatic success. Detailed
numeric, demographic, site suitability and legal interpretation still require
technical assessment of the saved source and output artifacts.

A missing norm or legal zone code may justify a **data-limited** result when the
actual source evidence demonstrates the gap. It must be distinguished from a
technical failure, unsupported integration or an unexecuted promise. The runner
is conservative: it does not automatically count partial/data-limited outcomes
as passed. Such outcomes need recorded human assessment; do not change the pass
threshold or rewrite failed attempts to reach the requested count.

`POST /llm/message` accepts the independent review material in a JSON body rather
than a long URL. The evaluator uses a separate context on the configured model;
it is not an independent human expert or a different model family.
