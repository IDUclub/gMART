# Real planning acceptance on dev

The target is five partner scenarios, three independent two-turn conversations
per scenario (15 episodes). The minimum requested milestone is 8 passed episodes;
full acceptance requires 15. Runs use actual dev models and MCP services. Fixtures,
precomputed replacement plans and direct tool probes do not count.

The cases cover industrial redevelopment, preinvestment capacity, education
infrastructure, comparison of three masterplans and revision after user feedback.
Their prompts and criteria are in `scenarios.py`. The default scenario is 772;
use a different scenario only after verifying it belongs to the same project.

Deploy the matching gMART revision and the ObjectEffectsAPI variant operation,
then configure the three optional MCP URLs and PZZ REST URL. Run from gMART:

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
not a fingerprint of every downstream service; record the GitOps image manifests
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
