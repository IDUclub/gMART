# Goal-driven orchestrator trial — 2026-09-13

Branch: `feat/goal-driven-orchestrator`, based on local integration fix `8770e11`.
The user approved replacing the mandatory detailed plan with a goal, acceptance
criteria and selection of one next action. This change is in gMART; the existing
Effects missing-normative fix and local service configurations remain in use.

## Implementation

- A goal records the original objective and immutable per-result requirements:
  source quote, specialist, scenario, entity type and required artifacts.
  The model selects numbered request fragments; application code binds their
  original text, avoiding a fragile requirement to reproduce exact quotations.
- The model chooses one action by `requirement_id`. The application binds it to
  the requirement's specialist/scenario. Supporting research can collect inputs
  without satisfying the final calculation requirement.
- Completion requires confirmed artifacts from the corresponding operation.
  Typed entity retrieval also checks type labels, table completeness, valid
  collection type and matching row/feature counts. Provision requires a table.
- Failed requirements retain their actionable blocker while independent work
  continues. Complete typed selections produce a count-comparison table using
  application arithmetic, even if another requirement is blocked.
  The model cannot declare missing data before attempting available requirements;
  a specialist must establish the blocker. This also rejects premature `blocked`,
  not only premature `completed` decisions.
- Goals, attempts and evidence are persisted in Redis and ChatStorage. Explicit
  continuation retains successful requirements and reopens failed attempts.
  New criteria use a new request in the same chat without `continue_from`.
- `ORCHESTRATOR_ANALYSIS_MODE=goal` is the default; `plan` remains as the baseline
  for comparison. Goal extraction defaults to `medium`; analytical decisions use
  `high`, with the existing bounded repair/fallback to `medium`.
- The existing shared budget, context paging, artifact transport and SSE replay
  are retained. `orchestrator_final.goal` exposes individual requirement status.

## Validation

957 unit tests passed with dummy service-auth configuration. The existing
POSIX-only `test_workspace_store.py` was excluded on Windows. New regression
coverage includes false completion, duplicate actions, partial/wrong-type evidence,
scope changes, independent work after failure, repeated continuation, supporting
research, budget exhaustion and SDK repair of the obsolete plan-shaped response.
Black and isort passed through `pre-commit run --all-files`.

The local eleven-service stack uses the requested generation endpoint
`http://10.32.11.27:8001/v1`, model `gpt-oss-20b`. Embedding configuration remains
`http://a.dgx:8010/v1/embeddings`, `ai-sage/Giga-Embeddings-instruct`.
Real scenario data were read; no scenario mutations or normative edits were made.

## Live development runs

These runs used evolving code and are not a statistical comparison of two fixed builds.

| Run | Request ID | Result |
| --- | --- | --- |
| 1 | `a9720393-5617-4bab-858b-19f6c17d913f` | Goal extraction with high reasoning exhausted output before returning JSON; no tools ran. Extraction changed to medium. |
| 2 | `b56d3b7a-fe9a-4d56-be7b-9c561530b675` | Goal captured all three results, but decisions omitted the top-level requirement ID. Removed the ambiguous nested step structure. |
| 3 | `982dc3c2-6696-44e0-a4b4-0f072fa53c25` | Goal extraction still failed validation after bounded repair; no tools ran. Isolated subsequent goal/action check passed. |
| 4 | `1bb26eb7-e47d-46ec-82a8-36a204b42471` | Full acceptance passed: 3 schools, 3 kindergartens, both full tables/layers, actual provision attempt, missing-normative blocker, exact persisted payloads and terminal replay. 153 s, 13 model calls, 9 tool calls, 60,159 charged tokens, one reasoning fallback. |
| 5 | `f9319837-b982-4a13-8ce0-6c83ad641886` | Count-summary implementation passed the same acceptance and also returned deterministic counts/comparison plus domain recovery text. 92 s, 11 model calls, 9 tool calls, 44,229 charged tokens, one fallback. |
| 6 | `0ec7eb37-65bc-47d8-b256-30914fd26cb2` | Same build as run 5 failed exact-source-quotation validation during goal extraction. Replaced model-copied quotations with application-bound numbered source fragments. |
| 7 | `2fe2ca3b-aaf8-4e5f-827a-da0f2a15579b` | Source-reference implementation passed full acceptance, including the persisted count-comparison table. 231 s, 14 model calls, 10 tool calls, 77,024 charged tokens, two fallbacks. |
| 8 | `d724456f-73b4-4e3a-9e31-bc88c799289d` | Same build returned 3+3 with artifacts, but the model assumed missing norms before attempting provision. Acceptance correctly failed. Extended the completion guard to reject unsupported `blocked` decisions before available specialist checks. |
| 9 | `71a73137-0575-4ef8-9bf3-e1c991a4203b` | Final build passed full acceptance: goal statuses, actual missing-normative result, both complete 3-row tables/3-feature layers, verified count-comparison table, exact ChatStorage payloads and terminal replay. 134 s, 9 model calls, 9 tool calls, 42,176 charged tokens, no fallback. |

Run 4 also exposed a presentation omission: the final prose only discussed the
normative although the counts were available in artifacts. The final implementation
adds a deterministic count summary and comparison table, and replaces the raw
HTTP exception in the blocker with the domain recovery action.

Local traces and exported artifacts are under `output/local-sdk-stack/goal-772-run*/`
in the parent gMART workspace and are not committed.
The final build therefore has one fresh full acceptance pass after the last guard
change. Earlier passes used preceding builds; they must not be presented as a
consecutive success series for the final build. All five local API health checks
returned HTTP 200 after the final restart.

## Limits

A missing normative is an expected blocked calculation, not a successful provision
score. The service reports territory 58 and service type 22: a relevant availability
radius/time and provision norm must be configured, or a territory with a normative
must be selected. Existing objects do not establish provision by themselves.

The model still interprets natural-language requirements and evaluates qualitative
conclusions. This trial does not prove universal semantic coverage, causal validity,
or a production success rate. Repeatable evaluation across fixed datasets and
controlled backend failures is still needed before claiming broad stability.
