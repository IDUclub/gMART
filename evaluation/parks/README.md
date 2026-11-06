# Live parks verification

Run from the repository root with configured `.env.agents`. This is an opt-in
evaluation that creates a real ParkPlanner task. `--chat-storage` additionally
creates a real test chat with the configured service identity. It uses real Redis,
HTTP providers and LLM responses; there are no stubbed model or generator responses.
Output may contain chat content and stays local; do not publish credentials.

```powershell
docker run --detach --name gmart-parks-live-redis --publish 127.0.0.1:16385:6379 redis:7 redis-server --appendonly yes
.venv/Scripts/python.exe -m evaluation.parks.run_live --out tmp/parks-live/full --model YOUR_VISION_MODEL --base-url http://YOUR_MODEL_SERVER/v1 --chat-storage
```

Use `--think-mode chat_template --think-effort low` to match the production
settings verified on 2026-11-06 when the local `.env.agents` still uses
`reasoning_effort`. These overrides only apply to the evaluation process.

The evaluation invokes the real gMART service pipeline in process. It checks
drafting, a natural-language edit, absence of generation before confirmation,
one submission after confirmation, GeoJSON delivery, ChatStorage structured
parts, analysis, and a new service instance reading the Redis journal. It does
not exercise a browser or the HTTP authentication boundary. A successful analysis
of this upstream's GeoJSON/TEP does not by itself prove image understanding: the
current upstream result contract supplies no images.

`--resume-task PATH` exercises recovery without submitting another park. The file
must contain `task_id`, `created_at` (Unix timestamp), and the upstream `payload`:
`parkGeoJson` (JSON string), `tags`, `locationRequests`. Use a task created for this
evaluation; it must belong to the test operator. Recovery deliberately enters
the real gMART generation checkpoint, then reads the task through the actual
provider API. Missing LLM connectivity makes the command fail rather than report
a successful end-to-end test. Inspect `report.json`, `events.json`, `history.json`.

Do not confuse the configured model name with a verified available model. The
live run on 2026-11-02 reached the external result and ChatStorage, then failed
because the configured LLM endpoint refused connections. See
[the verification report](../../docs/integrations/parks-live-check.md).
