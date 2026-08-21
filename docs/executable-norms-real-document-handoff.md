# Handoff: real-document executable-norm validation

Status captured on 2026-08-20 after the user interrupted the run. No secrets are recorded here.

## Objective

Validate the executable-norm pipeline end to end on a real published regulation:

- document: `СП 42.13330.2016` (revision containing amendments 1 and 2);
- target rule: clause 10.5 / table 10.2;
- expected rule for climate regions II–III: walking accessibility to a general education
  organisation is at most `0.5 km`;
- source PDF:
  `https://edu.mos-gaz.ru/upload/dynamic/2022-03/24/SP_42.13330.2016_Svod_pravil.pdf`;
- source PDF SHA-256:
  `b47fb058bdccee12718fd639387c9a0c5de05769e7ff9cd7f6ca6f2bee2b9cc0`.

The legal text and table were independently cross-checked against the Garant publication and the
Minstroy document catalogue before ingestion.

## Persisted state

IDU_DVD direct ingestion completed successfully:

- document name: `СП 42.13330.2016 — п. 10.5 (валидация)`;
- `doc_id`: `7d13c228-af4e-4a6e-8413-70cd745f0fca`;
- version: `ред. 2020 (Изм. 1-2)`;
- job: `35e7bb00-787b-4116-afc2-e6f5eba5a8d4` (`done`, one node);
- corpus: `federal_sp_validation`;
- stored metadata includes the source URL, PDF page 43, order `1034/пр`, source hash and
  `validation_fixture=true`.

The direct record is in the shared Qdrant/Redis volumes, so it survives application-container
restarts. NormGraph structural ingestion also completed: Neo4j contains one `Document` and one
`Clause` for this `doc_id`.

At handoff time Neo4j contains:

- restrictions for this document: `0`;
- current CheckPlans for this document: `0`.

Therefore the LLM extraction and the gMART geometry execution are the remaining work.

## Interrupted work

Two larger background IDU_DVD jobs were deliberately stopped and marked as errors by startup
cleanup:

- full 142-page document job `e0d4fc75-aad6-4016-adad-426ca9c08c2d` stopped during
  `structure-markup` at 23%; it must be uploaded again if full-document indexing is desired;
- six-page section-10 job `490f6e14-336b-4326-945e-c5ea121b0350` was still queued and must also
  be uploaded again if needed.

The standard NormGraph instance using the configured OpenAI-compatible `gpt-oss-20b` endpoint did
not return the one-clause extraction within several minutes. A temporary NormGraph instance was
then run with native Ollama `qwen2.5:7b-instruct`; model loading completed, but its extraction was
still generating when the user interrupted the run. Both temporary containers were stopped and
removed. The main `idu_dvd-app-1` container was stopped to halt background work.

## Resume point

1. Start IDU_DVD with the integration service-auth environment. The already completed direct
   record does not need to be uploaded again.
2. Confirm `GET /library/documents/7d13c228-af4e-4a6e-8413-70cd745f0fca` succeeds.
3. Run `POST /sync/documents/7d13c228-af4e-4a6e-8413-70cd745f0fca?replace=true` in NormGraph.
   If the primary LLM remains slow, use a temporary instance with
   `NG_LLM_PROVIDER=ollama`, `NG_LLM_MODEL=qwen2.5:7b-instruct`, and a bounded
   `NG_LLM_MAX_TOKENS` (the interrupted retry used 768).
4. Inspect the created restriction and CheckPlan. Expected semantics are `<= 500 m` for school
   walking accessibility in climate regions II–III. A plan that drops the climate condition or
   treats every table row as one unconditional value is incorrect and must not be approved.
5. Review/approve the correct plan, then run gMART compliance against the established scenario
   containing residential buildings and schools (scenario 772 was previously audited).
6. Compare per-feature evidence and summary with an independent 500 m geometry calculation.

All repository worktrees were clean before this handoff file was added. Temporary source files were
created only under `/tmp` and were not added to Git.
